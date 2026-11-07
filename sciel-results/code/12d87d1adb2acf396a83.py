"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Sieve for primes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    current_set = set(primes)
    
    def is_valid(s):
        items = list(s)
        m = len(items)
        for i in range(m):
            a = items[i]
            for j in range(i + 1, m):
                b = items[j]
                for k in range(j + 1, m):
                    c = items[k]
                    if (b * c) % a == 0:
                        return False
        return True

    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            # Try adding x
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                improved = True
                no_improve_count = 0
                random.shuffle(candidates)
                continue
            
            # If addition failed, try a swap
            # Pick a random y from current_set to remove
            # To be more targeted, we could pick y that is "small" or "conflicting"
            # But random is safer for exploration
            if random.random() < 0.1: # 10% chance to try a swap on failure
                if current_set:
                    y = random.choice(list(current_set))
                    swap_set = (current_set - {y}) | {x}
                    if is_valid(swap_set):
                        # Accept swap only if it doesn't reduce size (it doesn't)
                        # And hopefully leads to better state
                        current_set = swap_set
                        improved = True # Mark as changed to continue search
                        no_improve_count = 0
                        random.shuffle(candidates)

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    # Final polish: try all single additions again
    for x in candidates:
        if x in current_set:
            continue
        new_set = current_set | {x}
        if is_valid(new_set):
            current_set = new_set

    return list(current_set)

# EVOLVE-BLOCK-END
