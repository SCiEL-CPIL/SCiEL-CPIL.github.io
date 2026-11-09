import random
import time

N = 256

def construct():
    def greedy_maximal():
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        selected = []
        sums = set()
        for x in candidates:
            valid = True
            if 2 * x in sums:
                valid = False
            else:
                for s in selected:
                    if (x + s) in sums:
                        valid = False
                        break
            if valid:
                selected.append(x)
                sums.add(2 * x)
                for s in selected[:-1]:
                    sums.add(x + s)
        return selected, sums

    best_S = None
    best_size = N + 1
    
    start_time = time.time()
    time_limit = 28.0
    
    # Run multiple local search instances
    while time.time() - start_time < time_limit:
        # Initialize with a greedy maximal set
        S, sums = greedy_maximal()
        current_size = len(S)
        
        if current_size < best_size:
            best_size = current_size
            best_S = S.copy()
            
        # Early exit if we find a very small set
        if best_size <= 10:
            break
            
        # Local search loop
        # We try to remove an element. If the set remains maximal, we keep the removal.
        # If it becomes non-maximal, we add a random valid element (swap) and accept with some probability.
        
        while time.time() - start_time < time_limit:
            v = random.choice(S)
            S_prime = [x for x in S if x != v]
            S_prime_set = set(S_prime) # For fast membership
            
            # Compute sums for S_prime
            sums_prime = set()
            for i in range(len(S_prime)):
                sums_prime.add(2 * S_prime[i])
                for j in range(i + 1, len(S_prime)):
                    sums_prime.add(S_prime[i] + S_prime[j])
            
            # Check maximality of S_prime
            # An element x is "addable" if x not in S_prime and S_prime U {x} is Sidon.
            addable = []
            for x in range(1, N + 1):
                if x in S_prime_set:
                    continue
                can_add = True
                if 2 * x in sums_prime:
                    can_add = False
                else:
                    for y in S_prime:
                        if (x + y) in sums_prime:
                            can_add = False
                            break
                if can_add:
                    addable.append(x)
                    
            if not addable:
                # S_prime is maximal. We successfully reduced the size.
                S = S_prime
                sums = sums_prime
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_S = S.copy()
                    if best_size <= 10:
                        break
            else:
                # S_prime is not maximal. We must add one element to make it maximal.
                # This results in a swap: remove v, add u.
                # Size remains the same.
                # We accept this swap with probability 0.5 to allow exploration.
                if random.random() < 0.5:
                    u = random.choice(addable)
                    S = S_prime + [u]
                    # Update sums
                    sums = sums_prime
                    for y in S_prime:
                        sums.add(u + y)
                    sums.add(2 * u)
                # If we don't accept the swap, we just retry with the same S (next iteration picks new v)
                
    return best_S
