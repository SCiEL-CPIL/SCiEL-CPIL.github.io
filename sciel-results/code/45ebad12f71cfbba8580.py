import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def get_score(seq):
        """
        Returns a score where higher is better.
        Penalizes duplicates heavily.
        """
        products = set()
        n = len(seq)
        num_dups = 0
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    num_dups += 1
                    # Early exit? No, we need count for smooth gradient?
                    # Actually, for SA, just knowing it's invalid is often enough,
                    # but count helps guide away from worse invalid states.
                else:
                    products.add(prod)
        
        # If no duplicates, score is length.
        # If duplicates, score is negative large + length.
        if num_dups == 0:
            return n
        else:
            return -10000 * num_dups + n

    primes = get_primes(128)
    all_nums = list(range(2, 129))
    
    # Initial state: All primes
    current_set = set(primes)
    current_seq = sorted(current_set)
    current_score = get_score(current_seq)
    
    best_seq = current_seq[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    steps_per_temp = 50
    
    while time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            # Generate move
            move_type = random.choice(['add', 'remove', 'swap'])
            new_set = set(current_set) # Copy
            
            if move_type == 'add':
                candidates = [x for x in all_nums if x not in current_set]
                if not candidates: continue
                x = random.choice(candidates)
                new_set.add(x)
            elif move_type == 'remove':
                if not current_set: continue
                x = random.choice(list(current_set))
                new_set.remove(x)
            elif move_type == 'swap':
                candidates = [x for x in all_nums if x not in current_set]
                if not candidates or not current_set: continue
                x_out = random.choice(list(current_set))
                x_in = random.choice(candidates)
                new_set.remove(x_out)
                new_set.add(x_in)
            
            new_seq = sorted(new_set)
            new_score = get_score(new_seq)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                current_set = new_set
                current_seq = new_seq
                current_score = new_score
            else:
                # Prob = exp(delta / T)
                # delta is negative
                if T > 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_seq = new_seq
                        current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
                
        T *= cooling
        if T < 1.0:
            T = 1.0
            # Maybe restart or cool faster?
            # Just keep going with T=1 for local search
            
    return best_seq
