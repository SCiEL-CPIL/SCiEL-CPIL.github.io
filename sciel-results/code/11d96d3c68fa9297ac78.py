import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We should return well before that.
    # Let's aim for 55s to be safe, allowing for any slow final checks.
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        if n >= 0:
            sieve[0] = False
        if n >= 1:
            sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def is_valid(seq):
        """
        Fast check: return True if all consecutive block products are distinct.
        Early exits on first duplicate.
        """
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def get_score(seq):
        """
        Returns length if valid, else a large negative number.
        """
        if is_valid(seq):
            return len(seq)
        return -10000

    primes = get_primes(128)
    all_nums = list(range(2, 129))
    
    # 1. Greedy Initial Construction
    # Start with primes
    current_set = set(primes)
    current_seq = sorted(current_set)
    current_score = len(current_seq)
    
    # Try to add composites
    # Sort composites: numbers with fewer prime factors are "safer" (less likely to cause collisions)
    # But let's just try in increasing order first, it's simple and often works well.
    # Or sort by number of prime factors to minimize collision probability.
    
    def num_prime_factors(x):
        count = 0
        d = 2
        temp = x
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    composites = [x for x in all_nums if x not in current_set]
    # Sort by number of prime factors, then by value
    composites.sort(key=lambda x: (num_prime_factors(x), x))
    
    for x in composites:
        if time.time() > deadline - 10: # Leave time for SA
            break
        if x in current_set:
            continue
        new_set = current_set | {x}
        new_seq = sorted(new_set)
        if is_valid(new_seq):
            current_set = new_set
            current_seq = new_seq
            current_score = len(current_seq)
            
    best_seq = current_seq[:]
    best_score = current_score
    
    # 2. Simulated Annealing
    T = 50.0
    cooling = 0.999
    min_T = 1.0
    
    # We will run SA until time runs out or T is very low
    while time.time() < deadline - 5:
        # Generate a move
        # Weights: Add (if possible), Remove, Swap
        # Probability of add depends on how many candidates are left
        candidates = [x for x in all_nums if x not in current_set]
        
        # Decide move type
        # If we have many candidates, try add more often
        if candidates:
            r = random.random()
            if r < 0.4:
                move_type = 'add'
            elif r < 0.6:
                move_type = 'remove'
            else:
                move_type = 'swap'
        else:
            # No adds possible
            if random.random() < 0.5:
                move_type = 'remove'
            else:
                move_type = 'swap'
                
        new_set = set(current_set)
        
        if move_type == 'add':
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set.add(x)
            
        elif move_type == 'remove':
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set.remove(x)
            
        elif move_type == 'swap':
            if not current_set or not candidates:
                continue
            x_out = random.choice(list(current_set))
            x_in = random.choice(candidates)
            new_set.remove(x_out)
            new_set.add(x_in)
            
        new_seq = sorted(new_set)
        new_score = get_score(new_seq)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0:
            current_set = new_set
            current_seq = new_seq
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_seq = new_seq[:]
        else:
            if T > min_T:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_set = new_set
                    current_seq = new_seq
                    current_score = new_score
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    # 3. Final Local Search Polish
    # Try all single additions, removals, and swaps until no improvement
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try adding each missing number
        candidates = [x for x in all_nums if x not in current_set]
        for x in candidates:
            if time.time() > deadline - 2:
                break
            new_set = current_set | {x}
            new_seq = sorted(new_set)
            if is_valid(new_seq):
                current_set = new_set
                current_seq = new_seq
                current_score = len(current_seq)
                if current_score > best_score:
                    best_score = current_score
                    best_seq = new_seq[:]
                improved = True
                # Restart the loop to try more adds? Or just continue?
                # Restarting is safer to find multiple adds.
                break
        
        # Try removing each element
        if not improved:
            for x in list(current_set):
                if time.time() > deadline - 2:
                    break
                new_set = current_set - {x}
                new_seq = sorted(new_set)
                # Removal is always valid if original was valid? 
                # Yes, subsequence of a valid sequence is valid.
                # But we want to maximize length, so removal only helps if it allows
                # subsequent additions that yield a net gain. 
                # However, in this polish phase, we only accept if score increases.
                # Score of removal is len-1, which is < len. So we never accept a pure removal
                # in a "maximize length" local search unless we are looking at a combined move.
                # But wait, the SA might have gotten stuck. A removal might allow an add later.
                # In this simple local search, we only accept immediate improvements.
                # So removals are not accepted here unless we implement a 2-opt or similar.
                # Let's skip pure removals in the polish phase as they reduce length.
                pass

        # Try swaps
        if not improved:
            # This is O(N^2) checks. N is small (~30-40), so ~1600 checks.
            # Each check is O(N^2) ~ 1600 ops. Total ~2.5M ops. Might be slow in Python.
            # We should be careful.
            # Let's only do swaps if we have plenty of time.
            if time.time() < deadline - 5:
                for x_out in list(current_set):
                    if time.time() > deadline - 5:
                        break
                    for x_in in all_nums:
                        if x_in in current_set or x_in == x_out:
                            continue
                        new_set = (current_set - {x_out}) | {x_in}
                        new_seq = sorted(new_set)
                        if is_valid(new_seq):
                            # Since it's a swap, length is same. 
                            # But wait, if it's valid, score is same as current?
                            # No, score is length. If length is same, delta is 0.
                            # We only accept if delta > 0. 
                            # So pure swaps are never accepted in a "maximize length" hill climb.
                            # This suggests the polish phase is useless for swaps unless we look at 
                            # "is this swap necessary to allow a future add?".
                            # But that's not a local move.
                            # So, the main gain comes from the SA finding a sequence 
                            # where an add is possible, and the greedy/SA adding it.
                            pass
                    # break # No, we need to check all x_outs if we found an improvement? 
                    # Actually, if we don't accept swaps (delta=0), this loop is wasted.
                    # Let's remove the swap polish to save time, or make it smarter.
                    # The SA is the main driver for finding new local optima.
                    # The greedy add is the main driver for increasing length.
                    # The polish phase is mainly for "can I add one more?".
                    # So let's just do the "try all adds" part repeatedly.
                    pass

    # One last check: try all adds again
    if time.time() < deadline - 1:
        candidates = [x for x in all_nums if x not in current_set]
        for x in candidates:
            if time.time() > deadline - 1:
                break
            new_set = current_set | {x}
            new_seq = sorted(new_set)
            if is_valid(new_seq):
                current_set = new_set
                current_seq = new_seq
                current_score = len(current_seq)
                if current_score > best_score:
                    best_score = current_score
                    best_seq = new_seq[:]

    return best_seq
