import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit

    # 1. Precomputation
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]

    # 2. Validity Check using Integer Products
    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    # 3. Greedy Insertion Search
    def greedy_insertion(base_seq, remaining):
        current = list(base_seq)
        rem = list(remaining)
        # Sort remaining to try smaller numbers first? Or random?
        # Smaller numbers might fit in gaps better.
        rem.sort()
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to insert any remaining number
            # We iterate through rem. If we insert one, we restart.
            for i, val in enumerate(rem):
                if time.time() > deadline - 5:
                    return current
                
                # Try all positions
                for pos in range(len(current) + 1):
                    # Check strict increase
                    if pos > 0 and current[pos-1] >= val:
                        continue
                    if pos < len(current) and val >= current[pos]:
                        continue
                    
                    candidate = current[:pos] + [val] + current[pos:]
                    if is_valid(candidate):
                        current = candidate
                        rem.pop(i)
                        improved = True
                        break # Restart the outer loop with new state
                if improved:
                    break
        
        return current

    # 4. Local Search (Swaps) to escape local optima
    def local_search(seq, remaining, max_iter=500):
        current = list(seq)
        rem = list(remaining)
        used = set(current)
        
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try to swap an element in current with one in rem
            # This doesn't change length, but might allow further insertions.
            # Actually, we want to maximize length. Swaps are only useful if they
            # enable an insertion that was previously blocked.
            
            # A simpler local search: Try to insert a number from rem.
            # If we can't, try removing a number from current to make space?
            # No, we want to maximize length.
            
            # Let's try a different approach for local search:
            # Randomly shuffle rem and try greedy insertion again.
            # This is essentially a restart.
            
            # Better: Try to insert a number from rem.
            # If stuck, try to remove a number from current, insert a different one, 
            # and see if we can insert more?
            
            # Given the time, simple randomized restarts of greedy insertion
            # with different orderings of 'remaining' are more effective.
            
            # Let's just return current. The restarts in main loop handle diversity.
            pass
        
        return current

    # 5. Main Search Loop
    best_seq = []
    best_len = 0

    # Strategy 1: Start with all primes, insert composites greedily
    initial_seq = sorted(primes)
    remaining_comps = sorted(composites)
    
    # We will run multiple trials with different randomizations of the remaining list
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Shuffle remaining composites to explore different paths
        rem = list(remaining_comps)
        random.shuffle(rem)
        
        # Greedy insertion
        seq = greedy_insertion(initial_seq, rem)
        
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    # Strategy 2: If we have time, try local improvements on the best found
    # Specifically, try to insert any remaining numbers into best_seq
    if time.time() < deadline - 5:
        used = set(best_seq)
        remaining = [x for x in all_numbers if x not in used]
        # Try to insert any of these
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for val in remaining:
                for pos in range(len(best_seq) + 1):
                    if pos > 0 and best_seq[pos-1] >= val:
                        continue
                    if pos < len(best_seq) and val >= best_seq[pos]:
                        continue
                    candidate = best_seq[:pos] + [val] + best_seq[pos:]
                    if is_valid(candidate):
                        best_seq = candidate
                        remaining.remove(val)
                        improved = True
                        break
                if improved:
                    break

    return best_seq
