import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    
    def run_greedy(order):
        """
        Greedily build a set from 'order' such that for any two subsets of different
        cardinalities, their sums are distinct.
        """
        S = [set() for _ in range(61)]
        S[0] = {0}
        A = []
        current_size = 0
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            
            valid = True
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                A.append(x)
                current_size += 1
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
        
        return A

    def backtrack_search(pool, max_depth, time_limit):
        """
        Backtracking CSP search: try to build a set of size up to max_depth
        from pool, where no two subsets of different cardinalities have the same sum.
        Uses forward checking to prune early.
        """
        best_found = []
        
        def can_add(A, S, x):
            """Check if x can be added to current set A with sum-sets S."""
            n = len(A)
            for j in range(1, n + 2):
                if not S[j-1]:
                    continue
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(n + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            return False
            return True
        
        def search(A, S, pool, depth, target_size):
            nonlocal best_found
            if time.time() > time_limit:
                return
            if len(A) > len(best_found):
                best_found = list(A)
            if len(A) >= target_size:
                return
            
            # Forward checking: for each remaining candidate, check if it's valid
            valid_candidates = []
            for x in pool:
                if can_add(A, S, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return
            
            # Try candidates in order (smaller first tends to work better for sum constraints)
            for x in valid_candidates:
                if time.time() > time_limit:
                    return
                new_A = A + [x]
                new_S = [set() for _ in range(61)]
                new_S[0] = {0}
                for k in range(1, len(A) + 1):
                    new_S[k] = set(S[k])
                for k in range(len(A) + 1, 0, -1):
                    new_S[k].update(s + x for s in new_S[k-1])
                
                remaining_pool = [p for p in pool if p > x]
                search(new_A, new_S, remaining_pool, depth + 1, target_size)
            
            # Also try without picking the first valid candidate (to explore alternatives)
            # This is handled by trying all valid candidates above
        
        search([], [set() for _ in range(61)], pool, 0, max_depth)
        return best_found

    # First, get a good starting point from greedy
    best_set = run_greedy(list(range(1, 257)))
    
    # Try a few deterministic orderings
    for order in [list(range(256, 0, -1)),
                  list(range(128, 0, -1)) + list(range(129, 257)),
                  list(range(129, 257)) + list(range(128, 0, -1))]:
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Randomized greedy runs
    rng = random.Random(42)
    while time.time() - start_time < deadline - 10.0:
        order = list(range(1, 257))
        rng.shuffle(order)
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Now try backtracking to extend or find larger sets
    # Use the greedy result as a lower bound
    current_best_size = len(best_set)
    
    # Try backtracking with increasing target sizes
    # Start from a pool that excludes elements already in best_set, plus some new ones
    # Actually, let's try backtracking from scratch with a restricted pool for speed
    
    remaining_time = deadline - (time.time() - start_time)
    
    if remaining_time > 5.0:
        # Try backtracking with various pool sizes
        # Larger pools are more expensive; start moderate
        pool_sizes = [60, 80, 100, 130, 160, 200, 256]
        
        for ps in pool_sizes:
            if time.time() > deadline - 3.0:
                break
            remaining = deadline - (time.time() - start_time)
            if remaining < 3.0:
                break
            
            # Use a subset of the pool: prefer smaller numbers (easier to satisfy constraints)
            # Mix: take smallest ps/2 and some random from the rest
            base_pool = list(range(1, ps + 1))
            
            # Target: try to beat current best
            target = min(current_best_size + 2, ps)
            
            result = backtrack_search(base_pool, target, time.time() + remaining - 2.0)
            if len(result) > len(best_set):
                best_set = result
                current_best_size = len(best_set)
    
    # Final extension attempt on best_set
    if time.time() - start_time < deadline - 2.0:
        current_set = set(best_set)
        S = [set() for _ in range(61)]
        S[0] = {0}
        current_size = len(best_set)
        
        for k in range(1, current_size + 1):
            for combo in combinations(best_set, k):
                S[k].add(sum(combo))
        
        for x in range(1, 257):
            if x in current_set:
                continue
            if time.time() - start_time > deadline - 1.0:
                break
            
            valid = True
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_set.append(x)
                current_set.add(x)
                current_size += 1
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
    
    return [int(x) for x in best_set]
