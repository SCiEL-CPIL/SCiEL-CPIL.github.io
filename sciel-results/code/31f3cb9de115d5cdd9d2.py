import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses CRT decomposition for initialization and randomized local search for optimization.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin

    m = 205
    
    # 1. Compute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency list for fast conflict checking
    # adj[i] is a set of j such that (i-j) mod m is in forbidden
    adj = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j = (i - f) % m
            adj[i].add(j)
            # Also add reverse direction implicitly by symmetry of difference in set?
            # (i-j) in F => (j-i) = -(i-j). Is -1 a square mod 205?
            # -1 mod 5 = 4 (square). -1 mod 41 = 40. 40 is a square mod 41?
            # 40 = -1. (-1/41) = (-1)^((41-1)/2) = (-1)^20 = 1. Yes.
            # So -1 is a square mod 205.
            # Thus if d is a square, -d is a square.
            # So the graph is undirected.
            
    # 2. CRT Initialization
    # Find max independent sets in Z_5 and Z_41
    def max_independent_set_prime(p):
        # Brute force for small p
        # Squares mod p
        sqs = set()
        for x in range(1, p):
            sqs.add(pow(x, 2, p))
        sqs.discard(0)
        
        # Graph edges
        edges = [set() for _ in range(p)]
        for u in range(p):
            for v in range(u+1, p):
                diff = (u - v) % p
                if diff in sqs:
                    edges[u].add(v)
                    edges[v].add(u)
                    
        best_set = set()
        best_size = 0
        # Backtracking
        def backtrack(candidates, current):
            nonlocal best_set, best_size
            if not candidates:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
                return
            # Upper bound
            if len(current) + len(candidates) <= best_size:
                return
                
            # Pick vertex with highest degree in candidates
            v = max(candidates, key=lambda x: len(edges[x] & candidates))
            
            # Include v
            new_cand_inc = candidates - {v} - edges[v]
            backtrack(new_cand_inc, current | {v})
            
            # Exclude v
            new_cand_exc = candidates - {v}
            backtrack(new_cand_exc, current)
            
        backtrack(set(range(p)), set())
        return best_set

    I5 = max_independent_set_prime(5)
    I41 = max_independent_set_prime(41)
    
    # Construct initial set A
    # A = { a in Z_205 | a mod 5 in I5 and a mod 41 in I41 }
    initial_A = set()
    for a in range(m):
        if a % 5 in I5 and a % 41 in I41:
            initial_A.add(a)
            
    # Verify initial_A is valid (it should be, but let's check logic)
    # If u, v in A, u5 in I5, v5 in I5.
    # If u5 != v5, then u5-v5 is not in S5\{0} (by def of I5). So diff not square.
    # If u5 == v5, then u41 in I41, v41 in I41.
    # If u41 != v41, then u41-v41 is not in S41\{0}. So diff not square.
    # If u5 == v5 and u41 == v41, then u == v.
    # So initial_A is a valid independent set.
    
    best_set = set(initial_A)
    best_size = len(best_set)
    
    # 3. Local Search
    def is_valid_addition(v, current_set):
        for u in current_set:
            if v in adj[u]:
                return False
        return True
        
    def local_search(start_set, max_iter=5000):
        current_set = set(start_set)
        current_size = len(current_set)
        no_improve_count = 0
        max_no_improve = 100
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Try to add a random vertex
            improved = False
            candidates = list(set(range(m)) - current_set)
            random.shuffle(candidates)
            
            # Prioritize vertices with fewer conflicts
            def conflict_count(v):
                return len(adj[v] & current_set)
                
            candidates.sort(key=conflict_count)
            
            for v in candidates:
                if is_valid_addition(v, current_set):
                    current_set.add(v)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
                
            # If no addition, try a swap
            # Pick a random element in set to remove, and try to add a new one
            if current_size > 0:
                remove_candidates = list(current_set)
                random.shuffle(remove_candidates)
                swap_improved = False
                for u in remove_candidates:
                    # Tentatively remove u
                    tentative_set = current_set - {u}
                    # Try to add a vertex that was previously blocked by u
                    # or just any random vertex
                    add_candidates = list(set(range(m)) - tentative_set)
                    # Prioritize those that conflict with u (so removing u helps)
                    # and are valid for the rest
                    def score_add(w):
                        if is_valid_addition(w, tentative_set):
                            # Bonus if w conflicts with u
                            return 1 if w in adj[u] else 0
                        return -1
                            
                    add_candidates.sort(key=score_add, reverse=True)
                    for w in add_candidates:
                        if is_valid_addition(w, tentative_set):
                            # Perform swap
                            current_set.remove(u)
                            current_set.add(w)
                            # Size stays same, but structure changes
                            # This allows future additions
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                        
            if not improved and not swap_improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
        return current_set

    # Run multiple restarts
    best_global = set(initial_A)
    best_global_size = len(best_global)
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb the initial set or use a different initialization
        if i == 0:
            start = initial_A
        else:
            # Random restart: take a random subset of best_global or random valid set
            # To ensure validity, we can start from a small valid set and grow
            # Or just perturb best_global by removing a random element
            start = set(best_global)
            if start:
                # Remove 1-3 random elements to allow different search paths
                to_remove = random.sample(list(start), min(3, len(start)))
                for r in to_remove:
                    start.remove(r)
                    
        result = local_search(start)
        
        if len(result) > best_global_size:
            best_global_size = len(result)
            best_global = result
            
    # Final verification
    # Ensure no forbidden pairs
    final_list = sorted(best_global)
    for i in range(len(final_list)):
        for j in range(i+1, len(final_list)):
            diff = (final_list[i] - final_list[j]) % m
            if diff in forbidden:
                # Should not happen, but if it does, return empty or fix?
                # For safety, return a known valid small set
                return np.array([0], dtype=np.int64)
                
    return np.array(final_list, dtype=np.int64)

# EVOLVE-BLOCK-END
