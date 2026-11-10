import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s buffer for 60s limit
    
    # We want to return a list of ints
    # Initial best known valid set (from seed or simple construction)
    # The seed program had [1, 2, 4, 8, 16, 32, 64, 128, 256] which is size 9.
    # We will try to find a larger one.
    
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    # Candidates are 1..256
    # We try to build the set by adding elements.
    # Heuristic: Larger elements first to separate sum layers.
    candidates = list(range(256, 0, -1))
    
    # sums_by_k[k] is a set of sums for subsets of size k
    # Initially, with empty set, sums_by_k = [{0}, set(), ...]
    # We start with empty set.
    
    def check_valid(sums_by_k, max_k):
        """
        Check if sums_by_k[0..max_k] are pairwise disjoint.
        """
        for i in range(max_k + 1):
            for j in range(i + 1, max_k + 1):
                if not sums_by_k[i].isdisjoint(sums_by_k[j]):
                    return False
        return True

    def dfs(current_set, sums_by_k, max_k, candidates):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if not candidates:
            return
            
        # Try adding each candidate
        # Optimization: We can break early if we think we can't beat best_size?
        # No, we just need to find ANY set larger than best_size.
        
        for i, v in enumerate(candidates):
            if time.time() > deadline:
                return
                
            # Compute new sums if we add v
            # new_sums_by_k will have size max_k + 2 (indices 0 to max_k+1)
            new_sums_by_k = [set() for _ in range(max_k + 2)]
            
            # Copy old sums
            # S'_k = S_k U {s + v | s in S_{k-1}}
            # Note: S_0 is always {0}
            
            # We can do this in place on a copy or build new list
            # Building new list is safer
            
            for k in range(max_k + 1):
                # Existing sums of size k
                new_sums_by_k[k].update(sums_by_k[k])
                
            # Add new sums involving v
            for k in range(1, max_k + 2):
                # Sums of size k-1 from old set, plus v
                prev_sums = sums_by_k[k-1]
                for s in prev_sums:
                    new_sums_by_k[k].add(s + v)
                    
            # Check validity
            if check_valid(new_sums_by_k, max_k + 1):
                # Recurse
                new_current_set = current_set + [v]
                new_candidates = candidates[:i] + candidates[i+1:]
                dfs(new_current_set, new_sums_by_k, max_k + 1, new_candidates)
                
        # Note: We don't backtrack by removing from current_set because we passed copies/immutable lists
        # current_set is a list, we create new list in recursion.
        
    # Initial state: empty set
    # sums_by_k[0] = {0}, others empty
    initial_sums = [set() for _ in range(1)]
    initial_sums[0].add(0)
    
    dfs([], initial_sums, 0, candidates)
    
    # Ensure types are correct
    return [int(x) for x in best_set]
