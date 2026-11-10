import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s kill
    
    N = 256
    MAX_SIZE = 64
    
    def _can_add_to_A(a, sums_A, diffs_A, diffs_B, A_set):
        """Check if 'a' can be added to A without violating Sidon or diff constraints."""
        if a in A_set:
            return False
        
        # Check Sidon condition for A:
        # New sums: a+x for x in A, and a+a
        # These must not be in sums_A
        # And a+x must be unique for all x in A (implied if x are unique and a is fixed, 
        # but we must ensure a+x != a+y for x!=y, which is true since x!=y.
        # We must ensure a+x not in sums_A.
        # Also a+a not in sums_A.
        
        for x in A_set:
            if a + x in sums_A:
                return False
        
        if a + a in sums_A:
            return False
            
        # Check difference disjointness:
        # New diffs: |a-x| for x in A
        # These must not be in diffs_B
        # Note: diffs_A will be updated, but we only care if new diffs intersect diffs_B
        # Also, new diffs must not be 0 (a!=x guaranteed by a not in A_set)
        
        for x in A_set:
            d = abs(a - x)
            if d in diffs_B:
                return False
                
        return True

    def _can_add_to_B(b, sums_B, diffs_A, diffs_B, B_set):
        """Check if 'b' can be added to B without violating Sidon or diff constraints."""
        if b in B_set:
            return False
        
        for x in B_set:
            if b + x in sums_B:
                return False
        
        if b + b in sums_B:
            return False
            
        for x in B_set:
            d = abs(b - x)
            if d in diffs_A:
                return False
                
        return True

    def _add_to_A(a, A, A_set, sums_A, diffs_A):
        """Actually add 'a' to A and update state."""
        A.append(a)
        A_set.add(a)
        sums_A.add(a + a)
        for x in A_set:
            sums_A.add(a + x)
            diffs_A.add(abs(a - x))

    def _remove_from_A(a, A, A_set, sums_A, diffs_A):
        """Remove 'a' from A and update state. Note: sums and diffs might be shared, 
        so simple removal is tricky if we just use sets. 
        For backtracking, it's safer to recompute or use reference counting.
        Given the complexity, we will stick to a forward-only greedy with restarts 
        and a separate local search phase that swaps elements, avoiding complex removal logic 
        or implementing it carefully."""
        pass

    def _greedy_construct(seed, time_limit):
        random.seed(seed)
        A = []
        B = []
        A_set = set()
        B_set = set()
        
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Strategy: Alternate adding to A and B, preferring the smaller set.
        # Use a candidate pool. Smaller numbers tend to be better for Sidon packing? 
        # Actually, random is often better to avoid clumping, but let's try a mix.
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # We will try to add elements one by one.
        # To make it faster, we iterate through candidates and try to add to A or B.
        # If we can't add to either, we skip.
        
        # However, a better strategy is to pick a random candidate from the remaining pool
        # and try to place it. If it fails, try another.
        
        # Let's use a "best fit" approach: 
        # At each step, try to add to the smaller set. 
        # If that fails, try the larger set.
        
        used_global = set()
        
        while time.time() < time_limit:
            # Determine which set to grow
            if len(A) <= len(B):
                primary, secondary = 'A', 'B'
            else:
                primary, secondary = 'B', 'A'
                
            added = False
            
            # Try random candidates from unused numbers
            # To speed up, we can shuffle the remaining candidates occasionally
            # But checking all candidates is O(N) * check_cost.
            # Check cost is O(|Set|).
            # So one pass is O(N * |Set|).
            
            # Let's just iterate through a shuffled version of remaining candidates
            # If we find one that fits, we add it.
            
            # Optimization: Instead of shuffling the whole list every time, 
            # we can just pick random indices.
            
            # Let's try a few random candidates first, then fall back to scan if needed?
            # No, scan is deterministic if we sort. Random is better for diversity.
            
            # Let's pick 100 random candidates and try them.
            # If none work, try the other set.
            # If still none, break or restart.
            
            remaining = [x for x in range(1, N+1) if x not in used_global]
            if not remaining:
                break
            
            # Shuffle a subset to try
            random.shuffle(remaining)
            
            # Try primary set
            for c in remaining:
                if primary == 'A':
                    if _can_add_to_A(c, sums_A, diffs_A, diffs_B, A_set):
                        _add_to_A(c, A, A_set, sums_A, diffs_A)
                        used_global.add(c)
                        added = True
                        break
                else:
                    if _can_add_to_B(c, sums_B, diffs_A, diffs_B, B_set):
                        # Add to B
                        B.append(c)
                        B_set.add(c)
                        sums_B.add(c + c)
                        for x in B_set:
                            sums_B.add(c + x)
                            diffs_B.add(abs(c - x))
                        used_global.add(c)
                        added = True
                        break
                        
            if not added:
                # Try secondary set
                for c in remaining:
                    if primary == 'A':
                         # Primary was A, so secondary is B
                         if _can_add_to_B(c, sums_B, diffs_A, diffs_B, B_set):
                            B.append(c)
                            B_set.add(c)
                            sums_B.add(c + c)
                            for x in B_set:
                                sums_B.add(c + x)
                                diffs_B.add(abs(c - x))
                            used_global.add(c)
                            added = True
                            break
                    else:
                         # Primary was B, so secondary is A
                         if _can_add_to_A(c, sums_A, diffs_A, diffs_B, A_set):
                            _add_to_A(c, A, A_set, sums_A, diffs_A)
                            used_global.add(c)
                            added = True
                            break
                            
            if not added:
                # Stuck. Break.
                break
                
        return A, B

    def _local_search(A, B, time_limit):
        """
        Attempt to improve the solution by swapping elements or adding new ones.
        This is complex to implement correctly with the state.
        Simpler: Just do more greedy restarts with different seeds and initializations.
        """
        return A, B

    best_total = 0
    best_A = []
    best_B = []
    
    num_restarts = 500 # We have more time with faster checks
    
    for i in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Vary the seed and potentially the strategy
        A, B = _greedy_construct(i, deadline - 2.0)
        
        total = len(A) + len(B)
        
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
            # Optional: Local search here if time permits
            # For now, rely on the speed of the greedy check to allow many restarts

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        # Find a valid B element
        # Try 2
        if 2 not in best_A:
            best_B = [2]
        else:
            best_B = [3]

    return [best_A, best_B]
