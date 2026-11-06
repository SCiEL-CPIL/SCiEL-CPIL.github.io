import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 89.0  # Safety margin for 90s kill limit
    
    n = 256
    best_A = []
    best_B = []
    best_size = 0
    
    # Helper to check Sidon and Diff constraints for adding x to S
    # S: set of elements
    # sums_S: set of pair sums
    # diffs_S: set of nonzero differences
    # D_other: set of differences of the OTHER set
    def can_add(S, sums_S, diffs_S, D_other, x):
        # Check Sidon: new sums must not be in sums_S
        # New sums: {x + s for s in S} U {2x}
        # Check Diff Disjointness: new diffs must not be in D_other
        # Also check that new diffs don't conflict with S's own diffs (Sidon requires unique differences too)
        
        # Optimization: Check sums first, usually fails fast
        for s in S:
            if x + s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
            
        for s in S:
            d1 = x - s
            d2 = s - x
            # Check against other set's diffs
            if d1 in D_other or d2 in D_other:
                return False
            # Check against S's own diffs (for Sidon property - differences must be unique in magnitude)
            if d1 in diffs_S or d2 in diffs_S:
                return False
                
        return True

    def add_to_set(S, sums_S, diffs_S, x):
        # Add x to S, update sums and diffs
        S.add(x)
        for s in S:
            if s == x:
                continue
            sums_S.add(x + s)
            diffs_S.add(x - s)
            diffs_S.add(s - x)
        sums_S.add(2 * x)

    def recompute(S):
        sums = set()
        diffs = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
            for j in range(len(S_list)):
                if i != j:
                    diffs.add(S_list[i] - S_list[j])
        return sums, diffs

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)
        # Check Sidon
        if len(sums_A) != len(list(combinations_with_replacement(A, 2))):
            return False
        if len(sums_B) != len(list(combinations_with_replacement(B, 2))):
            return False
        # Check Disjoint Diffs
        if diffs_A & diffs_B:
            return False
        return True

    def fast_greedy(seed_offset=0):
        """Fast greedy construction with balanced growth strategy."""
        rng = random.Random(seed_offset)
        
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Balance: prefer adding to the smaller set
            if len(A) < len(B):
                order = ['A', 'B']
            elif len(B) < len(A):
                order = ['B', 'A']
            else:
                order = ['A', 'B'] if rng.random() < 0.5 else ['B', 'A']
            
            added = False
            for target in order:
                if target == 'A':
                    if len(A) >= 64:
                        continue
                    if can_add(A, sums_A, diffs_A, diffs_B, x):
                        add_to_set(A, sums_A, diffs_A, x)
                        added = True
                        break
                else:
                    if len(B) >= 64:
                        continue
                    if can_add(B, sums_B, diffs_B, diffs_A, x):
                        add_to_set(B, sums_B, diffs_B, x)
                        added = True
                        break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_add(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try to add more elements via local search with multiple passes."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        max_passes = 10
        for _ in range(max_passes):
            improved = True
            while improved:
                improved = False
                if time.time() - start_time > time_limit:
                    break
                for x in range(1, n + 1):
                    if x in A or x in B:
                        continue
                    
                    if len(A) < 64 and can_add(A, sums_A, diffs_A, diffs_B, x):
                        add_to_set(A, sums_A, diffs_A, x)
                        improved = True
                        break
                    if len(B) < 64 and can_add(B, sums_B, diffs_B, diffs_A, x):
                        add_to_set(B, sums_B, diffs_B, x)
                        improved = True
                        break
            
            if not improved:
                break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try swapping elements between A and B to improve balance and enable additions."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = True
        max_iters = 200
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > time_limit:
                break
            
            # Try moving x from A to B
            for x in list(A):
                if x in B:
                    continue
                if len(B) >= 64:
                    continue
                
                A_temp = A - {x}
                _, diffs_A_temp = recompute(A_temp)
                
                if can_add(B, sums_B, diffs_B, diffs_A_temp, x):
                    A.remove(x)
                    sums_A, diffs_A = recompute(A)
                    add_to_set(B, sums_B, diffs_B, x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try moving x from B to A
            for x in list(B):
                if x in A:
                    continue
                if len(A) >= 64:
                    continue
                
                B_temp = B - {x}
                _, diffs_B_temp = recompute(B_temp)
                
                if can_add(A, sums_A, diffs_A, diffs_B_temp, x):
                    B.remove(x)
                    sums_B, diffs_B = recompute(B)
                    add_to_set(A, sums_A, diffs_A, x)
                    improved = True
                    break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try removing an element and replacing with a different one to unlock additions."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = False
        current_size = len(A) + len(B)
        
        # Try removing each element from A and see if we can add 2+ new elements
        for x in list(A):
            if len(A) <= 1:
                continue
            A_temp = A - {x}
            sums_A_temp, diffs_A_temp = recompute(A_temp)
            
            A_new = set(A_temp)
            B_new = set(B)
            sums_A_new = set(sums_A_temp)
            diffs_A_new = set(diffs_A_temp)
            sums_B_new = set(sums_B)
            diffs_B_new = set(diffs_B)
            
            for y in range(1, n + 1):
                if y in A_new or y in B_new:
                    continue
                if len(A_new) < 64 and can_add(A_new, sums_A_new, diffs_A_new, diffs_B_new, y):
                    add_to_set(A_new, sums_A_new, diffs_A_new, y)
                    continue
                if len(B_new) < 64 and can_add(B_new, sums_B_new, diffs_B_new, diffs_A_new, y):
                    add_to_set(B_new, sums_B_new, diffs_B_new, y)
                    continue
            
            if len(A_new) + len(B_new) > current_size:
                A, B = A_new, B_new
                sums_A, diffs_A = sums_A_new, diffs_A_new
                sums_B, diffs_B = sums_B_new, diffs_B_new
                current_size = len(A) + len(B)
                improved = True
        
        # Same for B
        for x in list(B):
            if len(B) <= 1:
                continue
            B_temp = B - {x}
            sums_B_temp, diffs_B_temp = recompute(B_temp)
            
            A_new = set(A)
            B_new = set(B_temp)
            sums_A_new = set(sums_A)
            diffs_A_new = set(diffs_A)
            sums_B_new = set(sums_B_temp)
            diffs_B_new = set(diffs_B_temp)
            
            for y in range(1, n + 1):
                if y in A_new or y in B_new:
                    continue
                if len(A_new) < 64 and can_add(A_new, sums_A_new, diffs_A_new, diffs_B_new, y):
                    add_to_set(A_new, sums_A_new, diffs_A_new, y)
                    continue
                if len(B_new) < 64 and can_add(B_new, sums_B_new, diffs_B_new, diffs_A_new, y):
                    add_to_set(B_new, sums_B_new, diffs_B_new, y)
                    continue
            
            if len(A_new) + len(B_new) > current_size:
                A, B = A_new, B_new
                sums_A, diffs_A = sums_A_new, diffs_A_new
                sums_B, diffs_B = sums_B_new, diffs_B_new
                current_size = len(A) + len(B)
                improved = True
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    # Main search loop
    num_restarts = 90
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Greedy construction with randomization
        A, B, sums_A, diffs_A, sums_B, diffs_B = fast_greedy(restart)
        
        # Local search: add
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_add(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Local search: swap
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Local search: replace
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Final add pass
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_add(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = A
                best_B = B

    # Final check
    if not best_A or not best_B:
        return [[1], [2]]
        
    return [list(best_A), list(best_B)]
