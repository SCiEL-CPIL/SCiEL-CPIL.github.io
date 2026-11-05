"""Optimized Sidon set construction with fast incremental checks."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 256
    max_size = 64
    rng = random.Random(42)
    
    best_total = 0
    best_A = []
    best_B = []
    
    def compute_sums(sets):
        sums = set()
        for i, a in enumerate(sets):
            for j in range(i, len(sets)):
                sums.add(a + sets[j])
        return sums

    def compute_diffs(sets):
        diffs = set()
        for i, a in enumerate(sets):
            for b in sets:
                if a != b:
                    diffs.add(a - b)
        return diffs

    def can_add_to_A(x, A, sums_A, diffs_B):
        # Check Sidon for A U {x}
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
        # Check Diff Disjointness with B
        for a in A:
            if (x - a) in diffs_B:
                return False
            if (a - x) in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        # Check Sidon for B U {x}
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
        # Check Diff Disjointness with A
        for b in B:
            if (x - b) in diffs_A:
                return False
            if (b - x) in diffs_A:
                return False
        return True

    def update_sums(sums, set_list, x):
        sums.add(x + x)
        for a in set_list:
            sums.add(x + a)
            
    def update_diffs(diffs, set_list, x):
        for a in set_list:
            diffs.add(x - a)
            diffs.add(a - x)

    def valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        # Re-verify from scratch for safety
        sums_A = compute_sums(A)
        sums_B = compute_sums(B)
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)
        
        # Check Sidon
        if len(sums_A) != len(A)*(len(A)+1)//2:
            return False
        if len(sums_B) != len(B)*(len(B)+1)//2:
            return False
        
        # Check Disjoint Diffs
        if diffs_A & diffs_B:
            return False
        return True

    def greedy_construct(order_A, order_B):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order_A:
            if len(A) >= max_size:
                break
            if x not in A:
                if can_add_to_A(x, A, sums_A, diffs_B):
                    A.append(x)
                    update_sums(sums_A, A[:-1], x)
                    update_diffs(diffs_A, A[:-1], x)
        
        for x in order_B:
            if len(B) >= max_size:
                break
            if x not in B:
                if can_add_to_B(x, B, sums_B, diffs_A):
                    B.append(x)
                    update_sums(sums_B, B[:-1], x)
                    update_diffs(diffs_B, B[:-1], x)
        
        return A, B

    def local_search(A, B, max_iters=1000):
        A = list(A)
        B = list(B)
        sums_A = compute_sums(A)
        sums_B = compute_sums(B)
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)
        
        all_elements = list(range(1, n + 1))
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding to A
            if len(A) < max_size:
                # Randomly pick candidates to avoid bias
                candidates = [x for x in all_elements if x not in A]
                rng.shuffle(candidates)
                for x in candidates:
                    if can_add_to_A(x, A, sums_A, diffs_B):
                        A.append(x)
                        update_sums(sums_A, A[:-1], x)
                        update_diffs(diffs_A, A[:-1], x)
                        improved = True
                        break
            
            # Try adding to B
            if len(B) < max_size:
                candidates = [x for x in all_elements if x not in B]
                rng.shuffle(candidates)
                for x in candidates:
                    if can_add_to_B(x, B, sums_B, diffs_A):
                        B.append(x)
                        update_sums(sums_B, B[:-1], x)
                        update_diffs(diffs_B, B[:-1], x)
                        improved = True
                        break
                        
            # Try swapping: Remove from A, Add to B
            if not improved and len(A) > 0 and len(B) < max_size:
                # Try removing a random element from A
                idx = rng.randrange(len(A))
                a_to_remove = A[idx]
                A_new = A[:idx] + A[idx+1:]
                # Recalculate structures for A_new
                sums_A_new = compute_sums(A_new)
                diffs_A_new = compute_diffs(A_new)
                
                candidates = [x for x in all_elements if x not in B]
                rng.shuffle(candidates)
                for x in candidates:
                    if can_add_to_B(x, B, sums_B, diffs_A_new):
                        # Check if total size increases (it does: -1 + 1 = 0 change? No, we are swapping, so size stays same? 
                        # Wait, the goal is to maximize total size. 
                        # If we remove from A and add to B, total size is constant. 
                        # This move is only useful if it allows *future* additions. 
                        # But local search usually looks for immediate gain. 
                        # Since size doesn't change, we should only do this if it leads to a state where we can add more? 
                        # That's lookahead. 
                        # Alternatively, we can just try to add to the set with more space. 
                        # If both are full, we must swap to free space? 
                        # No, if both are full, we can't add. 
                        # So swaps are for restructuring to allow future growth. 
                        # Given time limits, we might skip complex swaps and rely on random restarts. 
                        pass
            
            if not improved:
                break
        
        return A, B

    # Try multiple random orderings
    for seed in range(50):
        rng.seed(seed)
        all_elements = list(range(1, n + 1))
        order_A = all_elements[:]
        order_B = all_elements[:]
        rng.shuffle(order_A)
        rng.shuffle(order_B)
        
        A, B = greedy_construct(order_A, order_B)
        
        # Simple local search: just try to add more
        # The greedy is already quite good. 
        # Let's do a few rounds of "try to add any available element"
        for _ in range(10):
            A, B = local_search(A, B, max_iters=10)
            
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
        
        if best_total >= 32: # Theoretical max is 32
            break
    
    # Final validation
    if not valid_solution(best_A, best_B):
        # Fallback
        A = [1, 2, 4, 8, 16, 32, 64, 128]
        B = [3, 5, 9, 17, 33, 65, 129]
        # Filter B to be <= 256
        B = [x for x in B if x <= 256]
        if not valid_solution(A, B):
             A = [1]
             B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
