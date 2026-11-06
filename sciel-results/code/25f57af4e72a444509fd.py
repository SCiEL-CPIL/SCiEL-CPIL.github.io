# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    best_size = 0
    best_A = []
    best_B = []
    
    candidates = list(range(1, 257))
    
    def is_valid_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, target):
        if target == 'A':
            if 2*x in sums_A:
                return False
            for a in A:
                if x + a in sums_A:
                    return False
            for a in A:
                if abs(x - a) in diffs_B:
                    return False
        else:
            if 2*x in sums_B:
                return False
            for b in B:
                if x + b in sums_B:
                    return False
            for b in B:
                if abs(x - b) in diffs_A:
                    return False
        return True

    def add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, target):
        if target == 'A':
            for a in A:
                sums_A.add(x + a)
                diffs_A.add(abs(x - a))
            sums_A.add(2*x)
            A.append(x)
        else:
            for b in B:
                sums_B.add(x + b)
                diffs_B.add(abs(x - b))
            sums_B.add(2*x)
            B.append(x)

    def rebuild_sets(A, B):
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        for i in range(len(A)):
            sums_A.add(2*A[i])
            for j in range(i+1, len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(abs(A[i] - A[j]))
                    
        for i in range(len(B)):
            sums_B.add(2*B[i])
            for j in range(i+1, len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(abs(B[i] - B[j]))
                    
        return sums_A, sums_B, diffs_A, diffs_B

    def run_search(seed):
        random.seed(seed)
        A = []
        B = []
        
        # Phase 1: Greedy
        order = list(candidates)
        random.shuffle(order)
        
        sums_A, sums_B, diffs_A, diffs_B = set(), set(), set(), set()
        
        for x in order:
            if time.time() - start_time > timeout:
                break
            valid_A = is_valid_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'A')
            valid_B = is_valid_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'B')
            
            if valid_A and valid_B:
                if len(A) <= len(B):
                    add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'A')
                else:
                    add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'B')
            elif valid_A:
                add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'A')
            elif valid_B:
                add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'B')
                
        current_size = len(A) + len(B)
        
        # Phase 2: Local Search with Kicks
        remaining = [x for x in candidates if x not in A and x not in B]
        random.shuffle(remaining)
        
        kicks = 0
        max_kicks = 20
        
        while time.time() - start_time < timeout and kicks < max_kicks:
            improved = False
            # Try to add any remaining element
            for x in remaining[:]:
                if time.time() - start_time > timeout:
                    break
                if x in A or x in B:
                    continue
                valid_A = is_valid_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'A')
                valid_B = is_valid_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'B')
                
                if valid_A:
                    add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'A')
                    remaining.remove(x)
                    improved = True
                elif valid_B:
                    add_element(A, B, sums_A, sums_B, diffs_A, diffs_B, x, 'B')
                    remaining.remove(x)
                    improved = True
            
            if improved:
                kicks = 0 # Reset kick counter if we made progress
                continue
            else:
                # Stuck. Perform a kick.
                # Remove a random element from A or B
                all_elements = A + B
                if not all_elements:
                    break
                
                # Prefer removing from larger set or random
                if A and (not B or random.random() < 0.5):
                    target = 'A'
                    elem_to_remove = random.choice(A)
                else:
                    target = 'B'
                    elem_to_remove = random.choice(B)
                
                # Remove
                if target == 'A':
                    A.remove(elem_to_remove)
                else:
                    B.remove(elem_to_remove)
                
                # Rebuild sets
                sums_A, sums_B, diffs_A, diffs_B = rebuild_sets(A, B)
                
                # Add removed element back to remaining
                remaining.append(elem_to_remove)
                random.shuffle(remaining)
                
                kicks += 1
        
        return A, B

    # Run multiple trials
    for seed in range(10000):
        if time.time() - start_time > timeout:
            break
        
        A, B = run_search(seed)
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
            
        # Early stop if we hit theoretical max?
        # Max Sidon size for 256 is ~16. Two sets max ~32.
        # If we hit 30, it's very good.
        if best_size >= 32:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
