import math
import random
import time

def construct():
    # Set a random seed for reproducibility if needed, but usually random is fine
    # We want to maximize: 1 + log(|A-A| / |A+A|) / log(2 * max(A) + 1)
    
    def calc_score(A):
        if not A or 0 not in A:
            return -1.0
        n = len(A)
        if n < 2:
            return -1.0
        max_a = max(A)
        if max_a == 0:
            return -1.0
        
        # Compute sumset and diffset
        # Using sets is O(n^2) which is fine for small n
        sum_set = set()
        diff_set = set()
        for a in A:
            for b in A:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        P = len(sum_set)
        D = len(diff_set)
        if P == 0:
            return -1.0
        ratio = D / P
        if ratio <= 0:
            return -1.0
        denom = math.log(2 * max_a + 1)
        if denom <= 0:
            return -1.0
        return 1.0 + math.log(ratio) / denom

    def local_search(initial_A, max_iters=500):
        """
        Perform local search by swapping elements.
        We maintain a set of elements. We try to replace an element with a random other element
        in the range [0, bound] to see if it improves the score.
        """
        A = list(initial_A)
        A.sort()
        current_score = calc_score(A)
        if current_score < 0:
            return A, current_score
            
        bound = max(A) + 10 # Search space slightly larger than current max
        best_set = A[:]
        best_score = current_score
        
        for _ in range(max_iters):
            # Try a random swap
            i = random.randint(0, len(A) - 1)
            if A[i] == 0:
                continue # Do not swap 0, it must be in the set
                
            # Choose a new value for A[i]
            # We pick a random value from a range. 
            # To keep max(A) small, we bias towards smaller values.
            # Let's try values in [1, bound]
            new_val = random.randint(1, bound)
            
            # Avoid duplicates
            if new_val in A:
                continue
                
            old_val = A[i]
            A[i] = new_val
            
            # Ensure 0 is still in A (it is, since we didn't touch index of 0 if we skipped it, 
            # but wait, if 0 was not at index i, it's still there. 
            # However, if we pick new_val=0, we'd have duplicate 0. 
            # The check `if new_val in A` handles this if 0 is in A.
            # But we must ensure 0 is IN the set. 
            # Since we never remove the element 0 (we skip i where A[i]==0), 
            # and we don't add duplicate 0s, 0 remains.
            
            new_score = calc_score(A)
            
            if new_score > best_score:
                best_score = new_score
                best_set = A[:]
                # Expand bound slightly if we found a better solution with larger max?
                # No, we want small max. 
                # Maybe adjust bound to be max(best_set) + 5 to keep search local
                bound = max(best_set) + 5
            else:
                # Revert
                A[i] = old_val
                
        return best_set, best_score

    best_global_score = -1.0
    best_global_set = [0, 1, 3, 7] # Fallback
    
    # We will try several sizes n
    # The score depends on n and max(A).
    # For small n, ratio is high. For large n, ratio drops but log(max) grows.
    # Let's try n from 4 to 15.
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer for 120s limit
    
    # Pre-compute some good starting points? 
    # Or just random starts.
    
    for n in range(4, 16):
        if time.time() - start_time > time_limit:
            break
            
        # Number of restarts for this size
        restarts = 5
        
        for r in range(restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random initial set
            # Pick n-1 random distinct integers from 1 to L
            # L should be smallish to keep max(A) small initially
            # Let's say L = 2 * n * 2 = 4n
            L = 4 * n
            candidates = list(range(1, L + 1))
            random.shuffle(candidates)
            initial_set = [0] + candidates[:n-1]
            initial_set.sort()
            
            # Run local search
            found_set, found_score = local_search(initial_set, max_iters=100)
            
            if found_score > best_global_score:
                best_global_score = found_score
                best_global_set = found_set
                
    # One final check with a slightly larger bound for the best set found
    # Maybe the local search got stuck. Let's do a broader search around the best set.
    if best_global_score > -1:
        # Use a larger bound for the final polish
        final_A, final_score = local_search(best_global_set, max_iters=200)
        if final_score > best_global_score:
            best_global_score = final_score
            best_global_set = final_A

    return best_global_set
