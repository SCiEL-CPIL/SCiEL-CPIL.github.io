import random
import time

def construct():
    # Structural construction: Small dense set S + Geometric tail G
    # S = {1, ..., 10}
    # G = {11 * 2^i for i in 0..9}
    
    S = list(range(1, 11))
    G = [11 * (2 ** i) for i in range(10)]
    
    A = sorted(S + G)
    
    # Verify constraints
    if len(A) != 20:
        # Fallback if logic changed
        A = list(range(1, 21))
        
    if max(A) > 10000:
        A = list(range(1, 21))

    def compute_scores(A):
        sumset = set()
        prodset = set()
        for i in range(20):
            for j in range(i, 20):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    current_score = compute_scores(A)
    start_time = time.time()
    timeout = 55
    
    # Local search to improve
    while time.time() - start_time < timeout:
        improved = False
        # Try swapping elements in S or G
        for i in range(20):
            if time.time() - start_time > timeout:
                break
            old_val = A[i]
            # Try small perturbations
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000 or new_val in A:
                    continue
                A[i] = new_val
                A_sorted = sorted(A)
                new_score = compute_scores(A_sorted)
                if new_score < current_score:
                    current_score = new_score
                    A = A_sorted
                    improved = True
                    break
                else:
                    A[i] = old_val
            if improved:
                break
        if not improved:
            break
            
    return A
