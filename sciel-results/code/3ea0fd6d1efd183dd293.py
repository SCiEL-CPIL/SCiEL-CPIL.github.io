import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # LCM of 1 to 30
    # 2329089562800
    L = 2329089562800
    A = [L // i for i in range(1, 31)]
    
    best_K = float('inf')
    best_coeffs = [0] * 28 + [1, -1] # Fallback: 1/29 - 1/30
    
    def eval_K(coeffs):
        k = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                k += c * A[i]
        return k
    
    # Initial greedy solution
    # Try to balance sum to 0
    def greedy_init():
        coeffs = [0] * 30
        current_sum = 0
        # Sort indices by A[i] descending (which is 0, 1, 2... so index 0 is largest)
        # A[0] = L/1, A[1] = L/2 ...
        for i in range(30):
            # Try -1, 0, 1
            best_c = 0
            best_abs = abs(current_sum)
            for c in [-1, 0, 1]:
                new_sum = current_sum + c * A[i]
                if abs(new_sum) < best_abs:
                    best_abs = abs(new_sum)
                    best_c = c
            coeffs[i] = best_c
            current_sum += best_c * A[i]
        return coeffs, current_sum

    init_coeffs, init_K = greedy_init()
    if abs(init_K) < abs(best_K) and init_K != 0:
        best_K = init_K
        best_coeffs = init_coeffs
    
    # Local search
    def local_search(coeffs, K):
        nonlocal best_K, best_coeffs
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomize order of indices to avoid bias
            indices = list(range(30))
            random.shuffle(indices)
            for i in indices:
                if time.time() >= deadline:
                    break
                current_c = coeffs[i]
                # Try the other two values
                for new_c in [-1, 0, 1]:
                    if new_c == current_c:
                        continue
                    delta = (new_c - current_c) * A[i]
                    new_K = K + delta
                    if new_K == 0:
                        # We want non-zero. 
                        # If we are currently far, moving to 0 is good progress towards smallness, 
                        # but we can't stop there. 
                        # We can allow moving to 0 if it reduces magnitude, 
                        # but then we must immediately move away.
                        # For simplicity, skip 0 in the acceptance criterion 
                        # unless we are stuck? 
                        # Actually, if we skip 0, we might miss the path to 1.
                        # Example: K=10, neighbor is 0, neighbor of 0 is 1.
                        # If we skip 0, we can't get to 1 via this path.
                        # So we should allow moving to 0.
                        # But we must ensure we don't RETURN 0.
                        # So: Accept if abs(new_K) < abs(K).
                        # If new_K == 0, accept it.
                        # The loop will continue, and from 0, it will try to move to 1, -1, etc.
                        pass 
                    
                    if abs(new_K) < abs(K):
                        # Accept
                        coeffs[i] = new_c
                        K = new_K
                        improved = True
                        # Update global best if non-zero
                        if K != 0 and abs(K) < abs(best_K):
                            best_K = K
                            best_coeffs = coeffs[:]
                        break # Restart scanning from index 0 (or next in shuffled list)
        
        # If we ended at K=0, we need to find the best non-zero neighbor
        if K == 0:
            # Find smallest non-zero K reachable by one flip
            min_neighbor = float('inf')
            best_neighbor_coeffs = None
            for i in range(30):
                for new_c in [-1, 0, 1]:
                    if new_c == coeffs[i]:
                        continue
                    delta = (new_c - coeffs[i]) * A[i]
                    new_K = delta # Since K was 0
                    if new_K != 0 and abs(new_K) < abs(min_neighbor):
                        min_neighbor = new_K
                        neighbor_coeffs = coeffs[:]
                        neighbor_coeffs[i] = new_c
                        best_neighbor_coeffs = neighbor_coeffs
            if best_neighbor_coeffs is not None:
                if abs(min_neighbor) < abs(best_K):
                    best_K = min_neighbor
                    best_coeffs = best_neighbor_coeffs
        else:
            # If we ended at non-zero K, update best
            if abs(K) < abs(best_K):
                best_K = K
                best_coeffs = coeffs[:]

    # Run multiple restarts
    while time.time() < deadline:
        # 1. Greedy restart
        coeffs, K = greedy_init()
        local_search(coeffs, K)
        
        # 2. Random restarts
        # If we have time, do random initializations
        if time.time() < deadline - 10:
            coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
            K = eval_K(coeffs)
            local_search(coeffs, K)

    # Final check: ensure best_coeffs is valid
    # If best_K is still inf (shouldn't happen), use fallback
    if best_K == float('inf'):
        return [0] * 28 + [1, -1]
        
    return best_coeffs
