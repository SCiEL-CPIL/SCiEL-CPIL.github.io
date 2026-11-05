import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    target_lo, target_hi = 128, 256
    target_range = range(target_lo, target_hi + 1)
    
    def compute_mults(A_mask):
        # A_mask is bool array of length 129
        # Compute number of unordered pairs (a,b) with a<=b, a+b=t
        # Using convolution: A * A gives ordered pairs. 
        # Unordered with equal: mult[t] = (conv[t] + (A_mask[t//2] if t%2==0 else 0)) // 2
        conv = np.convolve(A_mask.astype(np.int32), A_mask.astype(np.int32), mode='full')
        # conv[t] = sum_{a+b=t} A[a]*A[b] (ordered)
        # For unordered: if t is even, the pair (t/2, t/2) is counted once in ordered, 
        # but in unordered it's one pair. In ordered, it's 1. In unordered, it's 1.
        # Actually: ordered count = 2 * (unordered count with a<b) + (1 if t even and A[t/2] else 0)
        # So unordered = (ordered + (1 if t even and A[t/2] else 0)) // 2
        mults = np.zeros(target_hi + 1, dtype=np.int32)
        for t in target_range:
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and A_mask[t // 2]) else 0
            mults[t] = (ordered + diag) // 2
        return mults
    
    def eval_objective(A_mask):
        mults = compute_mults(A_mask)
        target_mults = mults[target_lo:target_hi+1]
        if np.min(target_mults) == 0:
            return float('inf'), mults
        return int(np.max(target_mults)), mults
    
    # Initialize: random A
    random.seed(42)
    best_A = np.zeros(n+1, dtype=bool)
    best_score = float('inf')
    
    # Try several random initializations
    for init_iter in range(20):
        if time.time() > deadline:
            break
        density = random.uniform(0.25, 0.45)
        A_mask = np.random.random(n+1) < density
        # Ensure non-empty
        if not A_mask.any():
            A_mask[random.randint(0, n)] = True
        
        score, _ = eval_objective(A_mask)
        if score < best_score:
            best_score = score
            best_A = A_mask.copy()
    
    # Local search on best_A
    A_mask = best_A.copy()
    current_score = best_score
    
    # Precompute for speed
    def quick_eval(A_mask):
        conv = np.convolve(A_mask.astype(np.int32), A_mask.astype(np.int32), mode='full')
        target_mults = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
        for i, t in enumerate(target_range):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and A_mask[t // 2]) else 0
            target_mults[i] = (ordered + diag) // 2
        if np.min(target_mults) == 0:
            return float('inf')
        return int(np.max(target_mults))
    
    current_score = quick_eval(A_mask)
    
    # Hill climbing
    no_improve = 0
    while time.time() < deadline and no_improve < 200:
        # Pick a random index to flip
        idx = random.randint(0, n)
        old_val = A_mask[idx]
        A_mask[idx] = not old_val
        
        new_score = quick_eval(A_mask)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_A = A_mask.copy()
        else:
            # Revert
            A_mask[idx] = old_val
            no_improve += 1
    
    # Final check
    final_score = quick_eval(best_A)
    if final_score < best_score:
        best_score = final_score
    
    return [i for i in range(n+1) if best_A[i]]
