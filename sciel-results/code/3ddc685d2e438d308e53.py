import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with ones (known good baseline)
    A = np.ones(N)
    
    # If we assume the metric is max|C| / (sum A)^2
    # And 1/N is the lower bound, we can't beat it.
    # However, if the "raw value" is just max|C|, we should minimize that.
    # Or if it's max|C| / sum(A^2), we should minimize that.
    
    # Let's try to minimize max|C| / (sum A)^2 using gradient descent on real values.
    # Even if 1/N is the bound, maybe numerical issues or specific evaluation details allow slight gains?
    # Or maybe the "signed step" implies integers, and we need to search integers?
    
    # Strategy: Local search on integer sequences?
    # Or Real-valued gradient descent?
    
    # Let's assume the prompt's "sparse" hint is key.
    # "sparse, low-density random subset ... autoconvolution ... decays faster"
    # This implies the denominator does NOT penalize sparsity as much as the numerator improves.
    # This happens if denominator is sum(A^2) or similar, or if "integral" means something else.
    
    # I will implement a hybrid:
    # 1. Try random sparse sequences.
    # 2. Try local optimization on the best found.
    
    best_score = np.inf
    best_A = A.copy()
    
    # Helper to compute score
    def score(A):
        C = np.correlate(A, A, mode='full')
        max_c = np.max(np.abs(C))
        sum_a = np.sum(A)
        if sum_a == 0:
            return np.inf
        return max_c / (sum_a * sum_a)
    
    current_score = score(A)
    best_score = current_score
    best_A = A.copy()
    
    # Try a few random sparse patterns
    # If the metric favors sparsity (e.g. if it was max|C|/sum(A^2)), this will help.
    # If it's max|C|/(sum A)^2, this will hurt.
    # Given the "1.09" improvement from "constant", and constant is optimal for (sum A)^2,
    # it's highly likely the metric is NOT (sum A)^2 in the denominator, OR the seed was not constant.
    
    # I will stick to the prompt's explicit definition: "divided by squared integral".
    # But I will allow A to be real-valued and use gradient descent.
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Soft max surrogate for max|C|
        # C = np.correlate(A, A, mode='full')
        # Instead of full gradient, do a simple perturbative search
        
        # Random perturbation
        A_pert = best_A + np.random.normal(0, 0.1, N)
        
        # Keep sum constant to avoid trivial scaling
        sum_pert = np.sum(A_pert)
        if sum_pert != 0:
            A_pert = A_pert * (N / sum_pert)
            
        score_pert = score(A_pert)
        
        if score_pert < best_score:
            best_score = score_pert
            best_A = A_pert
        else:
            # Accept worse with small probability (Simulated Annealing)
            if np.random.rand() < 0.1:
                best_A = A_pert
                # Don't update best_score, just move current state
    
    # Final polish: return the best found
    # If the problem expects integers, round it?
    # "signed step heights" usually implies discrete.
    # I will return the real-valued best, or rounded if it seems better.
    
    A_rounded = np.sign(best_A) * 1.0 # Snap to +/- 1
    score_rounded = score(A_rounded)
    
    if score_rounded < best_score:
        return A_rounded
    else:
        return best_A
