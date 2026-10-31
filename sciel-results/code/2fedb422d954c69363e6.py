import numpy as np
import time
import sys

N = 50

def is_sidon(A):
    """Check if A is a Sidon set (all sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set_greedy(n):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(n):
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

def compute_score(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * max|c|)
    where c is the autoconvolution of h.
    """
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.convolve(h, h, mode='full')
    max_c = np.max(np.abs(c))
    if max_c == 0:
        return 0.0
    return l2_sq / (h_sum * max_c)

def optimize_weights(sidon_set, max_time=5.0):
    """
    Optimize weights on the given Sidon set to maximize the score.
    Uses simple local search / coordinate descent.
    """
    K = len(sidon_set)
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0 / K  # Start with equal weights
    
    best_h = h.copy()
    best_score = compute_score(best_h)
    
    start = time.time()
    step = 0.1
    # Coordinate descent / local search
    # We want to maximize l2_sq / (sum * max_c)
    # Since sum is fixed (if we normalize), maximizing l2_sq is good, but max_c matters.
    # Actually, the ratio is scale invariant. 
    # Let's fix sum(h) = 1 to remove scale ambiguity.
    # Then we maximize sum(h^2) / max_c.
    
    # Normalize h to sum 1
    s = np.sum(h)
    if s > 0:
        h = h / s
    best_h = h.copy()
    best_score = compute_score(best_h)
    
    # Simple local search: try moving mass from i to j
    # This is O(K^2) per iteration. K is small (~7-8).
    # We can afford a few thousand iterations.
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        # Try all pairs (i, j) where i != j
        # Move delta from i to j
        # We need to ensure h_i - delta >= 0 and h_j + delta >= 0
        # Let's try a few delta values: 0.01, 0.02, ...
        
        # To speed up, only consider moves that might help.
        # But K is small, so brute force pairs is fine.
        
        for i in sidon_set:
            for j in sidon_set:
                if i == j:
                    continue
                # Try moving mass from i to j
                # Max delta is h[i]
                # We want to see if score improves.
                # Let's try a few step sizes
                for delta in [0.001, 0.005, 0.01, 0.02, 0.05]:
                    if delta > h[i]:
                        continue
                    h_new = h.copy()
                    h_new[i] -= delta
                    h_new[j] += delta
                    
                    # Check if valid (non-negative)
                    if np.any(h_new < 0):
                        continue
                    
                    score = compute_score(h_new)
                    if score > best_score + 1e-7:
                        best_score = score
                        best_h = h_new.copy()
                        improved = True
                        # Update current h to best found so far? 
                        # Or just update h to this better one?
                        h = h_new.copy()
                        break # Found a good move for this (i,j), try next? Or break out of delta loop?
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Find a large Sidon set
    A_greedy = find_sidon_set_greedy(N)
    best_A = A_greedy
    
    # Try to find a larger Sidon set with random restarts
    # Max size for N=50 is likely 7 or 8
    # Greedy usually finds a reasonably large one.
    # Let's try to find size 8, 9, etc.
    
    for target_size in range(len(A_greedy) + 1, 12):
        found = False
        # Limit tries to avoid timeout
        max_tries = 2000
        for _ in range(max_tries):
            if time.time() > deadline - 10:
                break
            A_try = sorted(np.random.choice(N, target_size, replace=False))
            if is_sidon(A_try):
                if target_size > len(best_A):
                    best_A = A_try
                    found = True
                    break # Found a valid one, move to next size? Or keep searching for better one?
                    # Usually we just want the largest size. 
                    # If we find one, we can stop for this size.
        
        if not found:
            # If we can't find size K, it's unlikely we find K+1.
            # But maybe we should try a bit harder?
            # For now, break.
            break
            
    # 2. Construct initial vector
    h = np.zeros(N)
    for idx in best_A:
        h[idx] = 1.0 / len(best_A)
        
    # 3. Optimize weights
    if time.time() < deadline - 5:
        time_budget = min(20.0, deadline - time.time() - 2)
        h_opt, score_opt = optimize_weights(best_A, max_time=time_budget)
        h = h_opt
        
    # 4. Final check and return
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    # Normalize? The problem says "nonnegative heights". 
    # The score is ratio, so scale doesn't matter. 
    # But the evaluator might expect specific scale? 
    # "maximize the ratio...". 
    # If I return h/sum(h), it's a probability distribution.
    # If I return h as is, it's fine.
    # Let's return the optimized weights.
    
    return list(h)

if __name__ == "__main__":
    # Simple test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {compute_score(np.array(res)):.4f}")
    print(f"Support size: {sum(1 for x in res if x > 0)}")
