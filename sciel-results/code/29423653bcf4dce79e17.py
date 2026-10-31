"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # FFT-based computation
        # Size of U is n
        n = len(U)
        # Max sum is 2*max_u
        # Max diff magnitude is max_u
        # We need to compute U+U and U-U
        
        # Determine FFT size
        # Convolution length needs to be at least len(A) + len(B) - 1
        # For U+U: max index 2*max_u. Length 2*max_u + 1.
        # For U-U: we compute U + (M - U). Max index M + M = 2M.
        # So N >= 2*max_u + 1.
        
        # Next power of 2
        N = 1
        while N < 2 * max_u + 1:
            N *= 2
        
        A = np.zeros(N)
        for u in U:
            A[u] = 1.0
            
        # For difference:
        # V = {M - u for u in U}
        B = np.zeros(N)
        for u in U:
            B[max_u - u] = 1.0
            
        # Sumset
        FA = np.fft.rfft(A)
        FA2 = FA * FA
        C_sum = np.fft.irfft(FA2, N)
        
        # Diffset (via U + V)
        FB = np.fft.rfft(B)
        FA_FB = FA * FB
        C_diff = np.fft.irfft(FA_FB, N)
        
        # Count non-zeros
        # Tolerance for floating point
        n_sum = np.sum(np.abs(C_sum) > 0.5)
        n_diff = np.sum(np.abs(C_diff) > 0.5)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Phase 1: Initial construction
    # Start with a small set and grow it
    current = [0, 1, 3]
    try_set(current)
    
    # Local Search
    # We will try to add, remove, or swap elements.
    # With FFT, we can do this for larger sets.
    
    no_improve = 0
    max_no_improve = 100
    max_size = 200 # Allow sets to grow
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        current = list(best_set)
        n = len(current)
        max_cur = max(current)
        
        # 1. Try adding an element
        if n < max_size:
            # Candidate positions:
            # - Near existing elements
            # - In gaps
            # - At the end (extending the range)
            
            candidates = set()
            # Extend range
            candidates.update(range(max_cur + 1, max_cur + 20))
            # Near existing
            for x in current:
                for dx in range(-5, 6):
                    nx = x + dx
                    if nx > 0: # Must be non-negative, 0 is already there
                        candidates.add(nx)
            
            # Sort candidates by some heuristic? e.g. distance to existing?
            # Just iterate
            for new_el in sorted(candidates):
                if new_el in current:
                    continue
                cand = sorted(current + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    improved = True
                    break
            
        # 2. Try removing an element
        if n > 3:
            # Remove element that contributes least to ratio?
            # Just try all
            for i in range(1, n): # Skip 0
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    improved = True
                    break
        
        # 3. Try swapping an element
        if n > 3:
            for i in range(1, n):
                old_el = current[i]
                # Try replacing with nearby values
                # Range: old_el +/- 10
                for new_el in range(max(1, old_el - 10), old_el + 11):
                    if new_el in current:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        improved = True
                        break
                if improved:
                    break
                    
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return list(best_set)
# EVOLVE-BLOCK-END
