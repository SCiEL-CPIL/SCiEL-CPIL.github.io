import numpy as np
import time

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak (discrete convolution scaled by DELTA)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums are distinct)."""
    sums = set()
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_max_sidon_set(n, max_k=None):
    """Find a Sidon set of maximum size in [0, n-1]."""
    if max_k is None:
        # Upper bound is approx sqrt(n) + 1
        max_k = int(np.sqrt(n)) + 2
    
    best_set = None
    # Try to find a set of size max_k, then max_k-1, etc.
    for k in range(max_k, 1, -1):
        # Use a simple greedy/backtracking approach or random restarts
        # For N=50, k is small (<= 8), so we can use a simple backtracking or even exhaustive search with pruning if needed.
        # But random restarts with greedy addition is faster.
        
        found = False
        for _ in range(1000): # 1000 restarts
            current_set = []
            # Start with 0
            current_set.append(0)
            used_sums = {0} # 0+0
            
            # Greedily add elements
            # To speed up, we can shuffle the candidate range
            candidates = list(range(1, n))
            np.random.shuffle(candidates)
            
            for c in candidates:
                if len(current_set) == k:
                    break
                # Check if adding c maintains Sidon property
                valid = True
                new_sums = set()
                for s in current_set:
                    new_sums.add(s + c)
                    if s + c in used_sums:
                        valid = False
                        break
                # Also check c+c
                if c + c in used_sums:
                    valid = False
                # Check new_sums against each other? No, sums involving c are distinct if c is distinct and previous sums were distinct?
                # Wait, s1+c and s2+c are distinct if s1!=s2.
                # But s1+c might equal s3+s4? No, we only added one element c.
                # The new sums are {s+c for s in current_set} U {c+c}.
                # We need to check if any of these are in used_sums.
                # We also need to check if any new sum equals another new sum.
                # s1+c == s2+c implies s1==s2.
                # s1+c == c+c implies s1==c. (c is not in current_set)
                # So new sums are distinct from each other.
                # We just need to check against used_sums.
                
                if valid:
                    current_set.append(c)
                    used_sums.update(new_sums)
                    used_sums.add(c+c) # Wait, c+c is already in new_sums if c is in current_set? No, c is not in current_set yet.
                    # Correction: new_sums contains s+c for s in current_set.
                    # c+c is a separate sum.
                    # So we add all s+c and c+c to used_sums.
                    # My loop above added s+c. I need to add c+c.
                    # Let's re-verify the logic.
                    
            if len(current_set) == k:
                best_set = current_set
                found = True
                break
        
        if found:
            return best_set
            
    return None

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # Try to find a Sidon set
    # Theoretical max size for N=50 is around 7-8.
    best_h = np.ones(N) / N # Fallback to uniform
    best_score = _score_raw(best_h)
    
    # Search for Sidon sets of size K from high to low
    # Start with K=8, 7, 6...
    for K in range(8, 2, -1):
        sidon_set = _find_max_sidon_set(N, K)
        if sidon_set is not None:
            h_candidate = np.zeros(N)
            for idx in sidon_set:
                h_candidate[idx] = 1.0 / K
            score = _score_raw(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate
            # If we found a good one, we might break or continue to see if larger K is possible?
            # The loop goes from high K to low. So if we find K=8, we should check if K=9 is possible?
            # My loop is `for K in range(8, 2, -1)`. It starts at 8.
            # If 8 fails, it tries 7.
            # So the first success is the highest K found in this range.
            # We should break after finding the first successful K?
            # Yes, because larger K gives higher score (Score = K*N).
            break

    # If no Sidon set found (unlikely for K>=3), we might try other structures?
    # But Sidon sets are the optimal structure for this specific "sparse" regime.
    # For dense regimes, uniform is optimal.
    # The score for Sidon K is K*N.
    # The score for Uniform is 1*N (since L2^2=1/N, M=1/N, Score = 1/N / (1/N) = 1? No.
    # Uniform Score: L2^2 = 1/N. M_conv = 1/N. Score = (1/N) / (1/N) = 1?
    # Wait, Score = L2^2 / (L1 * Delta * M_conv).
    # Uniform: L2^2 = 1/N. M_conv = 1/N.
    # Score = (1/N) / (1 * Delta * 1/N) = 1/Delta = 2N.
    # Sidon K: L2^2 = 1/K. M_conv = 2/K^2.
    # Score = (1/K) / (1 * Delta * 2/K^2) = K / (2 Delta) = K N.
    
    # So Uniform gives 2N.
    # Sidon K gives K N.
    # We need K > 2 to beat Uniform.
    # K=3 gives 3N = 150.
    # K=7 gives 7N = 350.
    
    return list(best_h)
