import numpy as np
import time
import random

N = 50

def find_sidon_set(N, K, attempts=1000):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        # Randomize the order of candidates to avoid bias
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            # Check if adding cand creates a collision
            conflict = False
            new_sums = set()
            
            # Check cand + cand
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            # Check cand + s for s in S
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums:
                        conflict = True
                        break
                    if sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_h = np.ones(N) / N
    best_ratio = np.sum(best_h**2) / (np.sum(best_h) * np.max(np.convolve(best_h, best_h)))
    
    # Search for largest Sidon set
    # Theoretical max K is around sqrt(N) * 1.5 or so, but bounded by sum range.
    # Let's try from 12 down to 5.
    for K in range(12, 4, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Increase attempts as K gets smaller (easier to find)
        # But we want the largest, so we try larger K first with reasonable attempts.
        attempts = 500 if K > 8 else 1000
        
        S = find_sidon_set(N, K, attempts)
        
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            ratio = np.sum(h**2) / (np.sum(h) * np.max(np.convolve(h, h)))
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
                # If we found a large Sidon set, it's likely optimal for this K.
                # We can break if K is large enough, but let's keep searching for larger?
                # No, we are iterating K downwards. So if we find K=10, we record it.
                # Then we try K=9? No, we want the MAX ratio.
                # Ratio = K for Sidon sets. So larger K is always better.
                # So if we find K=10, we should stop?
                # Wait, is it possible that a non-Sidon set of size 12 gives ratio > 10?
                # Unlikely, but possible if collisions are very mild.
                # However, Sidon sets are the "ideal" case.
                # Let's just find the largest Sidon set possible.
                # If we find K=10, we can stop searching for smaller Sidon sets?
                # Yes, because any smaller Sidon set gives lower ratio.
                # And non-Sidon sets are suboptimal compared to Sidon sets of same size?
                # Actually, a non-Sidon set of size 11 might beat a Sidon set of size 10?
                # Ratio for Sidon 10 is 10.
                # Ratio for non-Sidon 11?
                # If collisions are low, it might be close to 11.
                # But finding a non-Sidon set with optimized weights is harder.
                # Given the time limit, finding the largest Sidon set is the most robust strategy.
                pass

    # If we found a Sidon set, we are done.
    # If not, fallback to SLSQP?
    # The loop above covers K=12..5.
    # If none found (very unlikely for K>=5), best_h remains uniform.
    
    # One more check: try to optimize weights for the best Sidon set found?
    # We established equal weights are optimal.
    
    return list(best_h)
