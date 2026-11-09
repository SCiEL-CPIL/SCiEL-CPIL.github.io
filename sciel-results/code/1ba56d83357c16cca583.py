import numpy as np
import random

def construct():
    n = 128
    targets_start = 128
    targets_end = 256
    all_candidates = list(range(0, n + 1))
    
    best_A = None
    best_peak = float('inf')
    
    for seed in range(10):
        random.seed(seed)
        A = set()
        rep_counts = np.zeros(2 * n + 1, dtype=int)
        
        # Greedy Construction
        for _ in range(n + 1):
            if np.all(rep_counts[targets_start:targets_end + 1] > 0):
                break
            
            best_x = None
            best_score = float('inf')
            
            for x in all_candidates:
                if x in A:
                    continue
                
                new_counts = rep_counts.copy()
                for a in A:
                    s = x + a
                    new_counts[s] += 1
                s2 = 2 * x
                new_counts[s2] += 1
                
                sub_counts = new_counts[targets_start:targets_end + 1]
                
                if np.any(sub_counts == 0):
                    uncovered = np.sum(sub_counts == 0)
                    peak = np.max(sub_counts)
                    score = peak + uncovered * 1000
                else:
                    peak = np.max(sub_counts)
                    score = peak
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            A.add(best_x)
            for a in A:
                s = best_x + a
                rep_counts[s] += 1
            s2 = 2 * best_x
            rep_counts[s2] += 1
        
        # Local Search
        improved = True
        while improved:
            improved = False
            
            # Try Removal
            for x in list(A):
                test_counts = rep_counts.copy()
                for a in A:
                    if a == x:
                        continue
                    s = x + a
                    test_counts[s] -= 1
                s2 = 2 * x
                test_counts[s2] -= 1
                
                sub_counts = test_counts[targets_start:targets_end + 1]
                if np.any(sub_counts == 0):
                    continue
                
                peak = np.max(sub_counts)
                if peak < best_peak:
                    best_peak = peak
                    best_A = set(A)
                    best_A.remove(x)
                    A.remove(x)
                    rep_counts = test_counts
                    improved = True
                    break
            
            # Try Swap
            if not improved:
                for x in list(A):
                    # Sample a few y's
                    candidates_y = [y for y in all_candidates if y not in A]
                    if not candidates_y:
                        continue
                    for y in random.sample(candidates_y, min(10, len(candidates_y))):
                        test_counts = rep_counts.copy()
                        
                        # Remove x
                        for a in A:
                            if a == x:
                                continue
                            s = x + a
                            test_counts[s] -= 1
                        s2 = 2 * x
                        test_counts[s2] -= 1
                        
                        # Add y
                        for a in A:
                            if a == x:
                                continue
                            s = y + a
                            test_counts[s] += 1
                        s2y = 2 * y
                        test_counts[s2y] += 1
                        
                        sub_counts = test_counts[targets_start:targets_end + 1]
                        if np.any(sub_counts == 0):
                            continue
                        
                        peak = np.max(sub_counts)
                        if peak < best_peak:
                            best_peak = peak
                            best_A = set(A)
                            best_A.remove(x)
                            best_A.add(y)
                            A.remove(x)
                            A.add(y)
                            rep_counts = test_counts
                            improved = True
                            break
                    if improved:
                        break

    if best_A:
        return list(best_A)
    return list(range(0, n + 1, 2))
