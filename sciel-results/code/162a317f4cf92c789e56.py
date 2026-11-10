import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 600
    
    # Fallback: R=2 set
    # Vectors with two 1s and rest 0s, or one 1 and one -1, etc.
    # Actually, all vectors with exactly two non-zero entries which are +/- 1.
    r2_points = []
    for i in range(d):
        for j in range(i+1, d):
            # Signs for i and j
            for s_i in [1, -1]:
                for s_j in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s_i
                    v[j] = s_j
                    r2_points.append(v)
    r2_points = np.array(r2_points)
    best_points = r2_points
    best_k = len(r2_points)
    
    # Heuristic search for R=3
    # Vectors with three 1s (or -1s)
    # Condition: dot <= 1
    r3_candidates = []
    for comb in combinations(range(d), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[comb[0]] = s1
                    v[comb[1]] = s2
                    v[comb[2]] = s3
                    r3_candidates.append(v)
    r3_candidates = np.array(r3_candidates)
    
    # Greedy selection for R=3
    selected_r3 = []
    for v in r3_candidates:
        if time.time() > deadline - 60:
            break
        if not selected_r3:
            selected_r3.append(v)
            continue
        M = np.array(selected_r3)
        dots = M @ v
        if np.max(dots) <= 1:
            selected_r3.append(v)
            
    if len(selected_r3) > best_k:
        best_points = np.array(selected_r3)
        best_k = len(selected_r3)
        
    # Heuristic search for R=4 (Type C: four 1s)
    # Condition: dot <= 2
    r4_candidates = []
    for comb in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[comb[0]] = s1
                        v[comb[1]] = s2
                        v[comb[2]] = s3
                        v[comb[3]] = s4
                        r4_candidates.append(v)
    r4_candidates = np.array(r4_candidates)
    
    selected_r4 = []
    for v in r4_candidates:
        if time.time() > deadline - 60:
            break
        if not selected_r4:
            selected_r4.append(v)
            continue
        M = np.array(selected_r4)
        dots = M @ v
        if np.max(dots) <= 2:
            selected_r4.append(v)
            
    if len(selected_r4) > best_k:
        best_points = np.array(selected_r4)
        best_k = len(selected_r4)

    # Final check and formatting
    points = np.around(best_points).astype(np.int64)
    return points
