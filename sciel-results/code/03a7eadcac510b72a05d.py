import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580.0

    # Base: weight 2 vectors (220 points)
    base_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_list.append(p)
    base_points = np.array(base_list, dtype=np.int64)
    best_points = base_points.copy()
    best_k = base_points.shape[0]  # 220

    # Candidate generation for weight 3
    cands_w3 = []
    for combo in combinations(range(d), 3):
        for mask in range(1 << 3):
            v = np.zeros(d, dtype=np.int64)
            for idx, pos in enumerate(combo):
                v[pos] = 1 if (mask >> idx) & 1 else -1
            cands_w3.append(v)
    cands_w3 = np.array(cands_w3, dtype=np.int64)
    
    # Greedy selection for weight 3 vectors
    # Condition: dot(p, q) <= 1 for all pairs
    # This ensures dist^2 >= 4 >= max_norm^2 (which is 3)
    
    def greedy_select(cands, max_dot=1, seed=42):
        rng = random.Random(seed)
        indices = list(range(len(cands)))
        rng.shuffle(indices)
        
        selected = []
        selected_vectors = []
        
        for idx in indices:
            p = cands[idx]
            ok = True
            # Check against selected
            for q in selected_vectors:
                if np.dot(p, q) > max_dot:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_vectors.append(p)
                
        return selected, selected_vectors

    # Try multiple seeds for weight 3
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        sel_idx, sel_vecs = greedy_select(cands_w3, max_dot=1, seed=seed)
        if len(sel_vecs) > best_k:
            best_k = len(sel_vecs)
            best_points = np.array(sel_vecs, dtype=np.int64)

    # Try weight 4?
    # Norm^2 = 4. Max norm 4.
    # dist^2 >= 4.
    # dist^2 = 8 - 2 dot.
    # 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    # So for weight 4, we need dot(p, q) <= 2.
    
    cands_w4 = []
    for combo in combinations(range(d), 4):
        for mask in range(1 << 4):
            v = np.zeros(d, dtype=np.int64)
            for idx, pos in enumerate(combo):
                v[pos] = 1 if (mask >> idx) & 1 else -1
            cands_w4.append(v)
    cands_w4 = np.array(cands_w4, dtype=np.int64)
    
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        sel_idx, sel_vecs = greedy_select(cands_w4, max_dot=2, seed=seed)
        if len(sel_vecs) > best_k:
            best_k = len(sel_vecs)
            best_points = np.array(sel_vecs, dtype=np.int64)

    return np.array(best_points, dtype=np.int64)
