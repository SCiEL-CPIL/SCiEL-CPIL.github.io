import itertools
import numpy as np

D = 11

def construct():
    # 1. Generate weight-3 vectors
    w3_candidates = []
    for indices in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[indices[0]] = s1
                    v[indices[1]] = s2
                    v[indices[2]] = s3
                    w3_candidates.append(v)
                    
    # 2. Greedily select weight-3 vectors
    selected_w3 = []
    for v in w3_candidates:
        compatible = True
        for u in selected_w3:
            if sum(ui * vi for ui, vi in zip(u, v)) > 1:
                compatible = False
                break
        if compatible:
            selected_w3.append(v)
            
    # 3. Generate weight-2 vectors
    w2_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                w2_candidates.append(v)
                
    # 4. Select weight-2 vectors compatible with selected_w3 and each other
    selected_w2 = []
    for w in w2_candidates:
        # Check against selected_w3
        ok_w3 = True
        for v in selected_w3:
            if sum(wi * vi for wi, vi in zip(w, v)) > 1:
                ok_w3 = False
                break
        if not ok_w3:
            continue
            
        # Check against selected_w2
        ok_w2 = True
        for u in selected_w2:
            if sum(wi * ui for wi, ui in zip(w, u)) > 0:
                ok_w2 = False
                break
        if ok_w2:
            selected_w2.append(w)
            
    # Combine
    all_vectors = selected_w3 + selected_w2
    return np.array(all_vectors, dtype=np.int64)
