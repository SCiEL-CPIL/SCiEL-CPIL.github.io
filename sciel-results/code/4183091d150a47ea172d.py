import numpy as np
import time

N = 50

def _score(h):
    h = np.array(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # Autoconvolution: for discrete step function, sup is at lag 0
    # autoconv[k] = sum_i h[i]*h[i+k]
    # Sup = max over k of autoconv[k]
    # For nonnegative h, sup is at k=0 = l2_sq
    # But wait, maybe the problem defines it differently
    # Let's compute all lags
    autoconv = np.correlate(h, h, mode='full')
    # The full correlation has length 2N-1, centered at lag 0
    sup_autoconv = np.max(autoconv)
    if l1 == 0 or sup_autoconv == 0:
        return 0.0
    return l2_sq / (l1 * sup_autoconv)

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Greedy Sidon set
    sidon = []
    used_sums = set()
    for i in range(N):
        new_sums = set()
        for j in range(len(sidon)):
            new_sums.add(i + sidon[j])
        new_sums.add(2 * i)
        if not (new_sums & used_sums):
            sidon.append(i)
            used_sums |= new_sums
    
    h = [0.0] * N
    for idx in sidon:
        h[idx] = 1.0
    
    best_score = _score(h)
    best_h = list(h)
    
    # Local search: try swapping elements
    import random
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(N):
                if i == j:
                    continue
                # Try moving spike from i to j
                if h[i] > 0 and h[j] == 0:
                    h[i] = 0.0
                    h[j] = 1.0
                    s = _score(h)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_h = list(h)
                        improved = True
                    else:
                        h[j] = 0.0
                        h[i] = 1.0
                elif h[i] == 0 and h[j] > 0:
                    # Try adding spike at i (increase count)
                    h[i] = 1.0
                    s = _score(h)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_h = list(h)
                        improved = True
                    else:
                        h[i] = 0.0
                # Try removing spike at i
                if h[i] > 0:
                    h[i] = 0.0
                    s = _score(h)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_h = list(h)
                        improved = True
                    else:
                        h[i] = 1.0
        
        if not improved:
            # Try random restarts with different Sidon sets
            for _ in range(10):
                if time.time() > deadline:
                    break
                sidon2 = []
                used_sums2 = set()
                order = list(range(N))
                random.shuffle(order)
                for i in order:
                    new_sums = set()
                    for j in range(len(sidon2)):
                        new_sums.add(i + sidon2[j])
                    new_sums.add(2 * i)
                    if not (new_sums & used_sums2):
                        sidon2.append(i)
                        used_sums2 |= new_sums
                
                h2 = [0.0] * N
                for idx in sidon2:
                    h2[idx] = 1.0
                s2 = _score(h2)
                if s2 > best_score + 1e-12:
                    best_score = s2
                    best_h = list(h2)
                    improved = True
            if not improved:
                break
    
    return best_h
