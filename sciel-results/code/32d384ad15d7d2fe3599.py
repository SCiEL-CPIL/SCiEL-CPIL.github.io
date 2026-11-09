"""Heuristic based on multi-start 2-opt with OR-opt and frequency-based consensus."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    diag = np.eye(n, dtype=bool)

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=100):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline - 5.0:
                break
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = t[i], t[i + 1]
                    c, d = t[j], t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    def or_opt(t, max_passes=50):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline - 5.0:
                break
            for i in range(n):
                node = t[i]
                prev = t[(i - 1) % n]
                next_ = t[(i + 1) % n]
                best_j = -1
                best_delta = 0.0
                for j in range(n):
                    if j == i or j == (i - 1) % n or j == (i + 1) % n:
                        continue
                    a = t[j]
                    b = t[(j + 1) % n]
                    delta = (D[prev, node] + D[node, next_] + D[a, b]) - (D[prev, a] + D[a, node] + D[node, b])
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_j = j
                if best_j >= 0:
                    t.pop(i)
                    t.insert(best_j + 1, node)
                    improved = True
        return t

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    def random_tour(seed):
        rng = np.random.default_rng(seed)
        t = list(range(n))
        rng.shuffle(t)
        return t

    rng = np.random.default_rng(12345)
    tours = []
    
    # Generate diverse tours
    # 1. Nearest neighbor from all nodes
    for s in range(n):
        if time.time() > deadline - 30.0:
            break
        t = nearest_neighbor(s)
        t = two_opt(t, max_passes=100)
        t = or_opt(t, max_passes=50)
        tours.append(t)
        
    # 2. Add random starts to catch different local optima
    num_random_starts = 80
    for i in range(num_random_starts):
        if time.time() > deadline - 30.0:
            break
        s = int(rng.integers(n))
        t = nearest_neighbor(s)
        t = two_opt(t, max_passes=100)
        t = or_opt(t, max_passes=50)
        tours.append(t)
        
    # 3. Add some fully random tours
    num_full_random = 30
    for i in range(num_full_random):
        if time.time() > deadline - 30.0:
            break
        t = random_tour(1000 + i)
        t = two_opt(t, max_passes=100)
        t = or_opt(t, max_passes=50)
        tours.append(t)

    if not tours:
        t = nearest_neighbor(0)
        t = two_opt(t, max_passes=100)
        t = or_opt(t, max_passes=50)
        tours.append(t)

    # Sort tours by length
    tours_with_len = [(tour_len(t), t) for t in tours]
    tours_with_len.sort(key=lambda x: x[0])
    
    # Use the top 60% of tours to build the consensus
    num_top = max(1, int(len(tours_with_len) * 0.6))
    top_tours = [t for _, t in tours_with_len[:num_top]]
    
    # Frequency matrix with weighted contribution (better tours contribute more)
    freq = np.zeros((n, n), dtype=float)
    for rank, t in enumerate(top_tours):
        weight = 1.0 / (1.0 + rank * 0.1)  # Better tours get higher weight
        for i in range(n):
            u = t[i]
            v = t[(i + 1) % n]
            freq[u, v] += weight

    # Base heuristic: inverse distance
    safe_D = D.copy()
    safe_D[diag] = 1.0
    base = 1.0 / safe_D
    
    # Combine frequency and inverse distance
    # Heuristic = (freq + 1) * base^alpha
    alpha = 1.8
    H = (freq + 1.0) * (base ** alpha)
    
    # Ensure no zeros or non-finite values
    H[diag] = 1e-9
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
