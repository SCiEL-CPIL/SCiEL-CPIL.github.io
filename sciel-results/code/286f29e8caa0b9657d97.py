"""Direct heuristic: build a high-quality reference tour, then strongly promote
its edges in the attractiveness matrix so the colony follows a good skeleton."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=50):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
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

    # Build a pool of high-quality tours to establish a consensus heuristic
    rng = np.random.default_rng(0)
    pool = []
    
    def add_to_pool(tour):
        t = two_opt(tour)
        pool.append(t)
        return t

    # 1. Nearest-neighbor from all nodes
    for s in range(n):
        if time.time() > deadline - 10.0:
            break
        add_to_pool(nearest_neighbor(s))
        
    # 2. Random starts for diversity
    num_random = 50
    for _ in range(num_random):
        if time.time() > deadline - 10.0:
            break
        s = int(rng.integers(n))
        add_to_pool(nearest_neighbor(s))

    # 3. Perturb best tours and re-optimize
    if pool:
        # Sort pool by length, keep top 10
        pool.sort(key=tour_len)
        top_tours = pool[:10]
        for base_t in top_tours:
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                t = list(base_t)
                # Apply random 2-opt moves
                for _ in range(3):
                    i, j = int(rng.integers(n)), int(rng.integers(n))
                    if i > j:
                        i, j = j, i
                    if j - i >= 1:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                add_to_pool(t)

    if not pool:
        # Fallback
        pool.append(two_opt(nearest_neighbor(0)))

    # Build the attractiveness matrix based on edge frequency in the pool
    # Use exponential weighting to emphasize better tours
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    base = 1.0 / safe
    
    # Compute tour lengths
    lengths = [tour_len(t) for t in pool]
    min_len = min(lengths)
    
    # Weighted frequency: better tours get higher weight
    # Use exponential decay: weight = exp(-alpha * (L - L_min))
    alpha = 5.0  # Sharper focus on best tours
    weights = np.exp(-alpha * (np.array(lengths) - min_len))
    
    freq = np.zeros((n, n), dtype=float)
    for t, w in zip(pool, weights):
        for k in range(n):
            i = t[k]
            j = t[(k + 1) % n]
            freq[i, j] += w
            
    # Scale the frequency contribution
    # Use a stronger multiplier to make the heuristic more decisive
    base = base * (1.0 + 10.0 * freq)
    
    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
# EVOLVE-BLOCK-END
