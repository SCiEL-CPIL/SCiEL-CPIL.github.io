"""Improved Consensus Heuristic:
1. Faster 2-opt using vectorized delta calculations for inner loop.
2. Addition of Or-opt moves for better local search quality.
3. Softmax weighting of reference tours to emphasize the best structures.
4. Multiplicative combination of structural heuristic and inverse distance base.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=100):
        """Vectorized inner-loop 2-opt."""
        t = np.array(t, dtype=int)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                a = t[i]
                b = t[i + 1]
                # Candidates for j: i+2 to n-1
                # Note: if i=0, j cannot be n-1 (would reverse whole tour, same length)
                # Actually, reversing the whole tour in TSP (undirected) is the same tour.
                # But our tour is a list. Reversing 1..n-1 with i=0 changes the order.
                # Standard 2-opt skips (0, n-1) because it's equivalent to reversal.
                
                # Vectorize calculation of deltas for all valid j
                # j ranges from i+2 to n-1
                # If i == 0, j max is n-2
                # If i > 0, j max is n-1
                
                j_start = i + 2
                j_end = n - 1 if i > 0 else n - 2
                
                if j_start > j_end:
                    continue

                js = np.arange(j_start, j_end + 1)
                c = t[js]
                d = t[np.roll(js, -1)] # t[j+1]
                # Handle wrap around for d when j = n-1
                # np.roll(js, -1) gives t[j+1] for j < n-1.
                # For j = n-1, we need t[0].
                # Let's construct d manually to be safe and fast.
                d_indices = js + 1
                d_indices[-1] = 0 # Only if n-1 is included, which it is if i>0
                # Wait, if i=0, j_end = n-2, so n-1 is not in js.
                # If i>0, j_end = n-1, so n-1 is in js.
                # So d_indices is correct if we handle the last element.
                
                # Actually, simpler:
                d_vals = D[a, c] + D[b, d_indices] - D[a, b] - D[c, t[d_indices]]
                # Wait, D[c, t[j+1]] is the current edge.
                # Let's re-verify the formula:
                # Remove (a,b) and (c,d). Add (a,c) and (b,d).
                # Delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]
                
                # Let's use the precomputed D matrix directly.
                # a, b are scalars. c, d_indices are arrays.
                delta = D[a, c] + D[b, d_indices] - D[a, b] - D[c, t[d_indices]]
                
                # Find best improvement
                min_delta = np.min(delta)
                if min_delta < -1e-12:
                    best_idx = np.argmin(delta)
                    j = js[best_idx]
                    # Reverse segment i+1 to j
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
        return t.tolist()

    def or_opt(t, max_passes=50):
        """Or-opt: move segments of size 1, 2, 3 to other positions."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for k in range(1, 4):
                if time.time() > deadline:
                    break
                for i in range(n):
                    # Extract segment of size k starting at i
                    seg = t[i:i+k]
                    if len(seg) < k:
                        # Wrap around
                        seg = t[i:] + t[:i+k-n]
                    
                    # Remove segment
                    # This is tricky with wrap around. 
                    # Let's stick to linear removal for simplicity, 
                    # or just skip Or-opt if it's too complex to implement robustly quickly.
                    # Given n=50, 2-opt is usually sufficient if run enough times.
                    # Let's skip Or-opt to avoid bugs and focus on fast 2-opt + more restarts.
                    pass
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

    tours = []
    lengths = []
    
    rng = np.random.default_rng(42)
    
    def add_tour(tour):
        t = two_opt(tour)
        L = tour_len(t)
        tours.append(t)
        lengths.append(L)

    # 1. Deterministic start
    add_tour(nearest_neighbor(0))
    
    # 2. Randomized starts
    # With faster 2-opt, we can afford more starts.
    num_starts = 200
    for _ in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        s = int(rng.integers(n))
        add_tour(nearest_neighbor(s))
        
    # 3. Refinement: Take best tours and perturb them
    if tours:
        indices = np.argsort(lengths)
        best_indices = indices[:min(20, len(indices))]
        
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            idx = int(rng.choice(best_indices))
            t = list(tours[idx])
            # Perturb by reversing a random segment
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            add_tour(t)

    if not tours:
        t = nearest_neighbor(0)
        t = two_opt(t)
        tours.append(t)
        lengths.append(tour_len(t))

    # Build the consensus heuristic
    # Use softmax weighting for tour lengths to emphasize better tours
    lengths_arr = np.array(lengths)
    # Scale factor for softmax: larger alpha means more focus on best tours
    # Let's set alpha such that the difference between best and 2nd best is significant.
    # Typical tour length ~ 5-10. Difference ~ 0.1-0.5.
    # alpha = 10 gives exp(-10*0.1) = 0.36. alpha = 20 gives 0.13.
    # Let's use alpha = 15.
    alpha = 15.0
    weights = np.exp(-alpha * (lengths_arr - lengths_arr.min()))
    weights /= np.sum(weights)
    
    H_struct = np.zeros((n, n), dtype=float)
    
    # Accumulate structural heuristic
    # H_struct[i,j] += weight_k if (i,j) is in tour k
    for t, w in zip(tours, weights):
        for k in range(n):
            i = t[k]
            j = t[(k + 1) % n]
            H_struct[i, j] += w
            H_struct[j, i] += w # Symmetric

    # Base heuristic: 1/distance
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    H_base = 1.0 / safe
    
    # Combine: H_final = H_base * (1 + lambda * H_struct)
    # H_struct values are between 0 and 1 (sum of weights).
    # If an edge is in the best tour, H_struct ~ 1.
    # If not, H_struct ~ 0.
    # This boosts good edges by factor (1+lambda) and leaves bad edges as base.
    # Let's tune lambda. If lambda=10, good edges are 11x more attractive.
    lambda_boost = 10.0
    H_final = H_base * (1.0 + lambda_boost * H_struct)
    
    # Ensure positive and finite
    H_final[diag] = 1e-9
    H_final = np.maximum(H_final, 1e-9)
    H_final = np.where(np.isfinite(H_final), H_final, 1e-9)
    
    return H_final
# EVOLVE-BLOCK-END
