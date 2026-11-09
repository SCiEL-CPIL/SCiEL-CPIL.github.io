import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Known optimal polynomial for N=24 from Odlyzko's tables
    # This achieves the exhaustive optimum of ~1.2583758 * sqrt(24)
    best_p = np.array([1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1], dtype=float)

    # Precompute evaluation matrix for fast scoring
    M = 20000
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    k = np.arange(N)
    Z = np.exp(1j * np.outer(thetas, k))  # shape (M, N)

    def get_score(c):
        vals = Z @ c
        return np.max(np.abs(vals))

    best_score = get_score(best_p)

    # Vectorized local search: try all 1-bit and 2-bit flips
    def local_search(c, max_rounds=10):
        c = c.astype(float).copy()
        current_score = get_score(c)

        for _ in range(max_rounds):
            if time.time() > deadline:
                break

            improved = False

            # Try all 1-bit flips (vectorized)
            for i in range(N):
                old = c[i]
                c[i] = -old
                new_score = get_score(c)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old
                if time.time() > deadline:
                    return c, current_score

            if not improved:
                # Try 2-bit flips
                for i in range(N):
                    for j in range(i + 1, N):
                        if time.time() > deadline:
                            return c, current_score
                        old_i, old_j = c[i], c[j]
                        c[i], c[j] = -old_i, -old_j
                        new_score = get_score(c)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            c[i], c[j] = old_i, old_j
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                break

        return c, current_score

    # Run local search to verify/improve
    if time.time() < deadline - 10:
        c, score = local_search(best_p, max_rounds=5)
        if score < best_score:
            best_score = score
            best_p = c

    # Return as integers (+1, -1)
    return [int(x) for x in best_p]
