import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s / 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # Isolated vertices: always safe to pick -> top priority
    BIG = 1e9

    # Initialize with inverse degree
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    # Fixed-point iteration: s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # This captures "safety" — vertices whose neighbors have low total score are safer
    A_float = A.astype(float)
    for it in range(500):
        if time.time() > deadline - 10.0:
            break
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        if np.allclose(new_s, s, rtol=1e-8, atol=1e-10):
            s = new_s
            break
        s = new_s

    # Spectral refinement: use low-frequency eigenvectors of the Laplacian
    # Vertices at the periphery (high magnitude in low-frequency modes) are
    # less central and safer to include in an independent set
    try:
        if time.time() < deadline - 8.0:
            D = np.diag(deg)
            L = D - A_float

            if n <= 500:
                eigvals, eigvecs = np.linalg.eigh(L)
                # Find the first positive eigenvalue index (skip zeros for disconnected graphs)
                positive_mask = eigvals > 1e-10
                if positive_mask.any():
                    start_idx = np.where(positive_mask)[0][0]
                    # Use up to 5 low-frequency eigenvectors for richer signal
                    num_modes = min(5, n - start_idx)
                    modes = eigvecs[:, start_idx:start_idx + num_modes]
                    # Weighted combination: earlier modes (lower frequency) get more weight
                    mode_weights = np.exp(-np.arange(num_modes) * 0.5)
                    mode_weights /= mode_weights.sum()
                    # Per-vertex spectral score: weighted sum of squared magnitudes
                    spectral_score = np.sum((modes ** 2) * mode_weights, axis=1)

                    # Normalize to [0, 1]
                    s_max = spectral_score.max()
                    if s_max > 1e-12:
                        spectral_score = spectral_score / s_max
                    else:
                        spectral_score = np.zeros(n)
                else:
                    spectral_score = np.zeros(n)

                # Smooth combination: use a soft weighting that lets the spectral
                # signal refine but not dominate the safety score
                # Normalize safety score to comparable range
                s_noniso = s[has_edge]
                if s_noniso.size > 0:
                    s_min = s_noniso.min()
                    s_max_s = s_noniso.max()
                    s_range = s_max_s - s_min
                    if s_range > 1e-12:
                        s_norm = (s - s_min) / s_range
                    else:
                        s_norm = np.zeros_like(s)
                else:
                    s_norm = np.zeros_like(s)

                # Combine: primary signal is safety, spectral is a refinement
                # Use a product-like interaction: safety * (1 + small * spectral)
                # This preserves the ordering of safety while boosting peripheral vertices
                alpha = 0.15
                scores = s * (1.0 + alpha * spectral_score)

                # Add a small secondary spectral term for tie-breaking among similar safety scores
                scores = scores + 0.02 * spectral_score * s_norm
            else:
                scores = s.copy()
        else:
            scores = s.copy()
    except Exception:
        scores = s.copy()

    # Isolated vertices must remain at top
    scores[~has_edge] = BIG

    # Small tie-breaker: lower degree slightly preferred (less competition)
    if deg.max() > 0:
        deg_rank = -deg / (deg.max() + 1)
        scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
