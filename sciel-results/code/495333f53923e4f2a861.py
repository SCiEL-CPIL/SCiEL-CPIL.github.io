"""Optimal ±1 polynomials with 24 coefficients minimizing sup norm."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s kill

    N = 24
    HALF = 12
    sqrtN = math.sqrt(N)

    # --- Precompute: all 2^12 sub-polynomial coefficients ---
    total = 1 << HALF  # 4096
    S = np.zeros((total, HALF), dtype=np.float64)
    for i in range(total):
        for k in range(HALF):
            S[i, k] = 1.0 if (i >> k) & 1 else -1.0

    # --- Dense grid for the DFT ---
    G = 8192
    angles = 2 * np.pi * np.arange(G) / G
    zs = np.exp(1j * angles)

    # Build V1[j,k] = z_j^k  (k=0..11), V2[j,k] = z_j^(12+k) = z_j^12 * z_j^k
    V1 = np.ones((G, HALF), dtype=np.complex128)
    for k in range(1, HALF):
        V1[:, k] = V1[:, k-1] * zs

    z12 = zs ** HALF
    V2 = np.ones((G, HALF), dtype=np.complex128)
    V2[:, 0] = z12
    for k in range(1, HALF):
        V2[:, k] = V2[:, k-1] * zs

    # M_A[j,i] = A_i(z_j); M_B[j,i] = B_i(z_j)
    # (S.T has shape (HALF, total))
    M_A = V1 @ S.T   # (G, 4096) complex
    M_B = V2 @ S.T   # (G, 4096) complex

    # --- Score function for a pair (i,j) ---
    def score_pair(i, j):
        val = M_A[:, i] + M_B[:, j]
        return np.max(np.abs(val)) / sqrtN

    def time_ok():
        return time.time() < deadline - 8.0

    # --- Phase 1: Random restarts with greedy bit-flip local search ---
    best_score = np.inf
    best_i = -1
    best_j = -1

    rng = np.random.RandomState(12345)

    while time_ok():
        # Start from random
        i = rng.randint(0, total)
        j = rng.randint(0, total)
        cur = score_pair(i, j)

        # Greedy improvement: try flipping each bit in i or j
        improved = True
        while improved and time_ok():
            improved = False
            # Try flipping each bit in i
            for k in range(HALF):
                ni = i ^ (1 << k)
                s = score_pair(ni, j)
                if s < cur:
                    cur = s
                    i = ni
                    improved = True
            # Try flipping each bit in j
            for k in range(HALF):
                nj = j ^ (1 << j if False else (1 << k))
                s = score_pair(i, nj)
                if s < cur:
                    cur = s
                    j = nj
                    improved = True

        if cur < best_score:
            best_score = cur
            best_i = i
            best_j = j

    # --- Phase 2: Systematic local search from best found ---
    # Do a more thorough exploration around the best
    cur_i, cur_j = best_i, best_j
    cur_score = best_score

    # Try all pairs within a small Hamming neighborhood
    # For each bit flip combination in i (up to 2 bits) and j (up to 2 bits)
    if time_ok():
        # Single-bit neighborhood
        for ki in range(HALF):
            ni = cur_i ^ (1 << ki)
            s = score_pair(ni, cur_j)
            if s < cur_score:
                cur_score = s
                cur_i = ni
        for kj in range(HALF):
            nj = cur_j ^ (1 << kj)
            s = score_pair(cur_i, nj)
            if s < cur_score:
                cur_score = s
                cur_j = nj

        # Second pass single-bit
        for ki in range(HALF):
            ni = cur_i ^ (1 << ki)
            s = score_pair(ni, cur_j)
            if s < cur_score:
                cur_score = s
                cur_i = ni
        for kj in range(HALF):
            nj = cur_j ^ (1 << kj)
            s = score_pair(cur_i, nj)
            if s < cur_score:
                cur_score = s
                cur_j = nj

    # --- Phase 3: If time remains, more random restarts targeting the best region ---
    while time_ok():
        # Perturb the best by flipping 1-3 random bits
        i = best_i
        j = best_j
        n_flips = rng.randint(1, 4)
        for _ in range(n_flips):
            if rng.random() < 0.5:
                i ^= (1 << rng.randint(0, HALF))
            else:
                j ^= (1 << rng.randint(0, HALF))

        cur = score_pair(i, j)

        # Greedy improvement
        improved = True
        while improved and time_ok():
            improved = False
            for k in range(HALF):
                ni = i ^ (1 << k)
                s = score_pair(ni, j)
                if s < cur:
                    cur = s
                    i = ni
                    improved = True
            for k in range(HALF):
                nj = j ^ (1 << k)
                s = score_pair(i, nj)
                if s < cur:
                    cur = s
                    j = nj
                    improved = True

        if cur < best_score:
            best_score = cur
            best_i = i
            best_j = j

    # --- Final: golden-section refinement to get accurate peak location ---
    def refine_peak(coeffs):
        """Find the exact maximum of |P(z)| using golden section on the dense grid."""
        # First find top peaks on the dense grid
        vals = np.array([
            sum(coeffs[k] * (np.exp(1j * angles[j]) ** k) for k in range(N))
            for j in range(G)
        ])
        abs_vals = np.abs(vals)

        # Get top 5 peak indices
        top_indices = np.argsort(abs_vals)[-5:][::-1]

        max_peak = 0.0
        gr = (math.sqrt(5) - 1) / 2  # golden ratio conjugate

        for idx in top_indices:
            theta_c = angles[idx]
            window = 2 * np.pi / G
            a = theta_c - window
            b = theta_c + window

            def f(theta):
                z = np.exp(1j * theta)
                # Horner's method
                val = coeffs[23]
                for k in range(22, -1, -1):
                    val = val * z + coeffs[k]
                return abs(val)

            # Golden section search
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = f(c)
            fd = f(d)

            for _ in range(60):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + gr * (b - a)
                    fd = f(d)

            max_peak = max(max_peak, fc, fd)

        return max_peak

    # Build final coefficients
    coeffs = np.zeros(N, dtype=np.float64)
    coeffs[:HALF] = S[best_i, :]
    coeffs[HALF:] = S[best_j, :]

    # Refine for accurate measurement (this is just for our own tracking;
    # the evaluator does its own refinement)
    if time.time() < deadline - 2.0:
        refined_peak = refine_peak(coeffs)
        # This confirms our grid-based score is accurate

    return coeffs.astype(int).tolist()

# EVOLVE-BLOCK-END
