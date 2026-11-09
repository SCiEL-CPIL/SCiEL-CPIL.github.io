import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps_list.append((i0, i1, i2, i3))

    aps_arr = np.array(aps_list, dtype=np.int32)
    M = len(aps_arr)

    # Helper to count monochromatic APs
    def count_mono(c):
        return np.sum((c[aps_arr[:, 0]] == c[aps_arr[:, 1]]) &
                      (c[aps_arr[:, 1]] == c[aps_arr[:, 2]]) &
                      (c[aps_arr[:, 2]] == c[aps_arr[:, 3]]))

    def thue_morse(i):
        return bin(i).count('1') % 2

    # Generate diverse initial colorings
    initial_colors_list = []

    # Thue-Morse
    initial_colors_list.append(np.array([thue_morse(i) for i in range(n)], dtype=np.int8))

    # Randoms
    for _ in range(5):
        initial_colors_list.append(np.random.randint(0, 2, n).astype(np.int8))

    # Structured
    initial_colors_list.append(np.array([i % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 8) % 2 for i in range(n)], dtype=np.int8))

    best_score_global = float('inf')
    best_colors_global = None

    # Precompute incidence matrix A (M x n) where A[j, i] = 1 if position i is in AP j
    # This is used for spectral projection
    A = np.zeros((M, n), dtype=np.float32)
    for k in range(4):
        A[np.arange(M), aps_arr[:, k]] = 1.0

    # Compute SVD of A to get top singular vectors
    # We want to project onto the subspace orthogonal to the top singular vectors
    # This suppresses the modes that most strongly correlate with monochromatic APs
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Vt is (min(M,n) x n). We take the top K singular vectors
        K = 16  # Number of top singular vectors to project out
        top_V = Vt[:K, :]  # Shape (K, n)
        # Projection matrix onto orthogonal complement: P = I - V_top @ V_top.T
        # But we apply it to the vector, so: v_projected = v - V_top.T @ (V_top @ v)
        # Since V_top rows are orthonormal, V_top @ V_top.T = I_K
        # So: v_projected = v - V_top.T @ (V_top @ v)
    except Exception:
        top_V = None

    def spectral_project(v):
        """Project vector v onto subspace orthogonal to top singular vectors."""
        if top_V is None:
            return v
        # v is shape (n,), top_V is shape (K, n)
        # V_top @ v gives shape (K,)
        coeffs = top_V @ v  # (K,)
        # V_top.T @ coeffs gives shape (n,)
        projection = top_V.T @ coeffs  # (n,)
        return v - projection

    def run_spectral(initial_colors, seed_idx):
        nonlocal best_score_global, best_colors_global
        np.random.seed(200 + seed_idx)
        colors = initial_colors.copy().astype(np.float32)
        # Convert to {-1, 1} for spectral methods (more natural for linear algebra)
        # 0 -> -1, 1 -> +1
        colors_pm = 2 * colors - 1  # Now in {-1, 1}
        current_score = count_mono((colors_pm + 1) // 2)

        best_score_local = current_score
        best_colors_local = ((colors_pm + 1) // 2).copy()

        # Iterative spectral projection with thresholding
        for iteration in range(2000):
            if time.time() > deadline:
                break

            # Apply spectral projection
            colors_projected = spectral_project(colors_pm)

            # Threshold back to {-1, 1}
            colors_pm = np.sign(colors_projected)
            # Handle zeros (shouldn't happen often, but just in case)
            colors_pm[colors_pm == 0] = 1.0

            # Convert back to {0, 1} for scoring
            colors_01 = ((colors_pm + 1) // 2).astype(np.int8)
            new_score = count_mono(colors_01)

            if new_score < best_score_local:
                best_score_local = new_score
                best_colors_local = colors_01.copy()

            # Add small random perturbation to escape local minima
            if iteration % 10 == 0:
                num_perturb = np.random.randint(1, 5)
                perturb_positions = np.random.choice(n, num_perturb, replace=False)
                colors_pm[perturb_positions] *= -1  # Flip sign

            # Check if converged
            if new_score == 0:
                break

        # Update global best
        if best_score_local < best_score_global:
            best_score_global = best_score_local
            best_colors_global = best_colors_local.copy()

    # Run spectral method in parallel
    import threading
    lock = threading.Lock()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i, init_c in enumerate(initial_colors_list):
            if time.time() > deadline - 5:
                break
            futures.append(executor.submit(run_spectral, init_c, i))
        for future in futures:
            try:
                future.result()
            except Exception as e:
                pass

    if best_colors_global is None:
        return [int(c) for c in initial_colors_list[0]]

    return [int(c) for c in best_colors_global]
