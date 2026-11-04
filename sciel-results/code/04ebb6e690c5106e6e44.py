import numpy as np
import time
import math

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute exact cost."""
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
        nn = D.shape[0]
        J = np.eye(nn) - np.ones((nn, nn)) / nn
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        """Align X to Y and solve assignment problem."""
        from scipy.optimize import linear_sum_assignment

        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q

        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross

        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def wl_color_refinement(A, B, iterations=10):
        """
        Weisfeiler-Lehman style color refinement heuristic.
        Iteratively refines node labels based on neighborhood structures
        to derive a high-fidelity initial permutation.
        """
        # Initial colors: based on row sums (degree/weight)
        colors_A = A.sum(axis=1)
        colors_B = B.sum(axis=1)

        # Discretize colors to bins
        def discretize(colors, num_bins=50):
            unique_vals, indices = np.unique(colors, return_inverse=True)
            if len(unique_vals) > num_bins:
                # Use quantiles to bin
                bins = np.quantile(unique_vals, np.linspace(0, 1, num_bins + 1))
                indices = np.searchsorted(bins, colors, side='right') - 1
                indices = np.clip(indices, 0, num_bins - 1)
            return indices

        col_A = discretize(colors_A)
        col_B = discretize(colors_B)

        for _ in range(iterations):
            # Update colors based on neighbor colors
            new_col_A = np.zeros(n, dtype=np.int64)
            new_col_B = np.zeros(n, dtype=np.int64)

            for i in range(n):
                # For A, neighbor colors are col_A[j] for all j, weighted by A[i,j]
                # Simplified: use histogram of neighbor colors
                hist_A = np.bincount(col_A, weights=A[i, :], minlength=50)
                hist_B = np.bincount(col_B, weights=B[i, :], minlength=50)
                
                # Combine with current color
                key_A = tuple(hist_A) + (col_A[i],)
                key_B = tuple(hist_B) + (col_B[i],)
                
                # Hash to new color index (simplified)
                new_col_A[i] = hash(key_A) % 1000
                new_col_B[i] = hash(key_B) % 1000

            # Re-discretize
            col_A = discretize(new_col_A.astype(float))
            col_B = discretize(new_col_B.astype(float))

        # Match colors: for each color in A, find best matching color in B
        p = np.empty(n, dtype=np.int64)
        matched_B = set()
        
        # Sort by color frequency
        for i in range(n):
            best_j = -1
            best_score = -np.inf
            for j in range(n):
                if j in matched_B:
                    continue
                # Score based on color similarity
                score = 1.0 / (1.0 + abs(col_A[i] - col_B[j]))
                if score > best_score:
                    best_score = score
                    best_j = j
            
            if best_j == -1:
                # Fallback: assign unmatched
                for j in range(n):
                    if j not in matched_B:
                        best_j = j
                        break
            
            p[i] = best_j
            matched_B.add(best_j)

        return p

    def make_B_perm(p):
        """Build permuted B matrix: B_perm[i,j] = b_f[p[i], p[j]]."""
        return b_f[np.ix_(p, p)].copy()

    def swap_in_B_perm(B_perm, i, j):
        """Swap rows i,j and columns i,j of B_perm in place."""
        B_perm[[i, j], :] = B_perm[[j, i], :]
        B_perm[:, [i, j]] = B_perm[:, [j, i]]

    def efficient_local_search(p, current_cost, end_time, seed=12345, tol=-1e-10):
        """
        First-improvement 2-swap local search using B_perm for O(1) row/col access.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        B_perm = make_B_perm(curr_p)

        rng = np.random.RandomState(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)

            for i, j in pairs:
                if time.time() > end_time:
                    break

                drow = B_perm[j, :] - B_perm[i, :]
                dcol = B_perm[:, j] - B_perm[:, i]

                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]

                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

                if delta < tol:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    swap_in_B_perm(B_perm, i, j)

                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True

            if not improved:
                break

        return best_p, best_cost

    def sa_polish(curr_p, curr_cost, best_p, best_cost, B_perm, end_time, seed=111):
        """SA polish reusing existing B_perm. Slower cooling for longer effective SA."""
        best_p_sa = best_p.copy()
        best_cost_sa = best_cost
        rng_sa = np.random.RandomState(seed)
        T = 500.0
        cooling = 0.9995  # Slower cooling: ~1400 steps to reach T=1, allowing more exploration
        while time.time() < end_time and T > 0.5:
            i, j = rng_sa.choice(n, 2, replace=False)

            drow = B_perm[j, :] - B_perm[i, :]
            dcol = B_perm[:, j] - B_perm[:, i]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                swap_in_B_perm(B_perm, i, j)
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            T *= cooling
        return curr_p, curr_cost, best_p_sa, best_cost_sa

    # --- Phase 1: Multiple Initializations ---
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0

    candidates = []

    # 1. MDS Initialization (multi-dimension)
    for dim in [2, 3, 5]:
        if time.time() > deadline - 60.0:
            break
        try:
            X = mds_embedding(a_sym, dim)
            Y = mds_embedding(b_sym, dim)
            p_try = procrustes_alignment(X, Y)
            cost_try = compute_cost(p_try)
            candidates.append((p_try, cost_try))
        except Exception:
            pass

    # 2. Rank Matching Initialization
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    candidates.append((p_rank, cost_rank))

    # 3. WL Color Refinement Initialization
    if time.time() < deadline - 50.0:
        try:
            p_wl = wl_color_refinement(a_f, b_f, iterations=5)
            cost_wl = compute_cost(p_wl)
            candidates.append((p_wl, cost_wl))
        except Exception:
            pass

    # 4. Spectral Initialization (Hungarian on A@B.T)
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import linear_sum_assignment
            M = a_f @ b_f.T
            row_ind, col_ind = linear_sum_assignment(-M)
            p_spec = np.empty(n, dtype=np.int64)
            p_spec[row_ind] = col_ind
            cost_spec = compute_cost(p_spec)
            candidates.append((p_spec, cost_spec))
        except Exception:
            pass

    # 5. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    candidates.append((p_rand, cost_rand))

    # Select best candidate
    best_p_init, best_cost_init = min(candidates, key=lambda x: x[1])

    # --- Phase 2: Main Local Search Polish ---
    end_time_lk = deadline - 30.0
    best_p, best_cost = efficient_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)

    # --- Phase 3: Build B_perm once and reuse for SA ---
    B_perm_main = make_B_perm(best_p)

    # --- Phase 4: SA Polish with longer effective runtime ---
    if time.time() < deadline - 15.0:
        curr_p = best_p.copy()
        curr_cost = best_cost
        end_time_sa = deadline - 10.0
        curr_p, curr_cost, best_p_sa, best_cost_sa = sa_polish(
            curr_p, curr_cost, best_p, best_cost, B_perm_main, end_time_sa, seed=111
        )
        if best_cost_sa < best_cost:
            best_p = best_p_sa
            best_cost = best_cost_sa
            B_perm_main = make_B_perm(best_p)

    # --- Phase 5: Second SA restart with different seed if time remains ---
    if time.time() < deadline - 12.0:
        curr_p2 = best_p.copy()
        curr_cost2 = best_cost
        # Small perturbation before second SA
        rng_perturb = np.random.RandomState(789)
        for _ in range(3):
            i, j = rng_perturb.choice(n, 2, replace=False)
            curr_p2[i], curr_p2[j] = curr_p2[j], curr_p2[i]
            swap_in_B_perm(B_perm_main, i, j)
        # Recompute cost after perturbation
        curr_cost2 = compute_cost(curr_p2)
        end_time_sa2 = deadline - 5.0
        curr_p2, curr_cost2, best_p_sa2, best_cost_sa2 = sa_polish(
            curr_p2, curr_cost2, best_p, best_cost, B_perm_main, end_time_sa2, seed=456
        )
        if best_cost_sa2 < best_cost:
            best_p = best_p_sa2
            best_cost = best_cost_sa2

    # --- Phase 6: Final high-precision polish pass ---
    if time.time() < deadline - 2.0:
        end_time_final = deadline - 0.5
        p_final, cost_final = efficient_local_search(best_p, best_cost, end_time_final, seed=999, tol=-1e-12)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
