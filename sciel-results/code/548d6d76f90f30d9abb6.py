"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute difference matrices for fast delta computation
    # b_diff[u, v, k] = b[v, k] - b[u, k]  shape (n, n, n)
    b_diff = b[None, :, :] - b[:, None, :]  # (n, n, n)
    # b_col_diff[u, v, k] = b[k, v] - b[k, u]  shape (n, n, n)
    # b[k, v] - b[k, u]: for fixed u,v, this is b[:, v] - b[:, u]
    # b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # b[:, v] has shape (n,), so b_col_diff[:, :, v] = b[:, v][None, None, :] - b[:, u][None, None, :]
    # Actually: b_col_diff = b[None, :, :] - b[:, None, :] gives b_diff[u,v,k] = b[v,k]-b[u,k]
    # We need b_col_diff[u,v,k] = b[k,v] - b[k,u]
    # That's the transpose in the last two axes: b_col_diff = b_diff.transpose(0, 2, 1)? No.
    # b_diff[u, v, k] = b[v, k] - b[u, k]
    # b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # Note b[k, v] = b[v, k] only if b is symmetric, which it may not be.
    # So b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # = (b.T[v, k] - b.T[u, k]) ... no, b[k,v] is just b[k,v]
    # Let's precompute: b_col_diff[u, v, :] = b[:, v] - b[:, u]
    # b[:, v] is column v of b, shape (n,)
    # So b_col_diff = np.empty((n, n, n))
    # for u in range(n): for v in range(n): b_col_diff[u,v,:] = b[:,v] - b[:,u]
    # Vectorized: b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # = b.T[v, k] - b.T[u, k] ... no. b[k,v] is element (k,v) of b.
    # b.T has shape (n,n), b.T[v,k] = b[k,v]. So b_col_diff[u,v,k] = b.T[v,k] - b.T[u,k]
    # = b_diff.T... let me just compute it directly.
    # b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # For all k: b[:, v] - b[:, u]
    # b_col_diff = np.stack([np.stack([b[:, v] - b[:, u] for v in range(n)]) for u in range(n)])
    # Vectorized: b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # = b[None, :, :] ... hmm
    # b[:, :, v] would be b[k, v] for all k, but that's shape (n, n) with second index being v... 
    # Actually b has shape (n, n). b[:, v] is column v.
    # b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # Let's define: for each pair (u,v), the vector d_{u,v}[k] = b[k,v] - b[k,u]
    # b_col_diff = np.empty((n, n, n), dtype=a.dtype)
    # for u in range(n):
    #     for v in range(n):
    #         b_col_diff[u, v, :] = b[:, v] - b[:, u]
    # Vectorized: b_col_diff[u, v, k] = b[k, v] - b[k, u]
    # = b.T[v, k] - b.T[u, k]  ... wait b.T[v, k] = b[k, v]. Yes!
    # So b_col_diff[u, v, k] = b.T[v, k] - b.T[u, k]
    # = b_diff_T where b_diff_T[v, u, k] = b.T[v, k] - b.T[u, k] = b[k, v] - b[k, u]
    # So b_col_diff = b_diff with indices swapped: b_col_diff[u, v, k] = b_diff[v, u, k]? 
    # b_diff[v, u, k] = b[u, k] - b[v, k]. That's not what we want.
    # We want b[k, v] - b[k, u] = b.T[v, k] - b.T[u, k]
    # Let bT_diff[v, u, k] = b.T[v, k] - b.T[u, k] = b[k, v] - b[k, u]
    # bT_diff = b.T[None, :, :] - b.T[:, None, :]  # (n, n, n): bT_diff[v, u, k] = b.T[v,k]-b.T[u,k]
    # So b_col_diff[u, v, k] = bT_diff[v, u, k]
    # b_col_diff = bT_diff.transpose(1, 0, 2)  # b_col_diff[u, v, k] = bT_diff[v, u, k]
    bT_diff = b.T[None, :, :] - b.T[:, None, :]  # (n, n, n): bT_diff[v, u, k] = b[k, v] - b[k, u]
    b_col_diff = bT_diff.transpose(1, 0, 2)  # (n, n, n): b_col_diff[u, v, k] = b[k, v] - b[k, u]

    # a_row_diff[i, j, k] = a[i, k] - a[j, k]
    a_row_diff = a[None, :, :] - a[:, None, :]  # (n, n, n)
    # a_col_diff[i, j, k] = a[k, i] - a[k, j]
    aT_diff = a.T[None, :, :] - a.T[:, None, :]  # (n, n, n): aT_diff[i, j, k] = a[k, i] - a[k, j]
    a_col_diff = aT_diff  # a_col_diff[i, j, k] = a[k, i] - a[k, j]

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Initial solution: greedy rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(p)

    # Precompute all-pairs index arrays excluding i and j
    # For fast delta computation
    pair_idx = np.empty((n, n, n - 2), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            idx = np.concatenate([np.arange(i), np.arange(i + 1, j), np.arange(j + 1, n)])
            pair_idx[i, j, :] = idx
            pair_idx[j, i, :] = idx

    def compute_all_deltas(p):
        """Compute delta for all (i,j) swap pairs. Returns (n, n) array where delta[i,j] is the change."""
        deltas = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                idx = pair_idx[i, j, :]
                pm = p[idx]

                # term1: sum_k (a[i,k]-a[j,k]) * (b[v,p[k]]-b[u,p[k]])
                # = sum_k a_row_diff[i,j,k] * b_diff[u,v,p[k]]
                term1 = np.dot(a_row_diff[i, j, idx], b_diff[u, v, pm])

                # term2: sum_k (a[k,i]-a[k,j]) * (b[p[k],v]-b[p[k],u])
                # = sum_k a_col_diff[i,j,k] * b_col_diff[u,v,p[k]]
                term2 = np.dot(a_col_diff[i, j, idx], b_col_diff[u, v, pm])

                # term3: (a[i,j]-a[j,i]) * (b[v,u]-b[u,v])
                term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

                d = term1 + term2 + term3
                deltas[i, j] = d
                deltas[j, i] = d
        return deltas

    def local_search(p, max_time=None):
        """Exhaustive 2-opt local search until convergence."""
        if max_time is None:
            max_time = deadline
        while time.time() < max_time:
            deltas = compute_all_deltas(p)
            # Find best improving swap
            i_best, j_best = np.unravel_index(np.argmin(deltas), deltas.shape)
            if i_best == j_best or deltas[i_best, j_best] >= 0:
                break
            p[i_best], p[j_best] = p[j_best], p[i_best]
        return p

    # Simulated Annealing
    np.random.seed(42)

    t0 = 8000.0
    t_final = 0.05
    # We'll do multiple SA runs with different initial temperatures and perturbation levels

    num_sa_runs = 20
    for sa_run in range(num_sa_runs):
        if time.time() > deadline - 10:
            break

        if sa_run == 0:
            p = best_p.copy()
            # Scale temperature based on cost magnitude
            cost_scale = max(abs(best_cost) / 1000.0, 1.0)
            t = t0 * cost_scale
        else:
            p = best_p.copy()
            # Perturb: random swaps
            num_swaps = min(2 + sa_run // 5, 12)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            cur_cost = compute_cost(p)
            t = t0 * cost_scale * np.exp(-0.3 * sa_run)

        # SA loop
        no_improve_count = 0
        sa_best_p = p.copy()
        sa_best_cost = compute_cost(p)

        while time.time() < deadline - 5:
            # Compute all deltas
            deltas = compute_all_deltas(p)
            # Choose a random swap (or best)
            # For SA, pick random swap to explore
            i, j = np.random.choice(n, 2, replace=False)
            if i == j:
                continue
            d = deltas[i, j]

            if d < 0 or np.random.random() < np.exp(-d / max(t, 1e-10)):
                p[i], p[j] = p[j_best if False else j], p[i]
                cur_cost = compute_cost(p)
                if cur_cost < sa_best_cost:
                    sa_best_cost = cur_cost
                    sa_best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Cool down
                t *= 0.995
                if t < t_final:
                    t = t_final
            else:
                no_improve_count += 1

            # Early stop if stuck for too long
            if no_improve_count > 500:
                break

        # Final local search polish
        p = local_search(sa_best_p)
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    # Final exhaustive local search
    if time.time() < deadline:
        p = local_search(best_p)
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
