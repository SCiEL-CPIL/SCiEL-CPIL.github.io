"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Spectral Initialization
    # We want to align high-flow facilities with high-distance locations.
    # The QAP cost is sum a[i,j] b[p[i], p[j]].
    # If we decompose a and b into eigenbases, we can try to match principal components.
    # For symmetric positive semi-definite matrices, the largest eigenvectors correspond to the "directions" of highest activity.
    # Let a = V_a Lambda_a V_a^T and b = V_b Lambda_b V_b^T.
    # The optimal assignment often correlates the principal eigenvectors.
    # Specifically, if we sort facilities by their projection onto the largest eigenvector of A,
    # and locations by their projection onto the largest eigenvector of B, and match them,
    # we get a geometrically informed initial solution.
    
    try:
        # Compute eigenvalues and eigenvectors
        # We only need the top few (or just the top one) for a strong heuristic signal.
        # eigh returns eigenvalues in ascending order.
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        
        # Take the eigenvector corresponding to the largest eigenvalue
        # (last index in ascending order)
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        # Sort facilities by their score on v_a (descending)
        # Sort locations by their score on v_b (descending)
        # This aligns the "most active" facilities with the "most connected" locations.
        facility_scores = v_a
        location_scores = v_b
        
        heavy_facilities = np.argsort(-facility_scores, kind="stable")
        central_locations = np.argsort(-location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        
    except Exception:
        # Fallback to rank matching if spectral fails
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Helper to compute delta for swapping facilities i and j
    # p[i] = u, p[j] = v
    # Delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Note: We assume a and b are symmetric. If not, the formula is more complex.
    # QAPLIB instances typically have symmetric matrices.
    def get_delta(i, j, p, b_perm):
        """Compute delta for swapping facilities i and j using precomputed b_perm."""
        u = p[i]
        v = p[j]
        
        # b_perm[k, l] = b[p[k], p[l]]
        # We need sum_k (a[i,k] - a[j,k]) * (b_perm[k, v] - b_perm[k, u])
        # where b_perm[k, v] = b[p[k], v] and b_perm[k, u] = b[p[k], u]
        
        ai = a[i]
        aj = a[j]
        diff_a = ai - aj
        
        # b_perm[:, v] gives b[p[k], v] for all k
        # b_perm[:, u] gives b[p[k], u] for all k
        b_diff = b_perm[:, v] - b_perm[:, u]
        
        # Full sum over all k
        sum_all = np.dot(diff_a, b_diff)
        
        # Subtract k=i and k=j terms
        # k=i: diff_a[i] * (b_perm[i, v] - b_perm[i, u])
        term_i = diff_a[i] * (b_perm[i, v] - b_perm[i, u])
        
        # k=j: diff_a[j] * (b_perm[j, v] - b_perm[j, u])
        term_j = diff_a[j] * (b_perm[j, v] - b_perm[j, u])
        
        delta = 2 * (sum_all - term_i - term_j)
        return int(delta)

    def swap_in_b_perm(p, i, j, b_perm):
        """Update b_perm after swapping p[i] and p[j]."""
        u = p[i]
        v = p[j]
        p[i], p[j] = v, u
        # Swap rows and columns i and j in b_perm
        b_perm[[i, j], :] = b_perm[[j, i], :]
        b_perm[:, [i, j]] = b_perm[:, [j, i]]
        return p, b_perm

    def local_search(p, current_cost, b_perm, deadline):
        improved = True
        max_iters = 500
        iter_count = 0
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            # First-improvement: stop after finding first improving swap per pass
            for i in range(n):
                if time.time() > deadline:
                    return p, current_cost, b_perm
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p, b_perm)
                    if delta < 0:
                        p, b_perm = swap_in_b_perm(p, i, j, b_perm)
                        current_cost += delta
                        improved = True
                        break  # First improvement: restart scan
        return p, current_cost, b_perm

    start_time = time.time()
    timeout = 115.0  # Safety margin
    deadline = start_time + timeout
    
    # Precompute b_perm for fast delta computation
    b_perm = b[np.ix_(p, p)].copy()
    
    # ILS Loop with adaptive perturbation
    stagnation = 0
    while time.time() < deadline:
        # Local Search
        p, current_cost, b_perm = local_search(p, current_cost, b_perm, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            stagnation = 0
        else:
            stagnation += 1
        
        # Check time
        if time.time() > deadline:
            break
            
        # Adaptive Perturbation
        # Scale perturbation size with stagnation to escape deeper local optima
        base_swaps = max(2, n // 10)
        num_swaps = base_swaps + min(stagnation * 2, n // 5)
        
        for _ in range(num_swaps):
            if time.time() > deadline:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(i, j, p, b_perm)
            p, b_perm = swap_in_b_perm(p, i, j, b_perm)
            current_cost += delta
            
        # Occasional restart with fresh spectral init if very stuck
        if stagnation > 15 and time.time() < deadline - 5.0:
            try:
                eigvals_a, eigvecs_a = eigh(a)
                eigvals_b, eigvecs_b = eigh(b)
                v_a = eigvecs_a[:, -1]
                v_b = eigvecs_b[:, -1]
                heavy_facilities = np.argsort(-v_a, kind="stable")
                central_locations = np.argsort(-v_b, kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy_facilities] = central_locations
                b_perm = b[np.ix_(p, p)].copy()
                current_cost = int(np.sum(a * b_perm))
                stagnation = 0
            except Exception:
                pass

    return best_p.tolist()
# EVOLVE-BLOCK-END
