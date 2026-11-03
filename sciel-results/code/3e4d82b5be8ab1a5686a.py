"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)


def _benders_cut(a, b, p, cut_coeff, cut_const, n):
    """Generate a Benders cut from current assignment p.
    
    The linearized surrogate cost is sum_i sum_j a[i,j] * b[p[i],p[j]].
    We maintain a lower-bound surrogate: for each facility i assigned to location l,
    the marginal contribution is sum_j a[i,j] * b[l, p[j]].
    
    A Benders cut says: for any assignment q, 
    cost(q) >= sum_i c_i(q[i]) + const
    where c_i(l) = sum_j a[i,j] * b[l, p[j]] (the linearized cost of putting i at l,
    given current assignment of others).
    
    We update cut coefficients to be the pointwise minimum over all cuts generated so far.
    """
    # Linearized cost matrix: L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # This is a @ b.T[p]  ... let's compute efficiently
    # L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # = sum_j a[i,j] * b.T[p[j], l]
    # = (a @ b.T)[i, p[j]] summed over j... no
    # L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # Let's compute: for each i, L[i,:] = sum_j a[i,j] * b[:, p[j]]
    # = a[i,:] @ b[:, p[:]]  ... wait, that's sum_j a[i,j] * b[:,p[j]][l] which is what we want
    
    # L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # Compute as: L = a @ b[:, p]  -- no, dimensions don't work directly
    # a is (n,n), b[:, p] is (n,n) where column j of b[:,p] is b[:,p[j]]
    # (a @ b[:,p])[i,l] = sum_j a[i,j] * b[:,p][j,l] = sum_j a[i,j] * b[j, p[j]][l]... no
    # b[:,p][j,l] = b[j, p[l]]... that's not right either.
    # b[:, p] means for each row index r, the column is p. So b[:,p][r,l] = b[r, p[l]].
    # We want b[l, p[j]]. That's b.T[p[j], l] or equivalently, if we let M = b.T, then M[p[j], l].
    # L[i,l] = sum_j a[i,j] * b[l, p[j]]
    # Let's just compute it: for each l, L[:,l] = sum_j a[:,j] * b[l, p[j]]
    # b[l, p[j]] for all j is b[l, p] which is a vector of length n.
    # So L[:,l] = a @ b[l, p]  (matrix-vector product)
    # L = a @ b[:, p].T ? No.
    # Let's do: L[i,l] = sum_j a[i,j] * b[l, p[j]]
    # Define V[l,j] = b[l, p[j]]. Then L[i,l] = sum_j a[i,j] * V[l,j] = (a @ V.T)[i,l]
    # V = b[:, p] has shape (n,n) where V[l,j] = b[l, p[j]]. Yes!
    # So L = a @ b[:, p].T
    # But b[:, p].T has shape (n,n) where element [j,l] = b[l, p[j]]. 
    # (a @ b[:,p].T)[i,l] = sum_j a[i,j] * b[:,p].T[j,l] = sum_j a[i,j] * b[l,p[j]]. Correct!
    
    L = a @ b[:, p].T  # L[i,l] = sum_j a[i,j] * b[l, p[j]]
    
    # Update cut: cut_coeff[i, l] = min(cut_coeff[i, l], L[i, l])
    # cut_const stays the same for this formulation (each cut is independent)
    np.minimum(cut_coeff, L, out=cut_coeff)
    
    return L


def _surrogate_cost(cut_coeff, p, n):
    """Compute surrogate cost: sum_i cut_coeff[i, p[i]]"""
    return int(np.sum(cut_coeff[np.arange(n), p]))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110
    
    # Initial assignment: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Benders cut storage
    # cut_coeff[i, l] = min over all cuts of L_k[i, l]
    # Initialize with +inf, then generate first cut
    cut_coeff = np.full((n, n), np.inf, dtype=np.float64)
    
    # Generate initial cut
    _benders_cut(a, b, p, cut_coeff, 0, n)
    
    # Benders decomposition loop
    # Phase 1: Use surrogate to guide assignment, then polish with local search
    T = 5000.0
    T_min = 1.0
    cooling = 0.99999
    rng = np.random.default_rng(42)
    
    # We'll alternate between:
    # 1. SA on the exact QAP cost (with swaps)
    # 2. Periodically regenerate Benders cuts from good solutions
    # 3. Use surrogate to bias the search
    
    # Keep track of the surrogate-based "master" solution
    surrogate_best_p = best_p.copy()
    surrogate_best_cost = best_cost
    
    # Number of SA iterations since last cut regeneration
    iters_since_cut = 0
    cut_interval = 500  # regenerate cut every 500 iterations
    
    while time.time() < deadline:
        if T < T_min:
            T = 5000.0
            p = best_p.copy()
            current_cost = best_cost
            continue
        
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                surrogate_best_p = best_p.copy()
                surrogate_best_cost = best_cost
                iters_since_cut = 0  # reset cut counter on improvement
        
        iters_since_cut += 1
        T *= cooling
        
        # Periodically regenerate Benders cut from the best solution
        if iters_since_cut >= cut_interval:
            _benders_cut(a, b, best_p, cut_coeff, 0, n)
            iters_since_cut = 0
    
    # Final polish: local search from best solution
    # Steepest descent: find best improving swap
    p = best_p.copy()
    current_cost = best_cost
    n_pairs = n * (n - 1) // 2
    max_passes = 10
    pass_limit = (deadline - time.time()) / max_passes if max_passes > 0 else 0
    
    for _ in range(max_passes):
        if time.time() >= deadline - 2:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() >= deadline - 2:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_d
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
