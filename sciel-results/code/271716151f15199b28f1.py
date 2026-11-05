"""Multi-start local search for Maximum Cut with spectral initialization,
pairwise flips, ILS, and SA polish.

Improvements over baseline:
1. Fiedler vector (spectral) initialization for a high-quality start.
2. Pairwise flip moves in local search to escape single-flip local optima.
3. More diverse initializations including spectral-based starts.
4. Tighter time management to maximize search within the deadline.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigs


def _cut_value(s, A, E):
    """Compute cut value: s in {+1,-1}, cut = E/2 - 0.25 * s^T A s."""
    return 0.5 * E - 0.25 * float(s @ (A @ s))


def _fiedler_vector(A, n):
    """Compute Fiedler vector of the graph Laplacian via eigenvalue decomposition.
    
    Returns a sign vector s in {+1, -1}^n derived from the Fiedler vector.
    Falls back to random if eigendecomposition fails.
    """
    try:
        # Build sparse Laplacian
        A_sparse = csr_matrix(A)
        D = np.diag(A.sum(axis=1))
        L = D - A_sparse
        L = csr_matrix(L)
        
        # Find the second smallest eigenvalue (Fiedler value) and its eigenvector
        # eigs finds eigenvalues near sigma=0, so we look for the smallest non-zero
        try:
            # For a connected graph, the smallest eigenvalue is 0 with eigenvector all-ones
            # The Fiedler vector is the eigenvector of the second smallest eigenvalue
            vals, vecs = eigs(L, k=2, sigma=0.0, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals.real)
            fiedler_val = vals.real[idx[1]]
            fiedler_vec = vecs[:, idx[1]].real
        except Exception:
            # Fallback: use dense eigendecomposition for small graphs
            L_dense = L.toarray()
            vals, vecs = np.linalg.eigh(L_dense)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        
        # Convert to sign vector
        s = np.sign(fiedler_vec)
        s[s == 0] = 1.0  # Handle zeros
        return s
    except Exception:
        # Fallback to random
        return np.random.choice(np.array([-1.0, 1.0]), size=n)


def _local_search(s, A, E, deadline, rng=None, use_pairwise=True):
    """Steepest-ascent local search with incremental Ax update.
    
    Supports single-flip and pairwise-flip moves.
    """
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    Ax = A @ s
    
    if rng is None:
        rng = np.random.default_rng()
    
    while True:
        if time.time() > deadline:
            break
        
        # --- Single flip move ---
        gains = s * Ax
        i = int(np.argmax(gains))
        best_single_gain = gains[i]
        
        # --- Pairwise flip move (if enabled) ---
        best_pair_gain = -np.inf
        best_pair_i, best_pair_j = -1, -1
        
        if use_pairwise:
            # For pairwise flip of i and j:
            # New gain = 2 * (s[i]*A[i,j] + Ax[i]*s[j] + Ax[j]*s[i] - s[i]*s[j]*A[i,j])
            # Actually, let's derive it properly.
            # Current contribution to objective from edges involving i or j:
            # When we flip both i and j, the change in s^T A s is:
            # -4 * (s[i]*Ax[i] + s[j]*Ax[j] - 2*s[i]*s[j]*A[i,j])
            # So the gain (decrease in -s^T A s, i.e., increase in cut) is:
            # 4 * (s[i]*Ax[i] + s[j]*Ax[j] - 2*s[i]*s[j]*A[i,j]) / 4
            # = s[i]*Ax[i] + s[j]*Ax[j] - 2*s[i]*s[j]*A[i,j]
            # But we need to be careful: Ax already includes the contribution from i to j and vice versa.
            # Let me re-derive.
            # s^T A s = sum_{p,q} s[p] A[p,q] s[q]
            # Flip i and j: s[i] -> -s[i], s[j] -> -s[j]
            # Change = -2 * sum_q s[q] A[i,q] s[i] - 2 * sum_p s[p] A[p,j] s[j] + 4 * s[i] A[i,j] s[j]
            # Wait, this is getting complex. Let me use a simpler approach.
            # 
            # The gain from flipping i alone is g_i = s[i] * Ax[i]
            # The gain from flipping j alone is g_j = s[j] * Ax[j]
            # The gain from flipping both is:
            # g_{ij} = g_i + g_j - 2 * s[i] * s[j] * A[i,j]
            # because when we flip i, Ax[j] changes by -2*s[i]*A[i,j], and vice versa.
            # 
            # So for a pair (i,j), the combined gain is:
            # s[i]*Ax[i] + s[j]*Ax[j] - 2*s[i]*s[j]*A[i,j]
            # 
            # We want to find the best pair. A simple heuristic: check pairs involving
            # the top few single-flip candidates.
            
            # Get top K candidates for single flips
            K = min(10, n)
            top_idx = np.argsort(-gains)[:K]
            
            for ii in range(len(top_idx)):
                for jj in range(ii + 1, len(top_idx)):
                    i2 = top_idx[ii]
                    j2 = top_idx[jj]
                    pair_gain = gains[i2] + gains[j2] - 2.0 * s[i2] * s[j2] * A[i2, j2]
                    if pair_gain > best_pair_gain:
                        best_pair_gain = pair_gain
                        best_pair_i, best_pair_j = i2, j2
        
        # Decide which move to take
        if best_pair_gain > best_single_gain and best_pair_gain > 1e-9:
            # Take the pairwise flip
            i2, j2 = best_pair_i, best_pair_j
            old_s_i, old_s_j = s[i2], s[j2]
            s[i2] = -s[i2]
            s[j2] = -s[j2]
            # Update Ax: for each vertex k, Ax[k] changes by -2*old_s_i*A[k,i2] - 2*old_s_j*A[k,j2]
            Ax -= 2.0 * old_s_i * A[:, i2]
            Ax -= 2.0 * old_s_j * A[:, j2]
        elif best_single_gain > 1e-9:
            # Take the single flip
            old_s_i = s[i]
            s[i] = -s[i]
            Ax -= 2.0 * old_s_i * A[:, i]
        else:
            # No improving move
            break
    
    return s, _cut_value(s, A, E)


def _perturb(s, rng, k_min=5, k_max=15):
    """Randomly flip k vertices."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    k = int(rng.integers(k_min, k_max + 1))
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _perturb_adaptive(s, rng, no_improve, n):
    """Adaptive perturbation: larger when stuck longer."""
    s = s.astype(np.float64).copy()
    base_k = 5 + min(no_improve, 20)
    k = int(rng.integers(max(3, base_k - 3), base_k + 10))
    k = min(k, n // 2)
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _simulated_annealing(s, A, E, deadline, rng, t_start=3.0, t_end=0.05, cooling=0.998):
    """Simulated annealing with incremental Ax update."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    Ax = A @ s
    best_s = s.copy()
    best_val = _cut_value(s, A, E)
    cur = best_val
    T = t_start

    while T > t_end:
        if time.time() > deadline:
            break
        i = int(rng.integers(0, n))
        gain = s[i] * Ax[i]

        if gain > 0 or (gain < 0 and rng.random() < np.exp(gain / T)):
            old_s_i = s[i]
            s[i] = -s[i]
            Ax -= 2.0 * old_s_i * A[:, i]
            cur += gain
            if cur > best_val:
                best_val = cur
                best_s = s.copy()

        T *= cooling

    return best_s, best_val


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    E = 0.5 * float(A.sum())

    start = time.time()
    deadline = start + 110.0  # Safety margin under 150 s limit
    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = []

    # 1. Spectral (Fiedler vector) initialization
    fiedler_s = _fiedler_vector(A, n)
    inits.append(fiedler_s)
    inits.append(-fiedler_s)  # Complement

    # 2. Structured starts
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1.0
    inits.append(alt)
    inits.append(np.ones(n, dtype=np.float64))
    inits.append(-np.ones(n, dtype=np.float64))

    # 3. Random starts
    for _ in range(25):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=n))

    best_s = None
    best_val = -1.0

    # Phase 1: Local search from all starts (with pairwise flips)
    for s0 in inits:
        if time.time() > deadline - 30.0:
            break
        s, val = _local_search(s0, A, E, deadline - 30.0, rng=rng, use_pairwise=True)
        if val > best_val:
            best_val = val
            best_s = s.copy()

    # Phase 2: ILS on the best solution with adaptive perturbation
    if best_s is not None and time.time() < deadline - 25.0:
        current_s = best_s.copy()
        current_val = best_val
        no_improve_count = 0
        max_iter = 2000
        for _ in range(max_iter):
            if time.time() > deadline - 25.0:
                break
            if no_improve_count > 5:
                pert_s = _perturb_adaptive(current_s, rng, no_improve_count, n)
            else:
                pert_s = _perturb(current_s, rng)
            ls_s, ls_val = _local_search(pert_s, A, E, deadline - 25.0, rng=rng, use_pairwise=True)
            if ls_val > current_val:
                current_s = ls_s
                current_val = ls_val
                no_improve_count = 0
                if current_val > best_val:
                    best_val = current_val
                    best_s = current_s.copy()
            else:
                no_improve_count += 1

    # Phase 3: Simulated annealing polish
    if best_s is not None and time.time() < deadline - 10.0:
        sa_deadline = deadline - 10.0
        for _ in range(3):
            if time.time() > sa_deadline:
                break
            s, val = _simulated_annealing(
                best_s, A, E, sa_deadline, rng,
                t_start=3.0, t_end=0.05, cooling=0.998
            )
            s, val = _local_search(s, A, E, sa_deadline, rng=rng, use_pairwise=True)
            if val > best_val:
                best_val = val
                best_s = s.copy()

    # Final high-precision local search polish
    if best_s is not None and time.time() < deadline:
        s, val = _local_search(best_s, A, E, deadline, rng=rng, use_pairwise=True)
        if val > best_val:
            best_val = val
            best_s = s

    return (best_s > 0).astype(int)
