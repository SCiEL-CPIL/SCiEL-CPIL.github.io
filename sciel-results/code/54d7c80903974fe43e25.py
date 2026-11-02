"""Matrix multiplication tensor decomposition for 3x3x7."""
import time
import itertools
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE_MARGIN = 10.0  # seconds before the 150s kill


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] for M_{n,m,p}.
    Indexing: U rows = i*m+j, V rows = j*p+k, W rows = k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _rank_of_tensor(T):
    """Estimate the tensor rank via flattening to a matrix and computing rank.
    This gives an upper bound on the rank (the matricization rank)."""
    # Matricize along the first mode
    T1 = T.reshape(T.shape[0], -1)
    return np.linalg.matrix_rank(T1, tol=1e-10)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = len(U[0])
    # Reconstruct: for each column c, outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    reconstructed = np.zeros_like(T, dtype=np.int64)
    for c in range(rank):
        uc = np.array(U[:, c], dtype=np.int64)
        vc = np.array(V[:, c], dtype=np.int64)
        wc = np.array(W[:, c], dtype=np.int64)
        # Outer product
        reconstructed += np.einsum('i,j,k->ijk', uc, vc, wc)
    return np.array_equal(reconstructed, T)


def _verify_decomposition_fuzzy(U, V, W, n, m, p, tol=1e-6):
    """Verify with floating point tolerance."""
    T = _build_tensor(n, m, p)
    rank = len(U[0])
    reconstructed = np.zeros_like(T, dtype=np.float64)
    for c in range(rank):
        uc = np.array(U[:, c], dtype=np.float64)
        vc = np.array(V[:, c], dtype=np.float64)
        wc = np.array(W[:, c], dtype=np.float64)
        reconstructed += np.einsum('i,j,k->ijk', uc, vc, wc)
    return np.allclose(reconstructed, T, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank_via_linear_solve(n, m, p, target_rank, max_trials=50, seed=None):
    """Try to find a decomposition of given rank using random initialization and linear solve.
    
    The idea: we need to find U (nm x r), V (mp x r), W (pn x r) such that
    sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    
    This is a bilinear/trilinear problem. One approach: fix two matrices and solve for the third.
    But that's not quite right since it's trilinear.
    
    Alternative: use the "cut" method. Fix W to be random (or structured), then the problem becomes
    bilinear in U and V. We can iterate.
    
    Actually, a cleaner approach for small problems: 
    - Randomly initialize U, V, W
    - Iterate: fix two, solve for the third via least squares on the matricization
    """
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.RandomState(seed if seed is not None else 42)
    
    T = _build_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    r = target_rank
    
    for trial in range(max_trials):
        # Random initialization
        U = rng.randint(-2, 3, size=(nm, r)).astype(np.float64)
        V = rng.randint(-2, 3, size=(mp, r)).astype(np.float64)
        W = rng.randint(-2, 3, size=(pn, r)).astype(np.float64)
        
        # Alternating minimization
        converged = False
        for it in range(200):
            # Fix V and W, solve for U
            # T = sum_c u_c ⊗ v_c ⊗ w_c
            # Reshape: for each c, the contribution to T is u_c (nm) * v_c (mp) * w_c (pn)
            # Matricize T along mode 1: T1[i, (j,k)] = T[i,j,k]
            # T1 = sum_c u_c * (v_c ⊗ w_c)  where (v_c ⊗ w_c) is mp*pn vector
            # So T1 = U * (V ⊙ W)^T  where ⊙ is the Khatri-Rao product (column-wise outer product)
            
            # Khatri-Rao product of V and W: (mp*pn) x r
            # Column c of V⊙W is V[:,c] ⊗ W[:,c]
            VxW = np.zeros((mp * pn, r), dtype=np.float64)
            for c in range(r):
                VxW[:, c] = np.kron(V[:, c], W[:, c])
            T1 = T.reshape(nm, mp * pn)
            # Solve T1 = U * VxW^T  =>  U = T1 * (VxW^T)^{-1}  (least squares)
            try:
                U_new, _, _, _ = np.linalg.lstsq(VxW.T, T1.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                break
            
            # Fix U and W, solve for V
            # T2[(i,j), k] = T[i,j,k], shape (nm*mp, pn)
            # T2 = (U ⊙ V) * W^T ... hmm, let me think
            # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            # For fixed k: T[:,:,k] = sum_c (U[:,c]*V[:,c]) * W[k,c]
            # So T[:,:,k] = (U⊙V) * W[k,:]  where U⊙V is (nm*mp) x r with column c = U[:,c]⊗V[:,c]
            # Reshape T to (nm*mp, pn): T2[i*m+j, k]
            UxV = np.zeros((nm * mp, r), dtype=np.float64)
            for c in range(r):
                UxV[:, c] = np.kron(U[:, c], V[:, c])
            T2 = T.reshape(nm * mp, pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(W.T, T2.T, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                break
            
            # Fix U and V, solve for W
            UxV2 = np.zeros((nm * mp, r), dtype=np.float64)
            for c in range(r):
                UxV2[:, c] = np.kron(U[:, c], V[:, c])
            T3 = T.reshape(nm * mp, pn)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UxV2.T, T3.T, rcond=None)
                W = W_new.T
            except np.linalg.LinAlgError:
                break
            
            # Check convergence
            residual = 0.0
            reconstructed = np.zeros_like(T, dtype=np.float64)
            for c in range(r):
                reconstructed += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            residual = np.linalg.norm(reconstructed - T)
            if residual < 1e-8:
                converged = True
                break
        
        if converged:
            # Try to round to integers/halves
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify
            if _verify_decomposition_fuzzy(U_round, V_round, W_round, n, m, p, tol=1e-6):
                # Convert to lists
                U_list = [[float(x) if x != int(x) else int(x) for x in row] for row in U_round]
                V_list = [[float(x) if x != int(x) else int(x) for x in row] for row in V_round]
                W_list = [[float(x) if x != int(x) else int(x) for x in row] for row in W_round]
                return {"U": U_list, "V": V_list, "W": W_list}
            else:
                # Try without rounding
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_low_rank_structured(n, m, p):
    """Try some structured low-rank decompositions.
    
    For M_{3,3,3}, the known minimum rank is 23 (Lader). 
    For M_{3,3,7}, we can try to exploit the structure.
    
    One approach: decompose the 7-dimensional output into parts.
    E.g., write the 3x7 output as concatenation of a 3x3 and a 3x4 block.
    rank(M_{3,3,7}) <= rank(M_{3,3,3}) + rank(M_{3,3,4})
    
    But we don't know rank(M_{3,3,4}) easily.
    
    Another approach: use the fact that M_{3,3,7} can be embedded in a larger structure.
    
    Let's just try random searches for various ranks.
    """
    pass


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63
    
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Try to find lower rank decompositions
    # Known lower bound: rank >= max(nm, mp, pn) = max(9, 21, 21) = 21
    # Also rank >= ceil(nmp / max(n,m,p)) = ceil(63/7) = 9 (weak)
    # For 3x3x3, rank is 23. For 3x3x7, it's likely significantly higher.
    
    # Try ranks from a reasonable lower bound up
    # Let's try a few targeted ranks
    # First, try some ranks that might be achievable
    # The matricization rank gives an upper bound
    T = _build_tensor(n, m, p)
    mat_rank = _rank_of_tensor(T)
    
    # Try a range of ranks
    # We'll be strategic: try ranks that are multiples or near known values
    candidate_ranks = set()
    # Matricization rank is an upper bound
    candidate_ranks.add(mat_rank)
    # Try some values below
    for r in range(max(21, mat_rank - 10), min(best_rank, mat_rank + 5)):
        candidate_ranks.add(r)
    # Also try the schoolbook rank and some below
    for r in range(best_rank - 20, best_rank + 1):
        candidate_ranks.add(r)
    
    # Sort candidates, try from low to high
    candidate_ranks = sorted(candidate_ranks)
    
    # Limit the number of trials per rank to stay within time
    trials_per_rank = 10
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            break
        if time.time() > deadline:
            break
        
        trials_remaining = trials_per_rank
        trial_seed = 0
        while trials_remaining > 0 and time.time() < deadline:
            result = _try_rank_via_linear_solve(n, m, p, target_rank, 
                                                  max_trials=trials_remaining, 
                                                  seed=trial_seed)
            if result is not None:
                # Verify
                if _verify_decomposition_fuzzy(result["U"], result["V"], result["W"], n, m, p, tol=1e-4):
                    # Check rank
                    actual_rank = len(result["U"][0]) if result["U"] else 0
                    if actual_rank <= target_rank and actual_rank < best_rank:
                        best_rank = actual_rank
                        best = result
                        # Found a better one, stop trying this rank
                        trials_remaining = 0
                    else:
                        trials_remaining -= 1
                        trial_seed += 1
                else:
                    trials_remaining -= 1
                    trial_seed += 1
            else:
                trials_remaining = 0
    
    return best


# Keep the EVOLVE-BLOCK markers as required
# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
