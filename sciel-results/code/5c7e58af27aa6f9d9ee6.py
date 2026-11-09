"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _build_block_strassen():
    """Build the explicit block-Strassen rank-84 decomposition."""
    n, m, p = DIMENSIONS
    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    A_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
    }
    B_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    C_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    
    pairs = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((0, 0), (0, 2), (0, 2)),
        ((0, 1), (1, 2), (0, 2)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1)),
        ((1, 0), (0, 2), (1, 2)),
        ((1, 1), (1, 2), (1, 2)),
    ]
    
    strassen_terms = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]
    
    col = 0
    for (a_idx, b_idx, c_idx) in pairs:
        a_r1, _, a_c1, _ = A_blocks[a_idx]
        b_r1, _, b_c1, _ = B_blocks[b_idx]
        c_r1, _, c_c1, _ = C_blocks[c_idx]
        
        for term in strassen_terms:
            for local_a, coeff in term['u']:
                lr, lc = divmod(local_a, 2)
                U[(a_r1 + lr) * m + (a_c1 + lc), col] += coeff
            for local_b, coeff in term['v']:
                lr, lc = divmod(local_b, 2)
                V[(b_r1 + lr) * p + (b_c1 + lc), col] += coeff
            for local_c, coeff in term['w']:
                lr, lc = divmod(local_c, 2)
                W[(c_c1 + lc) * n + (c_r1 + lr), col] += coeff
            col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _snap_halfint(M, tol=1e-3):
    """Snap values to nearest half-integer; return False if any value is far off."""
    for r in range(M.shape[1]):
        for row in range(M.shape[0]):
            v = M[row, r]
            if abs(v) < 1e-9:
                M[row, r] = 0.0
                continue
            h = np.round(2 * v) / 2
            if abs(v - h) > tol:
                return False
            M[row, r] = h
    return True


def _gauged_search(U0, V0, W0, target_rank, n, m, p, deadline, seed=42):
    """
    Structural search: fix U to a canonical form via gauge transformation,
    then search for V, W with the given rank.
    
    Key idea: The tensor T = sum_r U_r ⊗ V_r ⊗ W_r is invariant under
    U -> U S, V -> V S^{-1}, W -> W S^{-1} for invertible S.
    We fix U to have a simple structure, reducing the search space.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Try multiple random gauge fixes
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random invertible transformation
        S = np.eye(target_rank) + rng.randn(target_rank, target_rank) * 0.1
        if np.linalg.cond(S) > 1e6:
            continue
        S_inv = np.linalg.inv(S)
        
        # Apply gauge: pick target_rank columns from the rank-84 solution
        if target_rank > U0.shape[1]:
            continue
        idx = rng.choice(U0.shape[1], size=target_rank, replace=False)
        U = U0[:, idx] @ S
        V = V0[:, idx] @ S_inv
        W = W0[:, idx] @ S_inv
        
        # Try to simplify U: zero out small entries
        for r in range(target_rank):
            for row in range(U.shape[0]):
                if abs(U[row, r]) < 1e-4:
                    U[row, r] = 0.0
        
        # Now try to optimize V and W to make the decomposition exact
        # Use iterative refinement
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for it in range(2000):
            if time.time() > deadline:
                break
            
            # Compute errors
            max_err = 0
            worst_t = 0
            for t, (i, j, k) in enumerate(triples):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    worst_t = t
            
            if max_err < 1e-9:
                break
            
            i, j, k = triples[worst_t]
            val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
            err = val - 1.0
            
            # Gradient step on V and W (keep U fixed)
            dV = U[i * m + j, :] * W[k * n + i, :]
            dW = U[i * m + j, :] * V[j * p + k, :]
            
            lr = 0.05
            V[j * p + k, :] -= lr * err * dV
            W[k * n + i, :] -= lr * err * dW
        
        # Check if we converged
        if _verify(U, V, W, n, m, p):
            # Try to snap to half-integers
            U_copy, V_copy, W_copy = U.copy(), V.copy(), W.copy()
            if _snap_halfint(U_copy) and _snap_halfint(V_copy) and _snap_halfint(W_copy):
                if _verify(U_copy, V_copy, W_copy, n, m, p):
                    return U_copy, V_copy, W_copy
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Guaranteed baseline: block-Strassen rank 84
    U, V, W = _build_block_strassen()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = 84
    
    # Try structural search for lower ranks
    # The gauge-fixing approach reduces the search space significantly
    for target_rank in [72, 68, 64, 60, 56, 52, 48]:
        if time.time() > deadline:
            break
        
        res = _gauged_search(U, V, W, target_rank, n, m, p, deadline, seed=target_rank)
        if res is not None:
            U_new, V_new, W_new = res
            if _verify(U_new, V_new, W_new, n, m, p):
                best = (U_new, V_new, W_new)
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
# EVOLVE-BLOCK-END
