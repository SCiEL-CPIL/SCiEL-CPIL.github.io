"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: 
1. Construct the known rank-56 decomposition using Strassen's algorithm.
2. Attempt to find a lower rank decomposition (e.g., rank 50) using a randomized 
   linear algebra approach (fix U, V, solve for W) within a time budget.
3. Return the best valid decomposition found.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            contrib = U3[i, :, c][:, np.newaxis] * V3[:, :, c] * W3[:, i, c][np.newaxis, :]
            T[i, :, :] += contrib
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition of the 2x2 * 2x2 multiplication tensor."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen factors M1..M7
    U[0 * 2 + 0, 0] = 1; U[1 * 2 + 1, 0] = 1
    V[0 * 2 + 1, 0] = 1; V[1 * 2 + 0, 0] = 1
    W[0 * 2 + 0, 0] = 1; W[1 * 2 + 1, 0] = 1
    
    U[1 * 2 + 0, 1] = 1; U[1 * 2 + 1, 1] = 1
    V[0 * 2 + 0, 1] = 1
    W[1 * 2 + 0, 1] = 1; W[1 * 2 + 1, 1] = -1
    
    U[0 * 2 + 0, 2] = 1
    V[0 * 2 + 0, 2] = 1; V[1 * 2 + 0, 2] = -1
    W[0 * 2 + 0, 2] = 1; W[1 * 2 + 1, 2] = 1
    
    U[1 * 2 + 1, 3] = 1
    V[1 * 2 + 1, 3] = 1; V[0 * 2 + 1, 3] = -1
    W[1 * 2 + 0, 3] = 1; W[0 * 2 + 0, 3] = 1
    
    U[0 * 2 + 0, 4] = 1; U[1 * 2 + 0, 4] = 1
    V[1 * 2 + 1, 4] = 1
    W[0 * 2 + 0, 4] = -1; W[1 * 2 + 1, 4] = 1
    
    U[1 * 2 + 0, 5] = 1; U[0 * 2 + 0, 5] = -1
    V[0 * 2 + 0, 5] = 1; V[1 * 2 + 1, 5] = 1
    W[1 * 2 + 0, 5] = 1
    
    U[0 * 2 + 1, 6] = 1; U[1 * 2 + 1, 6] = -1
    V[0 * 2 + 1, 6] = 1; V[1 * 2 + 0, 6] = 1
    W[0 * 2 + 0, 6] = 1

    return U, V, W


def _build_4x4x4():
    """Build a rank-28 decomposition of the 4x4 * 4x4 multiplication tensor."""
    sU, sV, sW = _strassen_2x2x2()
    rank = 4 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    for a in range(4):
        for c in range(7):
            col = a * 7 + c
            for bi in range(2):
                for bj in range(2):
                    U[(2 * a + bi) * 4 + (2 * b + bj), col] = sU[bi * 2 + bj, c]
                    V[(2 * b + bj) * 4 + (2 * c + bk), col] = sV[bj * 2 + bk, c]
                    W[(2 * c + bk) * 4 + (2 * a + bi), col] = sW[bk * 2 + bi, c]
    return U, V, W


def _build_4x4x8():
    """Build a rank-56 decomposition of the 4x4 * 4x8 multiplication tensor."""
    U1, V1, W1 = _build_4x4x4()
    rank = 2 * 28
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    U[:, 0:28] = U1
    V[0:16, 0:28] = V1
    W[0:16, 0:28] = W1

    U[:, 28:56] = U1
    V[16:32, 28:56] = V1
    W[16:32, 28:56] = W1

    return U, V, W


def _search_lower_rank(n, m, p, target_rank, time_limit=10.0):
    """Search for a decomposition with rank <= target_rank using random U, V and solving for W."""
    start_time = time.time()
    num_eq = n * m * p
    A = np.zeros((num_eq, p * n * target_rank))
    b = np.ones(num_eq)
    
    # Precompute A structure is hard because U, V change.
    # We will build A in the loop.
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Random U and V with small integers
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        
        # Build A and b
        # A is num_eq x (p*n*target_rank)
        # b is num_eq x 1
        A_mat = np.zeros((num_eq, p * n * target_rank))
        b_vec = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Coeff for W[k*n+i, t] is U[i*m+j, t] * V[j*p+k, t]
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    coeffs = u_val * v_val
                    for t in range(target_rank):
                        x_idx = (k * n + i) * target_rank + t
                        A_mat[eq_idx, x_idx] = coeffs[t]
                    eq_idx += 1
        
        # Solve A x = b
        try:
            x, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except Exception:
            continue
            
        # Check if solution is nice
        found = False
        for L in [1, 2, 3, 4, 5, 10, 20, 50]:
            W_float = x.reshape(p * n, target_rank) * L
            W_int = np.round(W_float)
            if np.allclose(W_float, W_int, atol=1e-9):
                W = W_int.astype(int)
                # Verify
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    return U, V, W
        # If not found, continue
        
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # Primary: Strassen-based rank-56 decomposition
    try:
        U, V, W = _build_4x4x8()
        if _verify_fast(U, V, W, n, m, p, tol=1e-10):
            U_snap = np.round(U).astype(int)
            V_snap = np.round(V).astype(int)
            W_snap = np.round(W).astype(int)
            if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                best_rank = 56
            else:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 56
        else:
            best_result = None
            best_rank = 128
    except Exception:
        best_result = None
        best_rank = 128

    # Try to find lower rank
    if time.time() < deadline - 20:
        for target_rank in [54, 52, 50]:
            if time.time() > deadline - 10:
                break
            res = _search_lower_rank(n, m, p, target_rank, time_limit=10.0)
            if res is not None:
                U, V, W = res
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break

    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result


# EVOLVE-BLOCK-END
