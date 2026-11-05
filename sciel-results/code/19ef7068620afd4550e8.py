import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def check_time():
    return (time.time() - START_TIME) >= TIME_LIMIT

def verify_decomposition(U, V, W, n, m, p, rank, atol=1e-8):
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    return np.allclose(T_recon, T_target, atol=atol)

def build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def solve_W(U, V, n, m, p, rank):
    """Given U,V, solve for W such that decomposition holds. Returns W or None."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    # M[i,j,k,r] = U[i,j,r]*V[j,k,r]; want sum_r M[i,j,k,r]*W[k,i,r] = 1
    # => (M^T) @ W_flat = ones, where M is (n*m*p, rank)
    M = np.einsum('ijr,jkr->ijk', U_r, V_r).reshape(-1, rank)
    ones = np.ones(n * m * p)
    try:
        W_flat, residuals, rank_svd, sv = np.linalg.lstsq(M, ones, rcond=None)
    except Exception:
        return None
    W = W_flat.reshape(p, n, rank)
    return W

def objective(U, V, n, m, p, rank):
    """Smooth objective: ||reconstructed - 1||_F^2, with W solved optimally."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    M = np.einsum('ijr,jkr->ijk', U_r, V_r).reshape(-1, rank)
    ones = np.ones(n * m * p)
    try:
        W_flat, _, _, _ = np.linalg.lstsq(M, ones, rcond=None)
    except Exception:
        return np.inf
    recon = W_flat @ M  # (n*m*p,)
    return float(np.sum((recon - ones) ** 2))

def build_strassen_block(n, m, p):
    """Block Strassen decomposition.

    Split n=4 into 2+2, p=6 into 3+3. The middle dim m=5 is shared.
    Block structure of C = A*B:
      C = [A1 B1; A2 B2] where A1,A2 are 2x5, B1,B2 are 5x3.
      C11 = A1*B1 (2x3), C12 = A1*B2, C21 = A2*B1, C22 = A2*B2.
    Each 2x3 * 3x3 ... no wait B is 5x3, A is 2x5, so A1*B1 is 2x5 * 5x3 = 2x3.
    Use Strassen on 2x2 blocks of the 2x3 result? 2x3 isn't square.

    Alternative: split n=4 into 2+2, m=5 into 2+3, p=6 into 3+3.
    Then A is 4x5 = [A11 A12; A21 A22] with A11,A21: 2x2, A12,A22: 2x3.
    B is 5x6 = [B11 B12; B21 B22] with B11,B21: 2x3, B12,B22: 3x3.
    C11 = A11 B11 + A12 B21 (2x2*2x3 + 2x3*3x3 = 2x3)
    C12 = A11 B12 + A12 B22 (2x3)
    C21 = A21 B11 + A22 B21 (2x3)
    C22 = A21 B12 + A22 B22 (2x3)
    This is messy. Let's just return schoolbook here.
    """
    return build_schoolbook(n, m, p)

def try_structured_search(n, m, p, max_rank=None):
    """Search for low-rank U,V such that W solved from them is small (ideally integer/half-integer)."""
    if max_rank is None:
        max_rank = n * m * p - 1

    best = None  # (U, V, W, rank, obj)
    vals = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    # Try ranks from a promising low value upward.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(20,30,24)=30.
    # Known: schoolbook 120. Try to beat it.
    target_ranks = list(range(30, max_rank + 1))

    for rank in target_ranks:
        if check_time():
            break
        # Number of restarts depends on time budget
        restarts = 200
        for restart in range(restarts):
            if check_time():
                break
            # Random integer/half-integer init
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            obj = objective(U, V, n, m, p, rank)
            if obj < 1e-10:
                W = solve_W(U, V, n, m, p, rank)
                if W is not None and verify_decomposition(U, V, W, n, m, p, rank):
                    return (U.copy(), V.copy(), W.copy(), rank)
            if best is None or obj < best[4]:
                W = solve_W(U, V, n, m, p, rank)
                if W is not None:
                    best = (U.copy(), V.copy(), W.copy(), rank, obj)

        # Gradient-free local polish on the best found for this rank
        if best is not None and best[3] == rank:
            U_b, V_b, W_b, _, obj_b = best
            for it in range(40):
                if check_time():
                    break
                # Perturb a random entry of U or V
                if random.random() < 0.5:
                    idx = np.random.randint(0, U_b.shape[0])
                    col = np.random.randint(0, rank)
                    U_new = U_b.copy()
                    U_new[idx, col] += np.random.choice([-0.5, 0.5])
                    obj_new = objective(U_new, V_b, n, m, p, rank)
                    if obj_new < obj_b:
                        U_b = U_new
                        obj_b = obj_new
                        W_b = solve_W(U_b, V_b, n, m, p, rank)
                else:
                    idx = np.random.randint(0, V_b.shape[0])
                    col = np.random.randint(0, rank)
                    V_new = V_b.copy()
                    V_new[idx, col] += np.random.choice([-0.5, 0.5])
                    obj_new = objective(U_b, V_new, n, m, p, rank)
                    if obj_new < obj_b:
                        V_b = V_new
                        obj_b = obj_new
                        W_b = solve_W(U_b, V_b, n, m, p, rank)
                if obj_b < 1e-10:
                    if W_b is not None and verify_decomposition(U_b, V_b, W_b, n, m, p, rank):
                        return (U_b.copy(), V_b.copy(), W_b.copy(), rank)

    return best[:4] if best is not None else None

def construct():
    n, m, p = DIMENSIONS

    U_school, V_school, W_school, rank_school = build_schoolbook(n, m, p)
    if not verify_decomposition(U_school, V_school, W_school, n, m, p, rank_school):
        pass

    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school

    result = try_structured_search(n, m, p, max_rank=rank_school - 1)
    if result is not None:
        U_found, V_found, W_found, rank_found = result
        if verify_decomposition(U_found, V_found, W_found, n, m, p, rank_found):
            best_U, best_V, best_W, best_rank = U_found, V_found, W_found, rank_found
            print(f"Found lower-rank decomposition with rank {rank_found}")

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
