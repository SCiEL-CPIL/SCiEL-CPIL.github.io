import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 130.0

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _validate_exact(U, V, W, n, m, p):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True

def _to_list(M):
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result

def _try_rank(r, n, m, p, seed=0):
    rng = np.random.default_rng(seed)
    # Try to find r components
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # We need sum_c (W_ki^c)^2 = 1
    # This is hard to satisfy exactly with random search.
    # Instead, we just check the UV=W condition.
    # If we find r components satisfying UV=W, we check if they sum to the tensor.
    # Actually, if UV=W, the tensor term is W^2.
    # So we need sum_c W_c^2 = 1.
    
    # This is a constraint satisfaction problem.
    # For now, just random search for UV=W.
    for c in range(r):
        # Generate random V_c
        V_c = rng.integers(-2, 3, size=(m, p)) / 2.0
        # Generate random U_c
        U_c = rng.integers(-2, 3, size=(n, m)) / 2.0
        # Compute W_c = U_c * V_c (element-wise product sum? No, U_ij V_jk)
        # We need U_ij V_jk = W_ki
        # This is only possible if U_ij V_jk is independent of j.
        # Check consistency
        valid = True
        W_c = np.zeros((p, n))
        for i in range(n):
            for k in range(p):
                prods = [U_c[i, j] * V_c[j, k] for j in range(m)]
                if not np.allclose(prods, prods[0], atol=1e-9):
                    valid = False
                    break
                W_c[k, i] = prods[0]
            if not valid: break
        if not valid:
            continue
        # Check if W_c has half-integer entries
        W2_c = np.round(2 * W_c)
        if not np.allclose(W_c, W2_c / 2.0, atol=1e-9):
            continue
            
        U[:, c] = U_c.flatten()
        V[:, c] = V_c.flatten()
        W[:, c] = W_c.flatten()
        
    # Verify the full decomposition
    # Check UV=W for all c
    if not _validate_exact(U, V, W, n, m, p):
        return None
        
    # Check sum W_c^2 = 1
    # W is (p*n, r)
    W_sq_sum = np.sum(W**2, axis=1)
    if not np.allclose(W_sq_sum, 1.0, atol=1e-9):
        return None
        
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p
    
    for r in range(24, best_rank):
        if time.time() > DEADLINE - 5:
            break
        res = _try_rank(r, n, m, p, seed=r)
        if res is not None:
            best = res
            best_rank = r
            break
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
