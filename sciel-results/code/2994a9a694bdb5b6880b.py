import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _make_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    # Check integer/half-integer
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    
    # Check tensor identity using einsum
    # U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _snap_and_check(U, V, W, tol=1e-9):
    # Snap to nearest half-integer
    U2 = np.round(U * 2) / 2.0
    V2 = np.round(V * 2) / 2.0
    W2 = np.round(W * 2) / 2.0
    
    # Check if the snap was close enough
    if np.linalg.norm(U - U2) > 1e-6 or np.linalg.norm(V - V2) > 1e-6 or np.linalg.norm(W - W2) > 1e-6:
        return None, None, None
        
    if _verify_fast(U2, V2, W2, tol):
        return U2, V2, W2
    return None, None, None

def _als_update(U, V, W, n, m, p, R):
    """
    One step of Alternating Least Squares.
    """
    # Update U
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i * m + j, :] = u

    # Update V
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j * p + k, :] = v

    # Update W
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k * n + i, :] = w

def _calculate_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.sum((T - 1.0)**2)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # 2. Try to find better rank using ALS
    # We iterate R from 119 down to 100.
    
    seeds = [42, 123, 999, 777, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for R in range(119, 99, -1):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        
        # Allocate time per rank
        # If time is tight, fewer restarts
        max_restarts = 2 if time_left < 30 else 5
        
        for seed_idx in range(max_restarts):
            if time.time() > deadline:
                break
                
            seed = seeds[seed_idx % len(seeds)]
            np.random.seed(seed)
            
            # Initialization: Small integers to stay near the manifold
            # Random integers in [-2, 2]
            U = np.random.randint(-2, 3, (n * m, R)).astype(float)
            V = np.random.randint(-2, 3, (m * p, R)).astype(float)
            W = np.random.randint(-2, 3, (p * n, R)).astype(float)
            
            # Run ALS
            prev_err = float('inf')
            stagnant_count = 0
            
            for it in range(100):
                if time.time() > deadline - 1.0:
                    break
                
                _als_update(U, V, W, n, m, p, R)
                
                err = _calculate_error(U, V, W, n, m, p)
                
                if err < 1e-6:
                    U_s, V_s, W_s = _snap_and_check(U, V, W)
                    if U_s is not None:
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                if err < prev_err * 0.99:
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                
                prev_err = err
                
                if stagnant_count > 20:
                    break
            
            # Final check
            U_s, V_s, W_s = _snap_and_check(U, V, W)
            if U_s is not None:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    return best_result

# Fallback in case construct is not called or for direct testing
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = DIMENSIONS
    ok = _verify_fast(U, V, W)
    if ok:
        print("Verification Passed")
    else:
        print("Verification Failed")
