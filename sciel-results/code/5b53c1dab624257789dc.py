import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    deadline = time.time() + 140  # Safety margin
    
    # Build rank-44 Strassen-based construction
    U, V, W = _build_rank44(n, m, p)
    
    # Verify correctness, fallback to schoolbook if needed
    if not _verify(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _try_reduce(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by checking for linear dependencies among rank-1 terms."""
    r = len(U[0])
    
    # Convert to arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor target
    target = np.zeros((n, m, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k, i] = 1.0
    
    # Try removing columns one by one and see if we can compensate
    for c in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Skip if already zero
        if np.allclose(U_arr[:, c], 0) and np.allclose(V_arr[:, c], 0) and np.allclose(W_arr[:, c], 0):
            U_arr = np.delete(U_arr, c, axis=1)
            V_arr = np.delete(V_arr, c, axis=1)
            W_arr = np.delete(W_arr, c, axis=1)
            r -= 1
            continue
        
        # Try to see if we can adjust other columns to compensate
        # This is a hard problem, so use a heuristic: try small perturbations
        # For now, just check norm
        norm_c = np.linalg.norm(U_arr[:, c]) + np.linalg.norm(V_arr[:, c]) + np.linalg.norm(W_arr[:, c])
        if norm_c < 1e-10:
            U_arr = np.delete(U_arr, c, axis=1)
            V_arr = np.delete(V_arr, c, axis=1)
            W_arr = np.delete(W_arr, c, axis=1)
            r -= 1
    
    # Convert back to lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    # Final verification
    if not _verify(U_out, V_out, W_out, n, m, p):
        # Return original if reduction broke something
        return U, V, W
    
    return U_out, V_out, W_out

def _build_rank44(n, m, p):
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition matches matrix multiplication tensor."""
    r = len(U[0])
    # Check T[i,j,k,i'] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i',r] = delta_{i,i'}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + ip][r_idx]
                    expected = 1.0 if i == ip else 0.0
                    if abs(s - expected) > 1e-10:
                        return False
    return True

def _try_reduce(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by checking for linear dependencies among rank-1 terms."""
    r = len(U[0])
    
    # Convert to arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor target
    target = np.zeros((n, m, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k, i] = 1.0
    
    # Try removing columns one by one and see if we can compensate
    for c in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Skip if already zero
        if np.allclose(U_arr[:, c], 0) and np.allclose(V_arr[:, c], 0) and np.allclose(W_arr[:, c], 0):
            U_arr = np.delete(U_arr, c, axis=1)
            V_arr = np.delete(V_arr, c, axis=1)
            W_arr = np.delete(W_arr, c, axis=1)
            r -= 1
            continue
        
        # Try to see if we can adjust other columns to compensate
        # This is a hard problem, so use a heuristic: try small perturbations
        # For now, just check norm
        norm_c = np.linalg.norm(U_arr[:, c]) + np.linalg.norm(V_arr[:, c]) + np.linalg.norm(W_arr[:, c])
        if norm_c < 1e-10:
            U_arr = np.delete(U_arr, c, axis=1)
            V_arr = np.delete(V_arr, c, axis=1)
            W_arr = np.delete(W_arr, c, axis=1)
            r -= 1
    
    # Convert back to lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    # Final verification
    if not _verify(U_out, V_out, W_out, n, m, p):
        # Return original if reduction broke something
        return U, V, W
    
    return U_out, V_out, W_out

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
