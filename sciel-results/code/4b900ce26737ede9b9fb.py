import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_small(U, V, W, r):
    """Verify 2x2x3 decomposition."""
    # U: 4 x r, V: 6 x r, W: 6 x r
    # i in 0..1, j in 0..1, k in 0..2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0
                for c in range(r):
                    s += U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c]
                expected = 1 if i == k else 0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _find_small_decomp(deadline):
    """Search for rank 10 or 11 decomposition of 2x2x3."""
    # Try rank 10 first, then 11
    for r in [10, 11]:
        attempts = 0
        while time.time() < deadline - 10 and attempts < 2000:
            attempts += 1
            U = np.random.randint(-1, 2, size=(4, r))
            V = np.random.randint(-1, 2, size=(6, r))
            W = np.random.randint(-1, 2, size=(6, r))
            if _verify_small(U, V, W, r):
                return U, V, W, r
    return None

def _kron_decompose(U1, V1, W1, r1):
    """Construct 4x4x9 from 2x2x3."""
    n, m, p = 4, 4, 9
    R = r1 * r1
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(3):
                        for k2 in range(3):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            k = k1*3 + k2
                            
                            for c1 in range(r1):
                                for c2 in range(r1):
                                    c = c1*r1 + c2
                                    
                                    u_val = U1[i1*2+j1, c1] * U1[i2*2+j2, c2]
                                    if u_val != 0:
                                        U[i, j, c] += u_val
                                        
                                    v_val = V1[j1*3+k1, c1] * V1[j2*3+k2, c2]
                                    if v_val != 0:
                                        V[j, k, c] += v_val
                                        
                                    w_val = W1[k1*2+i1, c1] * W1[k2*2+i2, c2]
                                    if w_val != 0:
                                        W[k, i, c] += w_val
                                        
    return U, V, W, R

def _verify_large(U, V, W, R):
    """Verify 4x4x9 decomposition."""
    n, m, p = 4, 4, 9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1 if i == k else 0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _schoolbook():
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

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Try Kronecker approach
    small_res = _find_small_decomp(deadline)
    if small_res is not None:
        U1, V1, W1, r1 = small_res
        U, V, W, R = _kron_decompose(U1, V1, W1, r1)
        
        # Verify large decomposition
        if _verify_large(U, V, W, R):
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback
    return _schoolbook()
