import time
import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    The decomposition is derived from the Kronecker product of two Strassen decompositions for 2x2 matrices.
    
    Interface:
    U: (16, 49) matrix where row i*4+j corresponds to A_{i,j}
    V: (16, 49) matrix where row j*4+k corresponds to B_{j,k}
    W: (16, 49) matrix where row k*4+i corresponds to C_{i,k}
    """
    
    # 1. Define 2x2 Strassen components
    # U: 7x4
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=np.int64)
    
    # V: 7x4
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=np.int64)
    
    # W: 7x4 (Raw matrix indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1))
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=np.int64)
    
    # 2. Construct 4x4 U, V, W using Kronecker products
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    for r in range(7):
        if time.time() > deadline:
            break
        for s in range(7):
            if time.time() > deadline:
                break
            col = r * 7 + s
            
            # Reshape to 2x2 matrices
            Ur = U2[r].reshape(2, 2)
            Us = U2[s].reshape(2, 2)
            Vr = V2[r].reshape(2, 2)
            Vs = V2[s].reshape(2, 2)
            Wr = W2[r].reshape(2, 2)
            Ws = W2[s].reshape(2, 2)
            
            # Kronecker products
            U_kron = np.kron(Ur, Us)
            V_kron = np.kron(Vr, Vs)
            W_kron = np.kron(Wr, Ws)
            
            # Fill U4 and V4 (Row-major flatten matches index I*4+J)
            U4[:, col] = U_kron.flatten()
            V4[:, col] = V_kron.flatten()
            
            # Fill W4
            # W4 index is K*4 + I.
            # W_kron has entries at (I, K).
            # We need to map (I, K) -> K*4 + I.
            # This is equivalent to flattening W_kron.T in row-major order.
            W4[:, col] = W_kron.T.flatten()
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

if __name__ == "__main__":
    # Quick validation
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test
    np.random.seed(0)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    C_expected = A @ B
    
    # Compute C_decomposed
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposed: C_{i,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) (sum_k B_{j,k} V_{j,k,r}) ... wait.
    # The tensor is T = sum_r u_r \otimes v_r \otimes w_r.
    # <A, U_r> <B, V_r> is the scalar M_r.
    # Then C = sum_r M_r W_r.
    
    # M_r = sum_{i,j} A_{i,j} U_{i,j,r}
    # U is 16x49. Index i*4+j.
    # V is 16x49. Index j*4+k.
    # W is 16x49. Index k*4+i.
    
    # Let's compute M
    A_flat = A.flatten() # row major: i*4+j
    B_flat = B.flatten() # row major: j*4+k
    
    AU = A_flat @ U # (49,)
    BV = B_flat @ V # (49,)
    M = AU * BV     # (49,)
    
    # C_flat_expected = sum_r M_r * W[:, r]
    # But W is stored as W[k*4+i, r].
    # So C_flat_decomposed = M @ W
    C_flat_calc = M @ W
    
    # C_flat_calc is indexed by k*4 + i.
    # We need to reshape it to (4,4) C[i,k].
    # k*4 + i -> if we reshape in Fortran order, index 0 is (0,0), 1 is (1,0)...
    # Let's check:
    # Linear index L = k*4 + i.
    # In Fortran order (column-major), index L corresponds to row L % 4, col L // 4.
    # So row = i, col = k.
    # So C_calc[i, k] is the value.
    
    C_calc = C_flat_calc.reshape((4, 4), order='F')
    
    if np.allclose(C_calc, C_expected, atol=1e-6):
        print("Validation Passed")
    else:
        print("Validation Failed")
        print(C_calc)
        print(C_expected)
