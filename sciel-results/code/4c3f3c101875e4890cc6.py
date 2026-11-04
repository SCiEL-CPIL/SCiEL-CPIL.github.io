import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _laderman_3x3x3():
    """
    Returns U, V, W for Laderman's algorithm for 3x3x3 matrix multiplication.
    Rank 23.
    U: 9x23, V: 9x23, W: 9x23
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Laderman's 23 multiplication terms
    # Each term is (a_coeffs, b_coeffs) where coeffs are length-9 vectors
    # U index: i*3+j for A[i,j]
    # V index: j*3+k for B[j,k]
    # W index: k*3+i for C[i,k]
    
    terms = []
    # 1. a00*b00
    terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]))
    # 2. a01*b10
    terms.append(([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    # 3. a02*b20
    terms.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]))
    # 4. a10*b01
    terms.append(([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    # 5. a11*b11
    terms.append(([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # 6. a12*b21
    terms.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]))
    # 7. a20*b02
    terms.append(([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    # 8. a21*b12
    terms.append(([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0]))
    # 9. a22*b22
    terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # 10. (a00+a11)*(b00+b11)
    terms.append(([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [-1,0,0,0,0,0,0,0,0]))
    # 11. (a01+a12)*(b10+b21)
    terms.append(([0,1,0,0,0,1,0,0,0], [0,0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,0,1]))
    # 12. (a02+a20)*(b20+b02)
    terms.append(([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]))
    # 13. (a10+a21)*(b01+b12)
    terms.append(([0,0,0,1,0,0,0,1,0], [0,1,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # 14. (a00+a01+a11)*(b00+b10+b11)
    terms.append(([1,1,0,0,1,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # 15. (a11+a12+a22)*(b11+b21+b22)
    terms.append(([0,0,0,0,1,1,0,0,1], [0,0,0,0,1,0,0,1,1], [0,0,0,0,0,0,0,0,0]))
    # 16. (a00+a02+a22)*(b00+b20+b22)
    terms.append(([1,0,1,0,0,0,0,0,1], [1,0,0,0,0,0,1,0,1], [1,0,0,0,0,0,0,0,0]))
    # 17. (a10+a11+a21)*(b01+b11+b12)
    terms.append(([0,0,0,1,1,0,0,1,0], [0,1,0,0,1,1,0,0,0], [0,0,0,1,0,0,0,0,0]))
    # 18. (a01+a12+a22)*(b10+b21+b22)
    terms.append(([0,1,0,0,0,1,0,0,1], [0,0,0,1,0,0,0,1,1], [0,0,0,0,0,0,0,1,0]))
    # 19. (a02+a10+a21)*(b20+b01+b12)
    terms.append(([0,0,1,1,0,0,0,1,0], [0,1,0,0,0,1,1,0,0], [0,0,0,0,1,0,0,0,0]))
    # 20. (a00+a01+a02+a11+a12+a22)*(b00+b10+b20+b11+b21+b22)
    terms.append(([1,1,1,0,1,1,0,0,1], [1,0,0,1,0,1,1,1,1], [1,0,0,0,0,0,0,0,0]))
    # 21. (a00+a01+a10+a11+a20+a21)*(b00+b01+b10+b11+b20+b21)
    terms.append(([1,1,0,1,1,0,1,1,0], [1,1,0,1,1,0,1,1,0], [0,0,0,0,0,0,0,0,1]))
    # 22. (a02+a01+a12+a11+a22+a21)*(b20+b10+b21+b11+b22+b12)
    terms.append(([0,1,1,0,1,1,0,1,1], [0,0,0,1,0,1,1,1,1], [0,0,0,0,0,0,0,0,1]))
    # 23. (a02+a01+a10+a12+a20+a21)*(b20+b01+b02+b12+b00+b10)
    terms.append(([0,1,1,1,0,1,1,1,0], [1,1,1,1,0,1,1,0,0], [0,0,0,0,0,0,0,0,1]))
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    for r in range(23):
        ca, cb, cc = terms[r]
        U[:, r] = ca
        V[:, r] = cb
        W[:, r] = cc
    
    return U, V, W

def _verify_laderman(U, V, W):
    """Verify Laderman's decomposition is correct."""
    A_test = np.random.rand(3, 3)
    B_test = np.random.rand(3, 3)
    C_test = A_test @ B_test
    
    U_flat = U.reshape(9, 23)
    V_flat = V.reshape(9, 23)
    W_flat = W.reshape(9, 23)
    
    C_calc = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            s = 0.0
            for r in range(23):
                u_prod = 0.0
                v_prod = 0.0
                for j in range(3):
                    u_prod += U_flat[i*3+j, r] * A_test[i, j]
                    v_prod += V_flat[j*3+k, r] * B_test[j, k]
                s += W_flat[k*3+i, r] * u_prod * v_prod
            C_calc[i, k] = s
    
    return np.allclose(C_test, C_calc, atol=1e-9)

def _schoolbook_3x3x3():
    """Schoolbook for 3x3 x 3x3, rank 9."""
    U = np.zeros((9, 9))
    V = np.zeros((9, 9))
    W = np.zeros((9, 9))
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, idx] = 1
                V[j*3+k, idx] = 1
                W[k*3+i, idx] = 1
                idx += 1
    return U, V, W

def _schoolbook_3x3x2():
    """Schoolbook for 3x3 x 3x2, rank 6."""
    # A is 3x3, B is 3x2, C is 3x2
    # U: 9x6 (i*3+j)
    # V: 6x6 (j*2+k)
    # W: 6x6 (k*3+i) where k is 0 or 1, i is 0,1,2
    # Actually W index: k*n+i where n=3, so k*3+i
    U = np.zeros((9, 6))
    V = np.zeros((6, 6))
    W = np.zeros((6, 6))
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*3+j, idx] = 1
                V[j*2+k, idx] = 1
                W[k*3+i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try to use Laderman for two 3x3x3 blocks and schoolbook for 3x3x2 block
    # Total: 23 + 23 + 6 = 52
    
    U_lad, V_lad, W_lad = _laderman_3x3x3()
    
    # Verify Laderman
    if not _verify_laderman(U_lad, V_lad, W_lad):
        # Laderman is incorrect, fall back to schoolbook
        return _schoolbook_full()
    
    U_sb3x2, V_sb3x2, W_sb3x2 = _schoolbook_3x3x2()
    
    # Build combined decomposition
    # Block 1: columns 0-2 of B (3x3 x 3x3)
    # Block 2: columns 3-5 of B (3x3 x 3x3)  
    # Block 3: columns 6-7 of B (3x3 x 3x2)
    
    rank_total = 23 + 23 + 6  # 52
    
    U_full = np.zeros((9, rank_total))
    V_full = np.zeros((24, rank_total))  # m*p = 3*8 = 24
    W_full = np.zeros((24, rank_total))  # p*n = 8*3 = 24
    
    # Block 1: Laderman for cols 0-2
    for r in range(23):
        # U: same as Laderman (9x1)
        U_full[:, r] = U_lad[:, r]
        # V: Laderman's V is 9x1 (j*3+k for k=0,1,2)
        # But we need V for j*8+k where k=0,1,2
        for j in range(3):
            for k in range(3):
                v_idx_lad = j*3+k
                v_idx_full = j*8+k
                V_full[v_idx_full, r] = V_lad[v_idx_lad, r]
        # W: Laderman's W is 9x1 (k*3+i for k=0,1,2)
        # We need W for k*3+i where k=0,1,2
        W_full[:, r] = W_lad[:, r]
    
    # Block 2: Laderman for cols 3-5
    offset = 23
    for r in range(23):
        col = r + offset
        # U: same as Laderman
        U_full[:, col] = U_lad[:, r]
        # V: Laderman's V (j*3+k) maps to full V (j*8 + (k+3))
        for j in range(3):
            for k in range(3):
                v_idx_lad = j*3+k
                v_idx_full = j*8 + (k+3)
                V_full[v_idx_full, col] = V_lad[v_idx_lad, r]
        # W: Laderman's W (k*3+i) maps to full W ((k+3)*3 + i)
        for k in range(3):
            for i in range(3):
                w_idx_lad = k*3+i
                w_idx_full = (k+3)*3 + i
                W_full[w_idx_full, col] = W_lad[w_idx_lad, r]
    
    # Block 3: Schoolbook for cols 6-7 (3x3 x 3x2)
    offset2 = 46
    for r in range(6):
        col = r + offset2
        # U: same as schoolbook 3x3x2
        U_full[:, col] = U_sb3x2[:, r]
        # V: schoolbook V is 6x1 (j*2+k for k=0,1)
        # Map to full V: j*8 + (k+6)
        for j in range(3):
            for k in range(2):
                v_idx_sb = j*2+k
                v_idx_full = j*8 + (k+6)
                V_full[v_idx_full, col] = V_sb3x2[v_idx_sb, r]
        # W: schoolbook W is 6x1 (k*3+i for k=0,1)
        # Map to full W: (k+6)*3 + i
        for k in range(2):
            for i in range(3):
                w_idx_sb = k*3+i
                w_idx_full = (k+6)*3 + i
                W_full[w_idx_full, col] = W_sb3x2[w_idx_sb, r]
    
    # Verify the full decomposition
    A_test = np.random.rand(3, 3)
    B_test = np.random.rand(3, 8)
    C_test = A_test @ B_test
    
    C_calc = np.zeros((3, 8))
    for i in range(3):
        for k in range(8):
            s = 0.0
            for r in range(rank_total):
                u_prod = 0.0
                v_prod = 0.0
                for j in range(3):
                    u_prod += U_full[i*3+j, r] * A_test[i, j]
                    v_prod += V_full[j*8+k, r] * B_test[j, k]
                s += W_full[k*3+i, r] * u_prod * v_prod
            C_calc[i, k] = s
    
    if not np.allclose(C_test, C_calc, atol=1e-9):
        # Verification failed, fall back to schoolbook
        return _schoolbook_full()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        return _schoolbook_full()
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

def _schoolbook_full():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
