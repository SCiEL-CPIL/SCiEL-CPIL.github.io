import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal: for each (i,j,k), sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal: for each (i,j,k) and (i2,j2,k2) != (i,j,k), the product should be 0
    # This is equivalent to: for each c, the rank-1 tensor U_c * V_c * W_c should only
    # have non-zero entries where all three indices are "consistent"
    # More efficiently: for each c, check that U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] is nonzero
    # only when (i,j,k) is a valid triple, and the sum over c gives the correct tensor.
    # Actually, the full check is: for all (i,j,k), the computed C[i,k] = sum_j A[i,j]B[j,k]
    # Let's just verify by computing the full tensor.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            val = np.sum(U_arr[i*m+j] * V_arr[j2*p+k2] * W_arr[k*n+i2])
                            # This should be 0 unless (i2,j2,k2) == (i,j,k)
                            # Wait, no. The tensor is T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                            # Actually the standard form is: T[i,j,k] where the tensor maps
                            # A[i,j] and B[j,k] to C[i,k]. The bilinear form is:
                            # <A,B> = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
                            # The rank-r decomposition: sum_c (sum_{i,j} U[i,j,c]*A[i,j]) *
                            #                                (sum_{j,k} V[j,k,c]*B[j,k]) *
                            #                                (sum_{k,i} W[k,i,c]*C[i,k])
                            # So T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            # And we need T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0.
                            # But actually T is only defined for valid triples. Let me just
                            # verify the full matrix product.
                            pass
    
    # Better approach: verify by checking that the decomposition correctly computes
    # C = A * B for random inputs
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        A_c = np.sum(U_arr[:, c].reshape(n, m) * A)
        B_c = np.sum(V_arr[:, c].reshape(m, p) * B)
        W_c = W_arr[:, c].reshape(p, n)
        C_computed += A_c * B_c * W_c.T
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    
    return True

def _strassen_2x2_factors():
    """Generate Strassen factors for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, 0, 1, 0, 0, 1, -1]    # C[1,1]
    ]
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Strassen."""
    U2, V2, W2 = _strassen_2x2_factors()
    
    U4 = [[0.0]*49 for _ in range(16)]
    V4 = [[0.0]*49 for _ in range(16)]
    W4 = [[0.0]*49 for _ in range(16)]
    
    for a in range(7):
        for b in range(7):
            c = a*7 + b
            for i in range(4):
                for j in range(4):
                    flat = i*4 + j
                    block_flat = (i//2)*2 + (j//2)
                    sub_flat = (i%2)*2 + (j%2)
                    U4[flat][c] = U2[block_flat][a] * U2[sub_flat][b]
                    V4[flat][c] = V2[block_flat][a] * V2[sub_flat][b]
                    W4[flat][c] = W2[block_flat][a] * W2[sub_flat][b]
    
    return U4, V4, W4

def _build_5x5x5_strassen():
    """Build rank-125 decomposition for 5x5x5 using 5x5x5 = 4x4x4 + 4x1x1 + 1x4x1 + 1x1x4 + 1x1x1."""
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # 5x5x5 can be decomposed as:
    # - 4x4x4 block (top-left)
    # - 4x1x1 strip (top-right of A, bottom of B)
    # - 1x4x1 strip (bottom of A, top of B)
    # - 1x1x4 strip (bottom-right of A, top of B)
    # - 1x1x1 corner
    
    U5 = [[0.0]*125 for _ in range(25)]
    V5 = [[0.0]*125 for _ in range(25)]
    W5 = [[0.0]*125 for _ in range(25)]
    
    # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3), rank 49
    for i in range(4):
        for j in range(4):
            for k in range(4):
                flat_U = i*5 + j
                flat_V = j*5 + k
                flat_W = k*5 + i
                for c in range(49):
                    U5[flat_U][c] += U4[i*4+j][c]
                    V5[flat_V][c] += V4[j*4+k][c]
                    W5[flat_W][c] += W4[k*4+i][c]
    
    # Part 2: 4x1x1 (i in 0..3, j=4, k=4), rank 4
    for i in range(4):
        c = 49 + i
        flat_U = i*5 + 4
        flat_V = 4*5 + 4
        flat_W = 4*5 + i
        U5[flat_U][c] = 1.0
        V5[flat_V][c] = 1.0
        W5[flat_W][c] = 1.0
    
    # Part 3: 1x4x1 (i=4, j in 0..3, k=4), rank 4
    for j in range(4):
        c = 53 + j
        flat_U = 4*5 + j
        flat_V = j*5 + 4
        flat_W = 4*5 + 4
        U5[flat_U][c] = 1.0
        V5[flat_V][c] = 1.0
        W5[flat_W][c] = 1.0
    
    # Part 4: 1x1x4 (i=4, j=4, k in 0..3), rank 4
    for k in range(4):
        c = 57 + k
        flat_U = 4*5 + 4
        flat_V = 4*5 + k
        flat_W = k*5 + 4
        U5[flat_U][c] = 1.0
        V5[flat_V][c] = 1.0
        W5[flat_W][c] = 1.0
    
    # Part 5: 1x1x1 (i=4, j=4, k=4), rank 1
    c = 61
    flat_U = 4*5 + 4
    flat_V = 4*5 + 4
    flat_W = 4*5 + 4
    U5[flat_U][c] = 1.0
    V5[flat_V][c] = 1.0
    W5[flat_W][c] = 1.0
    
    return U5, V5, W5

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    try:
        # 4x5x5 = 4x4x4 (top-left) + 4x1x5 (right strip)
        # The 4x4x4 part uses rank 49 (from Strassen)
        # The 4x1x5 part uses rank 5 (one term per k in 0..4)
        
        total_rank = 49 + 5  # = 54
        
        U = [[0.0]*total_rank for _ in range(n*m)]  # 20 x 54
        V = [[0.0]*total_rank for _ in range(m*p)]  # 25 x 54
        W = [[0.0]*total_rank for _ in range(p*n)]  # 20 x 54
        
        # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3), rank 49
        U4, V4, W4 = _build_4x4x4_strassen()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    flat_U = i*5 + j
                    flat_V = j*5 + k
                    flat_W = k*4 + i
                    for c in range(49):
                        U[flat_U][c] += U4[i*4+j][c]
                        V[flat_V][c] += V4[j*4+k][c]
                        W[flat_W][c] += W4[k*4+i][c]
        
        # Part 2: 4x1x5 (i in 0..3, j=4, k in 0..4), rank 5
        # For each k in 0..4, we need a separate term
        for k in range(5):
            c = 49 + k
            flat_U = 4*5 + k  # This is wrong, let me fix
            # Actually: for the (i, j=4, k) triple, we need:
            # U[i*5+4][c] * V[4*5+k][c] * W[k*4+i][c] = 1 for all i
            # and 0 for other combinations
            # So for each k, we set:
            #   U[i*5+4][c] = 1 for all i in 0..3
            #   V[4*5+k][c] = 1
            #   W[k*4+i][c] = 1 for all i in 0..3
            # But wait, this would make the term contribute to ALL (i,4,k') for all k'
            # because W[k*4+i][c] is only nonzero for the specific k.
            # Let me think again.
            # The term c contributes to the (i,j,k) entry if U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] != 0
            # For our term: U[i*5+4][c]=1 (for i in 0..3), V[4*5+k0][c]=1 (for specific k0), W[k0*4+i][c]=1 (for i in 0..3)
            # So this term only contributes to entries where j=4, k=k0, and any i in 0..3
            # That's exactly what we want!
            for i in range(4):
                flat_U = i*5 + 4
                flat_V = 4*5 + k
                flat_W = k*4 + i
                U[flat_U][c] = 1.0
                V[flat_V][c] = 1.0
                W[flat_W][c] = 1.0
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_fb = [[0] * rank_school for _ in range(n * m)]
            V_fb = [[0] * rank_school for _ in range(m * p)]
            W_fb = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col] = 1
                        V_fb[j*p+k][col] = 1
                        W_fb[k*n+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        # Fall back to schoolbook
        rank_school = n * m * p
        U_fb = [[0] * rank_school for _ in range(n * m)]
        V_fb = [[0] * rank_school for _ in range(m * p)]
        W_fb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
