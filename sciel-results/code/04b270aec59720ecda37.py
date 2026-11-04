import time
import random

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_terms():
    """
    Returns the 7 terms for 2x2 matrix multiplication using Strassen.
    Each term is a list of (U_idx, V_idx, W_idx, coeff) tuples.
    Indices are local 0..3 for 2x2 matrices.
    U: (i, j) -> i*2+j
    V: (j, k) -> j*2+k
    W: (k, i) -> k*2+i
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output: C00 = M1 - M4 + M5 + M7
    #         C01 = M3 + M5
    #         C10 = M2 + M4
    #         C11 = M1 - M3 + M2 + M6
    
    # For each Strassen term, we need to expand into rank-1 contributions.
    # Each term t has U_t[i,j] * V_t[j,k] * W_t[k,i] = coeff * M_t
    # where M_t is one of the 7 Strassen products.
    
    # Let's represent each term as a 2x2 U matrix, 2x2 V matrix, and 2x2 W matrix,
    # such that sum_t U_t[i,j] V_t[j,k] W_t[k,i] = C[i,k]
    
    # Actually, the standard way:
    # For term t, we have:
    #   A'_t[i,j] = U_t[i,j]
    #   B'_t[j,k] = V_t[j,k]  
    #   C'_t[i,k] = W_t[k,i]  (note the index swap for W)
    # And the product A'_t @ B'_t should equal W_t^T (transposed)
    # Wait, let me be more careful.
    
    # The tensor decomposition: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # For 2x2x2, T[i,j,k] = delta_{i,k} (identity), so we need:
    # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta_{i,k}
    
    # Strassen gives us 7 terms where:
    # M_t = A'_t @ B'_t, and C = combination of M_t's
    # But we need rank-1 tensors, not matrix products.
    
    # Actually, each Strassen term M_t = (linear combo of A) * (linear combo of B)
    # is NOT a rank-1 tensor in general. It's a product of two linear forms.
    # To get rank-1 tensor terms, we need to further decompose each M_t.
    
    # Wait, I think I'm overcomplicating this. Let me use the known explicit 
    # rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    
    # The 7 terms for Strassen, expressed as rank-1 tensors:
    # Term t contributes: U_t[i,j] * V_t[j,k] * W_t[k,i]
    
    # Let me look up the standard Strassen rank-1 decomposition.
    
    # Actually, Strassen's algorithm computes C = A@B using 7 scalar multiplications.
    # Each scalar multiplication m_t = (sum_{i,j} a_t[i,j] * A[i,j]) * (sum_{j,k} b_t[j,k] * B[j,k])
    # And then C[i,k] = sum_t c_t[i,k] * m_t
    
    # So the rank-1 tensor for term t is:
    # U_t[i,j] = a_t[i,j]
    # V_t[j,k] = b_t[j,k]
    # W_t[k,i] = c_t[i,k]
    
    # where a_t, b_t are 2x2 matrices and c_t is 2x2 matrix.
    
    # Strassen's 7 terms:
    # M1 = (A00+A11)(B00+B11): a1=[1,0,0,1], b1=[1,0,0,1]
    # M2 = (A10+A11)B00:       a2=[0,0,1,1], b2=[1,0,0,0]
    # M3 = A00(B01-B11):       a3=[1,0,0,0], b3=[0,1,0,-1]
    # M4 = A11(B10-B00):       a4=[0,0,0,1], b4=[0,0,1,-1]  # wait, B10-B00 means j=1,k=0 minus j=0,k=0
    # Let me redo with proper indices.
    # A[i,j], B[j,k], C[i,k]
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    # C00=C[0,0], C01=C[0,1], C10=C[1,0], C11=C[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    #   a1[i,j]: a1[0,0]=1, a1[1,1]=1, rest 0
    #   b1[j,k]: b1[0,0]=1, b1[1,1]=1, rest 0
    
    # M2 = (A10+A11)B00
    #   a2[i,j]: a2[1,0]=1, a2[1,1]=1, rest 0
    #   b2[j,k]: b2[0,0]=1, rest 0
    
    # M3 = A00(B01-B11)
    #   a3[i,j]: a3[0,0]=1, rest 0
    #   b3[j,k]: b3[0,1]=1, b3[1,1]=-1, rest 0
    
    # M4 = A11(B10-B00)
    #   a4[i,j]: a4[1,1]=1, rest 0
    #   b4[j,k]: b4[1,0]=1, b4[0,0]=-1, rest 0
    
    # M5 = (A00+A01)B11
    #   a5[i,j]: a5[0,0]=1, a5[0,1]=1, rest 0
    #   b5[j,k]: b5[1,1]=1, rest 0
    
    # M6 = (A10-A00)(B00+B01)
    #   a6[i,j]: a6[1,0]=1, a6[0,0]=-1, rest 0
    #   b6[j,k]: b6[0,0]=1, b6[0,1]=1, rest 0
    
    # M7 = (A01-A11)(B10+B11)
    #   a7[i,j]: a7[0,1]=1, a7[1,1]=-1, rest 0
    #   b7[j,k]: b7[1,0]=1, b7[1,1]=1, rest 0
    
    # Now C = combination:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M6
    
    # So for each term t, c_t[i,k] is the coefficient of M_t in C[i,k]:
    # c1[0,0]=1, c1[1,1]=1, rest 0
    # c2[1,0]=1, rest 0
    # c3[0,1]=1, c3[1,1]=-1, rest 0
    # c4[0,0]=-1, c4[1,0]=1, rest 0
    # c5[0,0]=1, c5[0,1]=1, rest 0
    # c6[1,1]=1, rest 0
    # c7[0,0]=1, c7[1,1]=1, rest 0
    
    terms = []
    # Each term: (a_matrix, b_matrix, c_matrix) as flat lists [a00,a01,a10,a11]
    a1 = [1,0,0,1]; b1 = [1,0,0,1]; c1 = [1,0,0,1]
    a2 = [0,0,1,1]; b2 = [1,0,0,0]; c2 = [0,0,1,0]
    a3 = [1,0,0,0]; b3 = [0,1,0,-1]; c3 = [0,1,0,-1]
    a4 = [0,0,0,1]; b4 = [-1,0,1,0]; c4 = [-1,0,1,0]
    a5 = [1,1,0,0]; b5 = [0,0,0,1]; c5 = [1,1,0,0]
    a6 = [-1,0,1,0]; b6 = [1,1,0,0]; c6 = [0,0,0,1]
    a7 = [0,1,0,-1]; b7 = [0,0,1,1]; c7 = [1,0,0,1]
    
    terms = [(a1,b1,c1),(a2,b2,c2),(a3,b3,c3),(a4,b4,c4),(a5,b5,c5),(a6,b6,c6),(a7,b7,c7)]
    return terms

def _embed_2x2_to_4x4(term, base_i, base_j, base_k, n=4):
    """
    Embed a 2x2x2 term into a 4x4x4 term at position (base_i, base_j, base_k).
    Returns (U, V, W) as dicts or sparse representations.
    """
    a, b, c = term
    U = {}
    V = {}
    W = {}
    for li in range(2):
        for lj in range(2):
            i = base_i + li
            j = base_j + lj
            if a[li*2+lj] != 0:
                U[(i, j)] = a[li*2+lj]
    for lj in range(2):
        for lk in range(2):
            j = base_j + lj
            k = base_k + lk
            if b[lj*2+lk] != 0:
                V[(j, k)] = b[lj*2+lk]
    for lk in range(2):
        for li in range(2):
            k = base_k + lk
            i = base_i + li
            if c[li*2+lk] != 0:
                W[(k, i)] = c[li*2+lk]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Strategy: Use Strassen for the 4x4x4 part (split into four 2x2x2 subproblems)
    # and schoolbook for the 4x4x3 part.
    # 
    # 4x4x4: Split A into 2x2 blocks, B into 2x2 blocks.
    # C = A@B where A is 4x4, B is 4x4 (first 4 columns of the 4x7 B)
    # Use 4 independent 2x2x2 Strassen decompositions: total rank 4*7 = 28
    # 
    # 4x4x3: Schoolbook: rank 4*4*3 = 48
    # 
    # Total: 28 + 48 = 76
    
    # Actually, we can do better. Let's use Strassen for the entire 4x4x7.
    # Split into 2x2 blocks in the row dimension, and handle columns.
    # 
    # Better approach: Use the known rank-23 decomposition for 4x4x4 if available,
    # but let's stick with Strassen which is well-understood.
    #
    # Let's use:
    # - For the first 4 columns of B (making it 4x4x4): use 4 independent 2x2x2 Strassen terms = 28
    # - For the last 3 columns of B (making it 4x4x3): use schoolbook = 48
    # Total: 76
    
    total_rank = 76
    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 28 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 28 rows
    
    col = 0
    
    # Part 1: 4x4x4 using Strassen (28 terms)
    # Split into 2x2 blocks:
    # Block (0,0): rows 0-1, cols 0-1
    # Block (0,1): rows 0-1, cols 2-3
    # Block (1,0): rows 2-3, cols 0-1
    # Block (1,1): rows 2-3, cols 2-3
    
    strassen_terms = _strassen_2x2_terms()
    
    for br in range(2):  # block row
        for bc in range(2):  # block col
            base_i = br * 2
            base_j = bc * 2
            base_k = bc * 2  # For 4x4x4, k dimension matches j dimension
            
            for term in strassen_terms:
                a, b, c = term
                for li in range(2):
                    for lj in range(2):
                        i = base_i + li
                        j = base_j + lj
                        if a[li*2+lj] != 0:
                            U[i*m + j][col] += a[li*2+lj]
                
                for lj in range(2):
                    for lk in range(2):
                        j = base_j + lj
                        k = base_k + lk
                        if b[lj*2+lk] != 0:
                            V[j*p + k][col] += b[lj*2+lk]
                
                for lk in range(2):
                    for li in range(2):
                        k = base_k + lk
                        i = base_i + li
                        if c[li*2+lk] != 0:
                            W[k*n + i][col] += c[li*2+lk]
                
                col += 1
    
    # Part 2: 4x4x3 using schoolbook (48 terms)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # columns 4, 5, 6
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify the decomposition with a random test
    if time.time() < deadline - 20:
        import numpy as np
        rng = np.random.default_rng(12345)
        
        # Test with several random matrices
        for test_iter in range(3):
            A = rng.integers(-5, 5, size=(n, m)).astype(float)
            B = rng.integers(-5, 5, size=(m, p)).astype(float)
            C_expected = A @ B
            
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for j in range(m):
                        total += np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    C_computed[i, k] = total
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # Fallback to schoolbook
                rank = n * m * p
                U = [[0.0] * rank for _ in range(n * m)]
                V = [[0.0] * rank for _ in range(m * p)]
                W = [[0.0] * rank for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m + j][col] = 1
                            V[j*p + k][col] = 1
                            W[k*n + i][col] = 1
                            col += 1
                return {"U": U, "V": V, "W": W}
            
            if time.time() > deadline - 30:
                break
    
    return {"U": U, "V": V, "W": W}
