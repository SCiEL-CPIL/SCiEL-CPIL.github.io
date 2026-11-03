import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Returns the 7 Strassen terms for 2x2 * 2x2.
    Each term is a tuple (U_vec, V_vec, W_vec) where vectors are length 4.
    Indices: 0:00, 1:01, 2:10, 3:11 for U (A) and V (B).
    W indices: 0:00, 1:01, 2:10, 3:11 for C.
    """
    terms = []
    # P1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1] # C00+=P1, C11+=P1? No.
    # C00 = P1 - P4 + P5 + P7? No.
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P2 + P5 - P6 - P7
    # C21 = P1 + P3 + P5 - P4
    # C22 = P3 + P6 + P4 - P1
    
    # Using 0-based local indices:
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    # C00, C01, C10, C11
    
    # P1
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0])) # C00 += P1
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1])) # C11 += P1 (Wait, C22 in 1-based is C11 in 0-based? No. C22 is bottom-right. Index 3.)
    # Let's map carefully.
    # C00 (0,0): P1 + P4 - P5 + P7
    # C01 (0,1): P2 + P5 - P6 - P7
    # C10 (1,0): P1 + P3 + P5 - P4
    # C11 (1,1): P3 + P6 + P4 - P1
    
    # I will generate the full U, V, W matrices for 2x2x2 Strassen directly.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1: A00,A11; B00,B11
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=-1 # C00 += P1, C11 -= P1? No.
    # C00 = P1 + P4 - P5 + P7
    # C11 = P3 + P6 + P4 - P1
    # So W[0][0]=1, W[3][0]=-1.
    
    # P2: A10,A11; B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1 # C01 += P2
    
    # P3: A00; B01,B11
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1 # C10 += P3
    W[3][2]=1 # C11 += P3
    
    # P4: A11; B10,B00
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1 # C00 += P4
    W[2][3]=-1 # C10 -= P4
    W[3][3]=1 # C11 += P4
    
    # P5: A00,A01; B11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1 # C00 -= P5
    W[1][4]=1 # C01 += P5
    W[2][4]=1 # C10 += P5
    
    # P6: A10,A00; B00,B01
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[1][5]=-1 # C01 -= P6
    W[3][5]=1 # C11 += P6
    
    # P7: A01,A11; B10,B11
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1 # C00 += P7
    W[1][6]=-1 # C01 -= P7
    
    return U, V, W

def _build_4x4_4x4_strassen():
    """
    Builds 4x4 * 4x4 decomposition using Strassen (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Strassen 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block offsets:
    # A11: 0, A12: 4, A21: 8, A22: 12 (row-major 4x4: i*4+j)
    # B11: 0, B12: 4, B21: 8, B22: 12
    # C11: 0, C12: 4, C21: 8, C22: 12
    
    def add_block(U, V, W, u_off, v_off, w_off, term_start, sign_u=1, sign_v=1, sign_w=1):
        for c in range(7):
            for i in range(4):
                if U2[i][c] != 0:
                    U[u_off+i][term_start+c] += U2[i][c] * sign_u
            for j in range(4):
                if V2[j][c] != 0:
                    V[v_off+j][term_start+c] += V2[j][c] * sign_v
            for k in range(4):
                if W2[k][c] != 0:
                    W[w_off+k][term_start+c] += W2[k][c] * sign_w

    # M1: (A11+A22)(B11+B22)
    # This is a product of two sums. We can't just add two Strassen decompositions.
    # We must expand (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This requires 4 Strassen decompositions (rank 7 each) = 28 terms.
    # Wait, the rank of 4x4x4 Strassen is 49?
    # If we use 7 multiplications of 2x2x2, and each 2x2x2 is rank 7, total 49.
    # But M1 is ONE multiplication of two 2x2 matrices?
    # No, in the recursive Strassen, M1 is the product of two 2x2 matrices.
    # If we compute that product with Strassen (rank 7), then M1 contributes 7 terms.
    # So total 7 * 7 = 49 terms.
    
    # So for M1, we need to compute (A11+A22) * (B11+B22).
    # Let X = A11+A22, Y = B11+B22.
    # We apply Strassen to X*Y.
    # The U-vector for X is A11+A22.
    # The V-vector for Y is B11+B22.
    # We can construct the U, V, W for this specific product by combining the blocks.
    
    # Helper to add a Strassen term for a specific X, Y block combination.
    # X is formed by A blocks, Y by B blocks.
    # We pass the offsets and signs.
    
    def add_strassen_term(U, V, W, a_offsets, b_offsets, c_offset, term_start, signs_u, signs_v, sign_w):
        # a_offsets: list of (offset, sign) for A blocks
        # b_offsets: list of (offset, sign) for B blocks
        # c_offset: offset for C block
        for c in range(7):
            # U part
            for i in range(4):
                val = 0
                for off, s in a_offsets:
                    if U2[i][c] != 0:
                        val += U2[i][c] * s
                if val != 0:
                    U[c_offset + i][term_start + c] += val * sign_w # Wait, sign_w applies to W? No, W is output.
                    # Actually, the term is U * V * W.
                    # We are constructing U, V, W columns.
                    # U column is the linear combo of A.
                    # V column is the linear combo of B.
                    # W column is the linear combo of C.
                    # The Strassen formula gives C = W * (U*V)? No.
                    # C = sum_k U[:,k] V[:,k] W[:,k]? No.
                    # C[i,k] = sum_t U[i,t] V[j,t] W[k,t]? No.
                    # C = U V^T W? No.
                    # Tensor decomposition: C[i,k] = sum_t U[i,t] V[j,t] W[k,t] is wrong.
                    # C[i,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] is for the full tensor.
                    # Here we are building blocks.
                    # For a block product C_block = X Y.
                    # C_block[i,k] = sum_t X[i,t] Y[t,k].
                    # In Strassen, X Y = sum_s U_s V_s W_s?
                    # No, Strassen gives C = sum_s (u_s . X) (v_s . Y) w_s^T?
                    # Actually, the rank-7 decomposition of 2x2x2 is:
                    # C = sum_{s=1}^7 u_s (X) v_s (Y) w_s^T (C)?
                    # No, it's: C[i,k] = sum_s U[i,s] V[k,s] W[s]? No.
                    # The standard form is:
                    # C[i,k] = sum_s A_u[i,s] B_v[k,s] C_w[s]?
                    # Let's stick to the matrices U, V, W I generated in _strassen_2x2_2x2.
                    # In that function, U[i,c], V[j,c], W[k,c].
                    # The product is sum_c U[i,c] V[j,c] W[k,c]?
                    # No, that's not right either.
                    # The tensor is T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
                    # For matrix mult, T[i,j,k] = A[i,j] B[j,k] C[k,i]? No.
                    # T[i,j,k] = 1 if i,j,k match?
                    # The decomposition is:
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # Tensor T[i,j,k] = delta(i,j,k) for the multiplication?
                    # No, the tensor for matmult is T[i,j,k] = 1.
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # This is a 3-tensor T[i,j,k] where T[i,j,k]=1.
                    # Decomposition: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
                    # So U[i,c] corresponds to A[i,j]? No.
                    # U has shape (n*m, rank). Index i*m+j.
                    # V has shape (m*p, rank). Index j*p+k.
                    # W has shape (p*n, rank). Index k*n+i.
                    
                    # In my _strassen_2x2_2x2, I returned U (4,7), V (4,7), W (4,7).
                    # This implies I was treating it as a 2x2x2 tensor with indices 0..3.
                    # U[i,c] is coeff for A[i_local, j_local]?
                    # No, U index is i*m+j. For 2x2, i*2+j.
                    # My U in _strassen_2x2_2x2 was 4x7.
                    # So U[0,c] is A00, U[1,c] is A01, U[2,c] is A10, U[3,c] is A11.
                    # This matches i*2+j.
                    # So the decomposition is valid.
                    
            # This approach of summing is getting complex.
            # I will just return the schoolbook 4x4x4 (rank 64) and 4x4x3 (rank 48) if Strassen is too risky.
            # 64 + 48 = 112. No improvement.
            # I will implement the 49+14=63 Strassen carefully.
            pass

    # Given time, I will implement the 63 rank solution directly by hardcoding the 49+14 terms.
    # Or use the 56+28=84 solution which is easier to verify?
    # 84 is an improvement.
    
    # Let's use the 84 rank solution (Schoolbook inner Strassen).
    # It is safer.
    
    U, V, W = _build_4x4_4x4_schoolbook_inner_strassen()
    return U, V, W, 56

def _build_4x4_4x4_schoolbook_inner_strassen():
    # 7 Strassen steps, each 4 schoolbook terms (rank 4 for 2x2x2? No, rank 8).
    # Wait, schoolbook 2x2x2 is rank 8.
    # 7 * 8 = 56.
    pass

# Fallback to Schoolbook 112 to ensure correctness.
def construct():
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

# EVOLVE-BLOCK-END
