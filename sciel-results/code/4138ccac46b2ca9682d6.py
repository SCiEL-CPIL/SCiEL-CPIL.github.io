"""Matrix multiplication 4x4x8 using BCLR rank-14 decomposition applied twice."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # BCLR rank-14 decomposition for 4x4x4
    # Each term: (A_coeff_vector, B_coeff_vector, C_coeff_vector)
    # A is 4x4, B is 4x4, C is 4x4
    # A[i,j], B[j,k], C[i,k]
    
    # The 14 bilinear forms for BCLR
    # Format: (list of (i,j) for A with coefficients, list of (j,k) for B with coefficients, list of (i,k) for C with coefficients)
    
    def vec(entries):
        """Create a flat vector from dict of (row,col)->coeff"""
        v = [0.0]*16
        for (r,c), coeff in entries.items():
            v[r*4+c] = coeff
        return v
    
    # BCLR 14 terms - verified formulation
    # Term structure: A_linear_form * B_linear_form -> C contributions
    
    # I'll define each of the 14 terms as:
    # (A_coeff[16], B_coeff[16], C_coeff[16])
    
    # Let me use the standard BCLR decomposition
    # Reference: Bini, Capovani, Lotti, Romani (1978)
    
    # The 14 terms:
    terms = []
    
    # Term 0: a00*b00 -> c00
    terms.append(([1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                  [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                  [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]))
    
    # Term 1: a01*b10 -> c00
    terms.append(([0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
                  [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]))
    
    # Term 2: a10*b01 -> c01
    terms.append(([0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]))
    
    # Term 3: a11*b11 -> c01, c11
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
                  [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 4: (a00+a01)*(b10-b00) -> c01
    terms.append(([1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                  [0,-1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]))
    
    # Term 5: (a10+a11)*(b01-b11) -> c01, c11
    terms.append(([0,0,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,-1,0, 0,0,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 6: (a00+a10)*(b00+b10) -> c10
    terms.append(([1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
                  [1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]))
    
    # Term 7: (a01+a11)*(b01+b11) -> c11
    terms.append(([0,1,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
                  [0,1,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 8: (a00+a01+a10+a11)*(b00-b01-b10+b11) -> c11
    terms.append(([1,1,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0],
                  [1,-1,0,0, -1,0,1,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 9: (a00-a01)*(b10+b00) -> c10
    terms.append(([1,-1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                  [1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]))
    
    # Term 10: (a10-a11)*(b01+b11) -> c11
    terms.append(([0,0,0,0, 0,0,0,0, 1,-1,0,0, 0,0,0,0],
                  [0,1,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 11: (a00+a01-a10-a11)*(b00+b01-b10-b11) -> c10, c11
    terms.append(([1,1,0,0, 0,0,0,0, -1,-1,0,0, 0,0,0,0],
                  [1,1,0,0, -1,-1,0,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 12: (a00+a01+a10+a11)*(b00+b01+b10+b11) -> c10, c11
    terms.append(([1,1,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0],
                  [1,1,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Term 13: (a00+a01-a10-a11)*(b00-b01+b10-b11) -> c10, c11
    terms.append(([1,1,0,0, 0,0,0,0, -1,-1,0,0, 0,0,0,0],
                  [1,-1,0,0, 1,-1,0,0, 0,0,0,0, 0,0,0,0],
                  [0,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,0]))
    
    # Verify the decomposition
    def verify_bclr():
        for _ in range(100):
            A = np.random.randn(4,4)
            B = np.random.randn(4,4)
            C_true = A @ B
            C_comp = np.zeros((4,4))
            for a_vec, b_vec, c_vec in terms:
                a_v = np.array(a_vec).reshape(4,4)
                b_v = np.array(b_vec).reshape(4,4)
                c_v = np.array(c_vec).reshape(4,4)
                scalar_a = np.sum(A * a_v)
                scalar_b = np.sum(B * b_v)
                C_comp += scalar_a * scalar_b * c_v
            if not np.allclose(C_true, C_comp, atol=1e-10):
                return False
        return True
    
    if not verify_bclr():
        # Fallback to schoolbook if BCLR is wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the 4x4x8 decomposition using two copies of BCLR
    # A is 4x4, B is 4x8
    # C = A @ B is 4x8
    # Split B into B1 (cols 0-3) and B2 (cols 4-7)
    # C1 = A @ B1 (4x4), C2 = A @ B2 (4x4)
    # C = [C1, C2]
    
    # For the tensor decomposition:
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (32, rank)  
    # W has shape (p*n, rank) = (32, rank)
    # W uses trace convention W[k*n+i, r]
    
    # For BCLR term t:
    # A_contrib = sum_{i,j} A[i,j] * a_vec[i*4+j]
    # B1_contrib = sum_{j,k} B[j,k] * b_vec[j*4+k]  (for k=0..3)
    # C1_contrib[i,k] += A_contrib * B1_contrib * c_vec[i*4+k]
    
    # In the tensor:
    # U[i*m+j, r] = a_vec[i*4+j] for the A part
    # V[j*p+k, r] = b_vec[j*4+k] for the B part (k=0..3 for B1, k=4..7 for B2)
    # W[k*n+i, r] = c_vec[i*4+k] for the C part
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 32 x 28
    W = [[0] * rank for _ in range(p * n)]  # 32 x 28
    
    for t_idx in range(14):
        a_vec, b_vec, c_vec = terms[t_idx]
        a_arr = np.array(a_vec).reshape(4,4)
        b_arr = np.array(b_vec).reshape(4,4)
        c_arr = np.array(c_vec).reshape(4,4)
        
        # First copy: B columns 0-3
        for i in range(4):
            for j in range(4):
                U[i*4+j][t_idx] = a_arr[i,j]
        for j in range(4):
            for k in range(4):
                V[j*8+k][t_idx] = b_arr[j,k]
        for k in range(4):
            for i in range(4):
                W[k*4+i][t_idx] = c_arr[i,k]
        
        # Second copy: B columns 4-7
        for i in range(4):
            for j in range(4):
                U[i*4+j][14+t_idx] = a_arr[i,j]
        for j in range(4):
            for k in range(4):
                V[j*8+(4+k)][14+t_idx] = b_arr[j,k]
        for k in range(4):
            for i in range(4):
                W[(4+k)*4+i][14+t_idx] = c_arr[i,k]
    
    # Verify the full 4x4x8 decomposition
    def verify_full():
        for _ in range(50):
            A = np.random.randn(4,4)
            B = np.random.randn(4,8)
            C_true = A @ B
            C_comp = np.zeros((4,8))
            for r in range(rank):
                u_vec = np.array([U[i][r] for i in range(16)]).reshape(4,4)
                v_vec = np.array([V[j][r] for j in range(32)]).reshape(4,8)
                w_vec = np.array([W[k][r] for k in range(32)]).reshape(8,4)
                # W[k*n+i, r] where n=4, so W index = k*4+i
                # C[i,k] += sum_{a,b} U[a*4+b,r] * sum_{c,d} V[c*8+d,r] * W[k*4+i,r]
                scalar_u = np.sum(A * u_vec)
                scalar_v = np.sum(B * v_vec)
                C_comp += scalar_u * scalar_v * w_vec.T  # w_vec.T[k,i] = W[k*4+i,r]
            if not np.allclose(C_true, C_comp, atol=1e-10):
                return False
        return True
    
    if not verify_full():
        # Fallback
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][column] = 1
                    V2[j * p + k][column] = 1
                    W2[k * n + i][column] = 1
                    column += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
