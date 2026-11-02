"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t_start = time.time()
    deadline = t_start + 110  # safety margin before 120s
    
    def build_decomposition_from_terms(rank_terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) tuples."""
        rank = len(rank_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][r] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][r] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][r] = coeff
        
        return U, V, W, rank
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is equivalent to: the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # Efficient verification using random test matrices
        np.random.seed(12345)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_arr[:, r].reshape(n, m)
            v_vec = V_arr[:, r].reshape(m, p)
            w_vec = W_arr[:, r].reshape(p, n)
            # C[i,k] += sum_j u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w_vec[k, i] * np.dot(u_vec[i, :], v_vec[:, k])
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        
        # Additional check with different seed
        np.random.seed(67890)
        A_test2 = np.random.randn(n, m)
        B_test2 = np.random.randn(m, p)
        C_expected2 = A_test2 @ B_test2
        C_computed2 = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_arr[:, r].reshape(n, m)
            v_vec = V_arr[:, r].reshape(m, p)
            w_vec = W_arr[:, r].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    C_computed2[i, k] += w_vec[k, i] * np.dot(u_vec[i, :], v_vec[:, k])
        
        return np.allclose(C_computed2, C_expected2, atol=1e-10)
    
    # Part 1: 4x4 * 4x4 for columns k=0,1,2,3 using Strassen on 2x2 blocks
    rank_terms = []
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):  # k1 = 0,1 for column blocks 0-1, 2-3
                k_base = k1 * 2  # columns k_base, k_base+1
                
                # Strassen's 7 products for 2x2 x 2x2 -> 2x2
                # Block indices: row block i1, col block j1 of A; row block j1, col block k1 of B
                # A block: rows i1*2..i1*2+1, cols j1*2..j1*2+1
                # B block: rows j1*2..j1*2+1, cols k_base..k_base+1
                # C block: rows i1*2..i1*2+1, cols k_base..k_base+1
                
                # M1 = (A[i1,0,0]+A[i1,1,1]) * (B[j1,0,0]+B[j1,1,1])
                # In block terms: (A00+A11)*(B00+B11) where subscripts are within the 2x2 block
                terms_strassen = [
                    # M1: (X00+X11)(Y00+Y11)
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k_base): 1, (j1*2+1, k_base+1): 1},
                     {(i1*2, k_base): 1, (i1*2+1, k_base+1): 1}),
                    # M2: (X10+X11)Y00
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k_base): 1},
                     {(i1*2+1, k_base): 1, (i1*2+1, k_base+1): -1}),
                    # M3: X00(Y01-Y11)
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k_base+1): 1, (j1*2+1, k_base+1): -1},
                     {(i1*2, k_base+1): 1, (i1*2+1, k_base+1): 1}),
                    # M4: X11(Y10-Y00)
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k_base): 1, (j1*2, k_base): -1},
                     {(i1*2, k_base): 1, (i1*2+1, k_base): 1}),
                    # M5: (X00+X01)Y11
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k_base+1): 1},
                     {(i1*2, k_base): -1, (i1*2, k_base+1): 1}),
                    # M6: (X10-X00)(Y00+Y01)
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k_base): 1, (j1*2, k_base+1): 1},
                     {(i1*2+1, k_base+1): 1}),
                    # M7: (X01-X11)(Y10+Y11)
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k_base): 1, (j1*2+1, k_base+1): 1},
                     {(i1*2, k_base): 1}),
                ]
                
                for u_dict, v_dict, w_dict in terms_strassen:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    # Part 2: 4x4 * 4x2 for columns k=4,5 using schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                rank_terms.append(
                    ({(i, j): 1},
                     {(j, k): 1},
                     {(i, k): 1})
                )
    
    U, V, W, rank = build_decomposition_from_terms(rank_terms)
    
    # Verify
    if verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
