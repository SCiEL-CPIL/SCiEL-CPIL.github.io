"""
Matrix multiplication 3x3 * 3x5 tensor decomposition.
Uses structured Strassen-like decomposition for low-rank exact solutions.
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Helper to add Strassen terms for a 2x2x2 block
    # i_range, j_range, k_range are the local indices within the block
    # global_offset_k is the offset for the global k index
    def add_strassen_block(i_start, j_start, k_start, U, V, W, current_col):
        # Local indices 0,1 map to global i_start, i_start+1 etc.
        # Strassen formulas for 2x2 matrices
        # A indices: (i,j) -> a = i*m + j
        # B indices: (j,k) -> b = j*p + k
        # C indices: (k,i) -> c = k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # (coeff_U, coeff_V, coeff_W) for each M_r
            # U coeffs: (i,j) weights
            # V coeffs: (j,k) weights
            # W coeffs: (k,i) weights
            
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):1}), # Wait, M2 contributes to C10 and C11
            # Let's just store the U, V, W vectors directly for each of the 7 products
            # and then combine them for C.
        ]
        
        # It is easier to define the 7 rank-1 tensors for the products M1..M7
        # and then define the linear combinations for C.
        # But the decomposition is a sum of rank-1 tensors U_r V_r W_r.
        # Strassen gives:
        # C = sum_{r=1}^7 M_r * (something)
        # Actually, C_{kl} = sum_{r} w_{kl,r} M_r
        # M_r = (sum_{ij} u_{ij,r} A_{ij}) * (sum_{jk} v_{jk,r} B_{jk})
        # So the total tensor is sum_{r} (sum_{ij} u_{ij,r} e_{ij}) ⊗ (sum_{jk} v_{jk,r} e_{jk}) ⊗ (sum_{kl} w_{kl,r} e_{kl})?
        # No, M_r is a scalar.
        # C = sum_r M_r * W_r
        # where W_r is a matrix with entries w_{kl,r}.
        # So the rank-1 term is (U_r ⊗ V_r) ⊗ W_r?
        # No, the tensor is T = sum_r U_r ⊗ V_r ⊗ W_r.
        # Here U_r is the vector of coefficients for A, V_r for B, W_r for C.
        
        # For M1:
        # U1: A00, A11 have coeff 1.
        # V1: B00, B11 have coeff 1.
        # W1: C00 has coeff 1, C11 has coeff 1.
        
        # Let's build the 7 terms.
        
        # Define local to global index mapping
        def idx_A(i, j): return (i_start + i) * m + (j_start + j)
        def idx_B(j, k): return (j_start + j) * p + (k_start + k)
        def idx_C(k, i): return (k_start + k) * n + (i_start + i)
        
        U_block = np.zeros((n*m, 7))
        V_block = np.zeros((m*p, 7))
        W_block = np.zeros((p*n, 7))
        
        # M1
        U_block[idx_A(0,0), 0] = 1; U_block[idx_A(1,1), 0] = 1
        V_block[idx_B(0,0), 0] = 1; V_block[idx_B(1,1), 0] = 1
        W_block[idx_C(0,0), 0] = 1; W_block[idx_C(1,1), 0] = 1
        
        # M2
        U_block[idx_A(1,0), 1] = 1; U_block[idx_A(1,1), 1] = 1
        V_block[idx_B(0,0), 1] = 1
        W_block[idx_C(0,1), 1] = 1; W_block[idx_C(1,1), 1] = -1 # C10 = M2..., C11 = -M2...
        # Wait, C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So W2 should have 1 at C10 and -1 at C11.
        
        # M3
        U_block[idx_A(0,0), 2] = 1
        V_block[idx_B(1,0), 2] = 1; V_block[idx_B(1,1), 2] = -1
        W_block[idx_C(0,1), 2] = 1; W_block[idx_C(1,1), 2] = 1 # C01 = M3+M5, C11 = ...+M3
        
        # M4
        U_block[idx_A(1,1), 3] = 1
        V_block[idx_B(0,1), 3] = 1; V_block[idx_B(0,0), 3] = -1
        W_block[idx_C(0,0), 3] = 1; W_block[idx_C(0,1), 3] = 1 # C00 = ...+M4, C10 = ...+M4
        
        # M5
        U_block[idx_A(0,0), 4] = 1; U_block[idx_A(0,1), 4] = 1
        V_block[idx_B(1,1), 4] = 1
        W_block[idx_C(0,0), 4] = -1; W_block[idx_C(0,1), 4] = 1 # C00 = ...-M5, C01 = ...+M5
        
        # M6
        U_block[idx_A(1,1), 5] = 1; U_block[idx_A(0,0), 5] = -1
        V_block[idx_B(0,0), 5] = 1; V_block[idx_B(1,0), 5] = 1
        W_block[idx_C(1,1), 5] = 1 # C11 = ...+M6
        
        # M7
        U_block[idx_A(0,1), 6] = 1; U_block[idx_A(1,1), 6] = -1
        V_block[idx_B(1,1), 6] = 1; V_block[idx_B(0,1), 6] = 1
        W_block[idx_C(0,0), 6] = 1 # C00 = ...+M7
        
        # Append to global matrices
        U[:, current_col:current_col+7] = U_block
        V[:, current_col:current_col+7] = V_block
        W[:, current_col:current_col+7] = W_block
        
        return current_col + 7

    # Initialize with schoolbook to identify terms to replace
    # We will construct the final U, V, W directly.
    # Rank 43 = 26 (from 3x3x3 part) + 17 (from 3x3x2 part)
    # 3x3x3 part: k in {0,1,2}. Strassen on k in {0,1}. Schoolbook on rest.
    # 3x3x2 part: k in {3,4}. Strassen on k in {3,4}. Schoolbook on rest (none, since it's exactly 2).
    # Wait, 3x3x2 part has k in {0,1} locally, which maps to global {3,4}.
    # So we apply Strassen to global k in {3,4} as well.
    
    # Total rank:
    # Part 1 (k=0,1,2):
    #   Strassen block (i=0,1; j=0,1; k=0,1): 7 terms.
    #   Schoolbook for remaining k=0,1,2 terms not in block:
    #     Total terms in 3x3x3: 27.
    #     Block terms: 8.
    #     Remaining: 19.
    #   Subtotal: 7 + 19 = 26.
    # Part 2 (k=3,4):
    #   Strassen block (i=0,1; j=0,1; k=3,4): 7 terms.
    #   Schoolbook for remaining k=3,4 terms not in block:
    #     Total terms in 3x3x2: 18.
    #     Block terms: 8.
    #     Remaining: 10.
    #   Subtotal: 7 + 10 = 17.
    # Total Rank: 43.
    
    total_rank = 43
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Part 1: k in {0,1,2}
    # Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, U, V, W, col)
    
    # Schoolbook for remaining terms in k in {0,1,2}
    # Terms where i not in {0,1} OR j not in {0,1} OR k not in {0,1}
    # But we already handled i in {0,1}, j in {0,1}, k in {0,1}.
    # So we need terms where:
    # (i in {0,1} AND j in {0,1} AND k in {0,1}) is EXCLUDED.
    # All other (i,j,k) with k in {0,1,2}.
    
    for i in range(n):
        for j in range(m):
            for k in range(3): # k=0,1,2
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Part 2: k in {3,4}
    # Strassen block for i in {0,1}, j in {0,1}, k in {3,4}
    # Local k in {0,1} maps to global k in {3,4}
    col = add_strassen_block(0, 0, 3, U, V, W, col)
    
    # Schoolbook for remaining terms in k in {3,4}
    for i in range(n):
        for j in range(m):
            for k in range(3, 5): # k=3,4
                # Local k' = k - 3
                if i in [0,1] and j in [0,1] and (k-3) in [0,1]:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Verify
    if col != total_rank:
        # Adjust if necessary, but should be exact
        pass
        
    # Verify decomposition
    approx = np.zeros_like(T)
    for r in range(total_rank):
        approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    if not np.allclose(T, approx, atol=1e-10):
        # Fallback to schoolbook
        U_sch = np.zeros((n*m, 45))
        V_sch = np.zeros((m*p, 45))
        W_sch = np.zeros((p*n, 45))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j, col] = 1
                    V_sch[j*p+k, col] = 1
                    W_sch[k*n+i, col] = 1
                    col += 1
        return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
