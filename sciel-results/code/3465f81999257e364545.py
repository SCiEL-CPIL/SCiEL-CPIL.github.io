"""Program for matrix multiplication tensor decomposition 3x3x8."""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x3x8 matrix multiplication.
    
    Strategy:
    - Split the 8 columns of B into 4 blocks of 2 columns each.
    - For each 3x3x2 block, use a rank-17 decomposition:
      * Strassen's rank-7 algorithm for the 2x2x2 subproblem (top-left 2x2 of A, first 2 cols of B block)
      * Schoolbook rank-10 for the remaining terms (involving row 2 of A or col 2 of the 3x2 B block)
    - Total rank: 4 * 17 = 68
    
    This beats the schoolbook rank of 72.
    
    Index conventions:
    - U: shape (n*m, rank) = (9, rank), indexed by i*m + j where i in [0,n), j in [m)
    - V: shape (m*p, rank) = (24, rank), indexed by j*p + k where j in [0,m), k in [0,p)
    - W: shape (p*n, rank) = (24, rank), indexed by k*n + i where k in [0,p), i in [0,n)
    
    The output C[i,k] = sum_j A[i,j] B[j,k]
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll build U, V, W as lists of lists
    # First, determine the total rank
    # Each 3x3x2 block uses rank 17
    # 4 blocks -> total rank 68
    total_rank = 68
    
    U = [[0] * total_rank for _ in range(n * m)]  # (9, 68)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 68)
    W = [[0] * total_rank for _ in range(p * n)]  # (24, 68)
    
    col_idx = 0
    
    for block in range(4):
        # Each block handles columns 2*block and 2*block+1 of B
        k0 = 2 * block      # first column index in this block
        k1 = 2 * block + 1  # second column index in this block
        
        # Strassen 2x2x2 decomposition for A[0:2, 0:2] * B[0:2, k0:k0+2]
        # This computes C[0:2, k0] and C[0:2, k1]
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)(B10)
        # M6 = (A01-A00)(B00+B10)
        # M7 = (A11-A10)(B01+B11)
        
        # Result:
        # C00 = M1 - M5 + M7
        # C01 = M3 + M6 + M7
        # C10 = M2 + M4 + M5
        # C11 = M1 - M2 + M4 - M6
        
        # We'll add each Strassen term as a rank-1 component with appropriate coefficients
        
        # Helper to add a term with coefficient
        def add_strassen_term(u_coeff, v_coeff, w_coeff):
            nonlocal col_idx
            # u_coeff: dict mapping (i,j) -> coeff for A[i,j]
            # v_coeff: dict mapping (j,k) -> coeff for B[j,k]
            # w_coeff: dict mapping (i,k) -> coeff for C[i,k]
            
            for (i, j), c_u in u_coeff.items():
                if c_u != 0:
                    U[i * m + j][col_idx] += c_u
            for (j, k), c_v in v_coeff.items():
                if c_v != 0:
                    V[j * p + k][col_idx] += c_v
            for (i, k), c_w in w_coeff.items():
                if c_w != 0:
                    W[k * n + i][col_idx] += c_w
            col_idx += 1
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to C00 (+1) and C11 (+1)
        add_strassen_term(
            u_coeff={(0,0): 1, (1,1): 1},
            v_coeff={(0, k0): 1, (1, k0): 1},
            w_coeff={(0, k0): 1, (1, k1): 1}
        )
        
        # M2 = (A10+A11)(B00)
        # Contributes to C10 (+1) and C11 (-1)
        add_strassen_term(
            u_coeff={(1,0): 1, (1,1): 1},
            v_coeff={(0, k0): 1},
            w_coeff={(1, k0): 1, (1, k1): -1}
        )
        
        # M3 = A00(B01-B11)
        # Contributes to C01 (+1)
        add_strassen_term(
            u_coeff={(0,0): 1},
            v_coeff={(0, k1): 1, (1, k1): -1},
            w_coeff={(0, k1): 1}
        )
        
        # M4 = A11(B10-B00)
        # Contributes to C10 (+1) and C11 (-1)
        add_strassen_term(
            u_coeff={(1,1): 1},
            v_coeff={(1, k0): 1, (0, k0): -1},
            w_coeff={(1, k0): 1, (1, k1): -1}
        )
        
        # M5 = (A00+A10)(B10)
        # Contributes to C00 (-1) and C10 (+1)
        add_strassen_term(
            u_coeff={(0,0): 1, (1,0): 1},
            v_coeff={(1, k0): 1},
            w_coeff={(0, k0): -1, (1, k0): 1}
        )
        
        # M6 = (A01-A00)(B00+B10)
        # Contributes to C01 (+1) and C11 (-1)
        add_strassen_term(
            u_coeff={(0,1): 1, (0,0): -1},
            v_coeff={(0, k0): 1, (1, k0): 1},
            w_coeff={(0, k1): 1, (1, k1): -1}
        )
        
        # M7 = (A11-A10)(B01+B11)
        # Contributes to C01 (+1) and C00 (+1)
        add_strassen_term(
            u_coeff={(1,1): 1, (1,0): -1},
            v_coeff={(0, k1): 1, (1, k1): 1},
            w_coeff={(0, k1): 1, (0, k0): 1}
        )
        
        # Now add schoolbook terms for the remaining parts:
        # Terms involving A[2, :] or B[:, k1] or B[:, k0] where the other index is 2
        
        # C[2, k0] = A[2,0]B[0,k0] + A[2,1]B[1,k0] + A[2,2]B[2,k0]
        # C[2, k1] = A[2,0]B[0,k1] + A[2,1]B[1,k1] + A[2,2]B[2,k1]
        # C[0, k1] already has contributions from Strassen, but we need to check if we're double counting
        # Wait, the Strassen part only uses A[0:2, 0:2] and B[0:2, k0:k0+2]
        # So terms with A[2,:] or B[2,:] are not covered by Strassen
        
        # Actually, let me reconsider. The Strassen part computes:
        # C[0:2, k0] and C[0:2, k1] using only A[0:2, 0:2] and B[0:2, k0:k0+2]
        # But the full computation requires:
        # C[i, k] = sum_{j=0}^{2} A[i,j] B[j,k]
        # So for C[0, k0], we need A[0,0]B[0,k0] + A[0,1]B[1,k0] + A[0,2]B[2,k0]
        # The Strassen part only gives A[0,0]B[0,k0] + A[0,1]B[1,k0] (partially, via the 2x2x2)
        # We still need to add A[0,2]B[2,k0], A[1,2]B[2,k0], etc.
        
        # Let me redo this more carefully.
        
        # The 2x2x2 Strassen handles the subproblem:
        # A_sub = A[0:2, 0:2] (2x2)
        # B_sub = B[0:2, k0:k0+2] (2x2)
        # C_sub = C[0:2, k0:k0+2] (2x2)
        # So it computes C_sub = A_sub * B_sub exactly.
        
        # The remaining terms are:
        # For C[i, k] where i in {0,1} and k in {k0,k1}:
        #   C[i,k] += A[i,2] * B[2,k]
        # For C[2, k] where k in {k0,k1}:
        #   C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
        
        # So we need:
        # 4 terms for A[i,2]*B[2,k] where i in {0,1}, k in {k0,k1}
        # 6 terms for C[2,k] computation
        # Total 10 additional terms.
        
        # Term 1: A[0,2]*B[2,k0] -> C[0,k0]
        U[0 * m + 2][col_idx] = 1
        V[2 * p + k0][col_idx] = 1
        W[k0 * n + 0][col_idx] = 1
        col_idx += 1
        
        # Term 2: A[0,2]*B[2,k1] -> C[0,k1]
        U[0 * m + 2][col_idx] = 1
        V[2 * p + k1][col_idx] = 1
        W[k1 * n + 0][col_idx] = 1
        col_idx += 1
        
        # Term 3: A[1,2]*B[2,k0] -> C[1,k0]
        U[1 * m + 2][col_idx] = 1
        V[2 * p + k0][col_idx] = 1
        W[k0 * n + 1][col_idx] = 1
        col_idx += 1
        
        # Term 4: A[1,2]*B[2,k1] -> C[1,k1]
        U[1 * m + 2][col_idx] = 1
        V[2 * p + k1][col_idx] = 1
        W[k1 * n + 1][col_idx] = 1
        col_idx += 1
        
        # Term 5: A[2,0]*B[0,k0] -> C[2,k0]
        U[2 * m + 0][col_idx] = 1
        V[0 * p + k0][col_idx] = 1
        W[k0 * n + 2][col_idx] = 1
        col_idx += 1
        
        # Term 6: A[2,1]*B[1,k0] -> C[2,k0]
        U[2 * m + 1][col_idx] = 1
        V[1 * p + k0][col_idx] = 1
        W[k0 * n + 2][col_idx] = 1
        col_idx += 1
        
        # Term 7: A[2,2]*B[2,k0] -> C[2,k0]
        U[2 * m + 2][col_idx] = 1
        V[2 * p + k0][col_idx] = 1
        W[k0 * n + 2][col_idx] = 1
        col_idx += 1
        
        # Term 8: A[2,0]*B[0,k1] -> C[2,k1]
        U[2 * m + 0][col_idx] = 1
        V[0 * p + k1][col_idx] = 1
        W[k1 * n + 2][col_idx] = 1
        col_idx += 1
        
        # Term 9: A[2,1]*B[1,k1] -> C[2,k1]
        U[2 * m + 1][col_idx] = 1
        V[1 * p + k1][col_idx] = 1
        W[k1 * n + 2][col_idx] = 1
        col_idx += 1
        
        # Term 10: A[2,2]*B[2,k1] -> C[2,k1]
        U[2 * m + 2][col_idx] = 1
        V[2 * p + k1][col_idx] = 1
        W[k1 * n + 2][col_idx] = 1
        col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
