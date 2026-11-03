import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Block decomposition:
    # A is 4x5, B is 5x5
    # Split A into A1 (4x4) and a2 (4x1)
    # Split B into B1 (4x5) and b2 (1x5)
    # AB = A1*B1 + a2*b2
    # B1 = B1a (4x4) + b1b (4x1)
    # AB = A1*B1a + A1*b1b + a2*b2
    
    # Part 1: A1*B1a (4x4 * 4x4) using Strassen with schoolbook 2x2: rank 56
    # Part 2: A1*b1b (4x4 * 4x1) schoolbook: rank 16
    # Part 3: a2*b2 (4x1 * 1x5) schoolbook: rank 20
    # Total: 92
    
    rank = 56 + 16 + 20
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: Strassen 4x4 with schoolbook 2x2
    # Strassen formulas for C = A*B where A,B are 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mi is a 2x2 product, done with schoolbook (4 multiplications)
    # Total: 7 * 4 = 28 rank for the 4x4x4 core... 
    # Wait, each 2x2 product has rank 4 (schoolbook 2x2x2)
    # So 7 * 4 = 28? No, schoolbook 2x2x2 has rank 4.
    # Actually 2x2 matrix multiplication tensor has rank 4 (schoolbook).
    # So 7 * 4 = 28 for Strassen 4x4.
    # Hmm, let me reconsider. The rank of 2x2x2 is 4 (schoolbook).
    # Strassen uses 7 such multiplications, so rank = 7*4 = 28.
    # But we need to account for the additions/subtractions in the U and V factors.
    
    # Let me implement this properly.
    # For each Strassen multiplication Mi = Xi * Yi where Xi, Yi are 2x2:
    # We use schoolbook: 4 terms
    # For term (i,j,k): U[x_i*m+x_j] += coeff, V[y_j*p+y_k] += coeff, W[z_k*n+z_i] += coeff
    
    # Let me build the Strassen decomposition explicitly.
    
    # 2x2 indices within each 2x2 block: (0,0), (0,1), (1,0), (1,1)
    # For a 4x4 matrix, block (r,c) covers rows 2r,2r+1 and cols 2c,2c+1
    # Element at global (i,j) in 4x4: block (i//2, j//2), local ((i%2), (j%2))
    
    # Strassen 4x4x4 decomposition:
    # We'll build U, V, W for the 4x4 core.
    
    col = 0
    
    # Define helper to add a schoolbook 2x2 multiplication
    # Xi is specified by list of (u_idx, coeff) pairs
    # Yi is specified by list of (v_idx, coeff) pairs
    # Result Z = Xi*Yi contributes to W at positions determined by output block
    
    def add_schoolbook_2x2(u_terms, v_terms, w_positions, col_start):
        """
        u_terms: list of (u_row_index, coeff) for the 2x2 input A
        v_terms: list of (v_row_index, coeff) for the 2x2 input B  
        w_positions: dict mapping (local_row, local_col) to (w_row_index, coeff)
        """
        nonlocal col
        for (u_i, c_u) in u_terms:
            for (v_j, c_v) in v_terms:
                for (local_r, local_c), (w_idx, c_w) in w_positions.items():
                    U[u_i][col] += c_u
                    V[v_j][col] += c_v
                    W[w_idx][col] += c_w
                col += 1
        return col
    
    # Actually, let me think more carefully.
    # The tensor T[i,j,k] for 4x4*4x4 = 4x4 has T[i,j,k] = 1 if i,j,k in 0..3.
    # U is 16xR, V is 16xR, W is 16xR where R is rank.
    # U[i,j] indexes A[i,j], V[j,k] indexes B[j,k], W[k,i] indexes C[k,i]
    # Wait, the convention is W[k*n+i] where n=4 for the core.
    
    # Let me just directly construct the Strassen 4x4x4 decomposition.
    
    # For Strassen, we have 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is a tensor of rank 4.
    # The total rank is 7*4 = 28... but that's not right either.
    # The rank of 2x2x2 matrix multiplication is 4 (schoolbook).
    # Strassen shows it can be done with 7 scalar multiplications for 2x2.
    # Wait no - Strassen shows 2x2 matrix mult can be done with 7 multiplications.
    # So the rank of 2x2x2 is 7 (Strassen) or 4 (schoolbook is not minimal).
    # Actually schoolbook 2x2x2 has rank 4, but Strassen shows rank <= 7.
    # Wait, I'm confusing myself. Let me reconsider.
    
    # 2x2 matrix multiplication: C = AB where A,B are 2x2.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Schoolbook: 4 multiplications: A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,1], A[1,1]*B[1,1]
    # Plus additions to get each C entry.
    # The rank of the 2x2x2 tensor is 4 (schoolbook achieves it, and it's known to be 4).
    
    # Strassen for 2x2 uses 7 multiplications of linear combinations.
    # But those 7 "multiplications" are scalar multiplications, not 2x2 matrix multiplications.
    # So Strassen 2x2 has rank 7, which is worse than schoolbook rank 4.
    
    # I think I was confused. Let me restart the analysis.
    
    # For 4x4*4x4:
    # Schoolbook: 4*4*4 = 64
    # Block Strassen: Split into 2x2 blocks, use Strassen (7 mults of 2x2)
    # Each 2x2 multiplication has rank 4 (schoolbook)
    # Total: 7 * 4 = 28
    # But we also need additions/subtractions handled by the linear factors.
    # The rank of the whole 4x4x4 decomposition is 28 with this approach.
    # Wait, that seems too good. Let me verify.
    
    # Actually, I recall that the rank of 4x4x4 matrix multiplication is at most 49
    # using Strassen (7^2 = 49 since log_2(7) * 2 ≈ 3.8, and 4^3.8 ≈ 76... no)
    # Strassen: n^log_2(7). For n=2: 2^2.8 = 7. For n=4: 4^2.8 = 16^1.4 ≈ 25.4
    # So rank of 4x4x4 is at most 49 by naive Strassen (applying Strassen twice: 7*7=49).
    # Or 28 by one level of Strassen with schoolbook 2x2.
    
    # Let me just go with the simpler approach: use rank 49 for 4x4x4.
    # No wait, let me just implement the full schoolbook for now and get a valid answer,
    # then optimize.
    
    # Schoolbook 4x5*5x5: rank = 4*5*5 = 100
    # That's the baseline. Let me at least get a valid decomposition.
    
    # Let me just do schoolbook for the whole thing first.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
