import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Construct a rank-50 exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy:
    - Split the 4x7 matrix B into a 4x4 part and a 4x3 part.
    - The 4x4 part is split into four 2x2 blocks. Each 2x2 block multiplication
      with the 2x2 A submatrix uses Strassen's algorithm (rank 7 per 2x2x2 block).
      Total for the 4x4 part: 4 * 7 = 28 terms.
    - The 4x3 part is handled by direct multiplication: for each of the 3 columns,
      we compute 2x4 * 4x1, which has rank 4 (one term per element of the 4-dim vector B_col).
      Total for the 4x3 part: 3 * 4 = 12 terms.
    - Total rank: 28 + 12 = 40? Wait, let me recalculate.
    
    Actually, let me think more carefully.
    
    A is 2x4, B is 4x7, C is 2x7.
    
    Split B into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2.
    
    For A*B1: Split A into [A00, A01] where A00 is 2x2, A01 is 2x2.
    Split B1 into [[B00, B01], [B10, B11]] where each is 2x2.
    C1 = A*B1 = [A00, A01] * [[B00, B01], [B10, B11]]
               = [A00*B00 + A01*B10, A00*B01 + A01*B11]
    
    Each 2x2 * 2x2 product can be done with Strassen (7 multiplications).
    So A00*B00: 7 terms, A01*B10: 7 terms, A00*B01: 7 terms, A01*B11: 7 terms.
    Total for C1: 28 terms.
    
    For A*B2: B2 is 4x3. Split into columns: b0, b1, b2 (each 4x1).
    A*b0: This is 2x4 * 4x1. The result is a 2x1 vector.
    C2 = [A00*b0_top + A01*b0_bottom, ...] where b0_top is 2x1, b0_bottom is 2x1.
    Each A00*b0_top is 2x2 * 2x1, which has rank 4 (each element of the result is a dot product of a 2-vector with a 2-vector, but actually it's simpler: the tensor for 2x2*2x1 has rank 4).
    
    Wait, 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    This is 4 rank-1 terms.
    
    So for each column of B2, we need 4 terms.
    Total for A*B2: 3 * 4 = 12 terms.
    
    Total rank: 28 + 12 = 40.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    rank = 40
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A * B1 where B1 is the first 4 columns of B (4x4)
    # Split A into [A00, A01] (each 2x2)
    # Split B1 into [[B00, B01], [B10, B11]] (each 2x2)
    # C1 = [A00*B00 + A01*B10, A00*B01 + A01*B11]
    
    # Strassen for 2x2 * 2x2:
    # Given A (2x2) and B (2x2), C = A*B
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to map this into our U, V, W framework.
    # For each Strassen term, we have:
    # U[i*m+j, c] = a (coefficient for A_ij)
    # V[j*p+k, c] = b (coefficient for B_jk)
    # W[k*n+i, c] = w (coefficient for C_ki, using the official trace convention W[k*n+i,r])
    
    # Wait, the convention is W[k*n+i, r] where C = A*B, C is n x p.
    # So C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Actually, the standard tensor decomposition is:
    # C[i,k] = sum_r sum_j sum_l U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] * A[i,j] * B[l,k]
    # Wait, I need to be careful with the indices.
    
    # Let me re-read the task:
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So W is indexed by (k, i) where k is the column index of B (0 to p-1) and i is the row index of A (0 to n-1).
    # The decomposition is:
    # C[i, k] = sum_r U[i*m + j, r] * A[i, j] * V[j*p + k, r] * B[j, k] * W[k*n + i, r]
    
    # Wait, that doesn't look right. Let me think again.
    # The standard matrix multiplication tensor is:
    # T_{i,j,k} = 1 if we're multiplying A_{i,j} and B_{j,k} to get C_{i,k}
    # The decomposition is:
    # T = sum_r U^{(r)} ⊗ V^{(r)} ⊗ W^{(r)}
    # where U^{(r)} is a vector of length n*m (indexed by i*m+j),
    # V^{(r)} is a vector of length m*p (indexed by j*p+k),
    # W^{(r)} is a vector of length p*n (indexed by k*n+i).
    
    # So C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    
    # This means for each term r, we have:
    # U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    
    # For the schoolbook, U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for the corresponding (i,j,k).
    
    # Now, for Strassen, we need to express the product of two 2x2 matrices using 7 terms.
    # Let A be 2x2 with indices (i1, j1) where i1, j1 in {0, 1}.
    # Let B be 2x2 with indices (j2, k2) where j2, k2 in {0, 1}.
    # The product C = A*B has indices (i1, k2).
    
    # In the global indexing:
    # i = i1 (since n=2, i is 0 or 1)
    # j = j1 for A00, j1+2 for A01 (since A is 2x4, split into two 2x2 blocks)
    # k = k2 for the first 4 columns of B (B1), k2+4 for the last 3 columns (B2)
    
    # So for A00 (i1=0,1; j1=0,1): global j = j1
    # For A01 (i1=0,1; j1=0,1): global j = j1+2
    # For B00 (j2=0,1; k2=0,1): global j = j2, k = k2
    # For B01 (j2=0,1; k2=0,1): global j = j2, k = k2+2
    # For B10 (j2=0,1; k2=0,1): global j = j2+2, k = k2
    # For B11 (j2=0,1; k2=0,1): global j = j2+2, k = k2+2
    
    # Let me define a helper to add a Strassen term.
    
    def add_strassen_term(A_row_offset, A_col_offset, B_row_offset, B_col_offset, 
                          m1, m2, m3, m4, m5, m6, m7, C_col_offset):
        """
        Add a Strassen decomposition for a 2x2 * 2x2 product.
        
        A is at rows [A_row_offset, A_row_offset+1], cols [A_col_offset, A_col_offset+1]
        B is at rows [B_row_offset, B_row_offset+1], cols [B_col_offset, B_col_offset+1]
        C result is at rows [0, 1] (since n=2), cols [C_col_offset, C_col_offset+1]
        
        The 7 Strassen multiplications are:
        m1 = (A00+A11)*(B00+B11)
        m2 = (A10+A11)*B00
        m3 = A00*(B01-B11)
        m4 = A11*(B10-B00)
        m5 = (A00+A01)*B11
        m6 = (A10-A00)*(B00+B01)
        m7 = (A01-A11)*(B10+B11)
        
        C00 = m1 + m4 - m5 + m7
        C01 = m3 + m5
        C10 = m2 + m4
        C11 = m1 - m2 + m3 + m6
        """
        nonlocal col
        
        # For each of the 7 terms, we need to set U, V, W appropriately.
        # The term r corresponds to one of the m1..m7 products.
        
        # Let's define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
        # where U_coeffs is a dict from (i, j) to coeff,
        # V_coeffs is a dict from (j, k) to coeff,
        # W_coeffs is a dict from (k, i) to coeff.
        
        # m1 = (A00+A11)*(B00+B11)
        # A00 is at (A_row_offset, A_col_offset), A11 is at (A_row_offset+1, A_col_offset+1)
        # B00 is at (B_row_offset, B_col_offset), B11 is at (B_row_offset+1, B_col_offset+1)
        # C00 is at (0, C_col_offset), C11 is at (1, C_col_offset+1)
        
        terms = []
        
        # m1 = (A00+A11)*(B00+B11)
        terms.append({
            'U': { (0, A_col_offset): 1, (1, A_col_offset+1): 1 },
            'V': { (B_row_offset, B_col_offset): 1, (B_row_offset+1, B_col_offset+1): 1 },
            'W': { (C_col_offset, 0): 1, (C_col_offset+1, 1): 1 }
        })
        
        # m2 = (A10+A11)*B00
        terms.append({
            'U': { (1, A_col_offset): 1, (1, A_col_offset+1): 1 },
            'V': { (B_row_offset, B_col_offset): 1 },
            'W': { (C_col_offset, 1): 1, (C_col_offset+1, 1): 1 }
        })
        
        # m3 = A00*(B01-B11)
        terms.append({
            'U': { (0, A_col_offset): 1 },
            'V': { (B_row_offset, B_col_offset+1): 1, (B_row_offset+1, B_col_offset+1): -1 },
            'W': { (C_col_offset+1, 0): 1, (C_col_offset+1, 1): 1 }
        })
        
        # m4 = A11*(B10-B00)
        terms.append({
            'U': { (1, A_col_offset+1): 1 },
            'V': { (B_row_offset+1, B_col_offset): 1, (B_row_offset, B_col_offset): -1 },
            'W': { (C_col_offset, 0): 1, (C_col_offset+1, 1): 1 }
        })
        
        # m5 = (A00+A01)*B11
        terms.append({
            'U': { (0, A_col_offset): 1, (0, A_col_offset+1): 1 },
            'V': { (B_row_offset+1, B_col_offset+1): 1 },
            'W': { (C_col_offset, 0): -1, (C_col_offset+1, 0): 1, (C_col_offset+1, 1): 1 }
        })
        
        # Wait, I need to be more careful. The Strassen algorithm computes:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # So for each m, it contributes to specific C elements with specific signs.
        # Let me redo this.
        
        terms = []
        
        # m1 = (A00+A11)*(B00+B11)
        # Contributes: +1 to C00, +1 to C11
        terms.append({
            'U': { (0, A_col_offset): 1, (1, A_col_offset+1): 1 },
            'V': { (B_row_offset, B_col_offset): 1, (B_row_offset+1, B_col_offset+1): 1 },
            'W': { (C_col_offset, 0): 1, (C_col_offset+1, 1): 1 }
        })
        
        # m2 = (A10+A11)*B00
        # Contributes: +1 to C10, -1 to C11
        terms.append({
            'U': { (1, A_col_offset): 1, (1, A_col_offset+1): 1 },
            'V': { (B_row_offset, B_col_offset): 1 },
            'W': { (C_col_offset, 1): 1, (C_col_offset+1, 1): -1 }
        })
        
        # m3 = A00*(B01-B11)
        # Contributes: +1 to C01, +1 to C11
        terms.append({
            'U': { (0, A_col_offset): 1 },
            'V': { (B_row_offset, B_col_offset+1): 1, (B_row_offset+1, B_col_offset+1): -1 },
            'W': { (C_col_offset+1, 0): 1, (C_col_offset+1, 1): 1 }
        })
        
        # m4 = A11*(B10-B00)
        # Contributes: +1 to C00, +1 to C10
        terms.append({
            'U': { (1, A_col_offset+1): 1 },
            'V': { (B_row_offset+1, B_col_offset): 1, (B_row_offset, B_col_offset): -1 },
            'W': { (C_col_offset, 0): 1, (C_col_offset, 1): 1 }
        })
        
        # m5 = (A00+A01)*B11
        # Contributes: -1 to C00, +1 to C01
        terms.append({
            'U': { (0, A_col_offset): 1, (0, A_col_offset+1): 1 },
            'V': { (B_row_offset+1, B_col_offset+1): 1 },
            'W': { (C_col_offset, 0): -1, (C_col_offset+1, 0): 1 }
        })
        
        # m6 = (A10-A00)*(B00+B01)
        # Contributes: +1 to C11
        terms.append({
            'U': { (1, A_col_offset): 1, (0, A_col_offset): -1 },
            'V': { (B_row_offset, B_col_offset): 1, (B_row_offset, B_col_offset+1): 1 },
            'W': { (C_col_offset+1, 1): 1 }
        })
        
        # m7 = (A01-A11)*(B10+B11)
        # Contributes: +1 to C00
        terms.append({
            'U': { (0, A_col_offset+1): 1, (1, A_col_offset+1): -1 },
            'V': { (B_row_offset+1, B_col_offset): 1, (B_row_offset+1, B_col_offset+1): 1 },
            'W': { (C_col_offset, 0): 1 }
        })
        
        for term in terms:
            for (i, j), coeff in term['U'].items():
                U[i*m + j, col] += coeff
            for (j, k), coeff in term['V'].items():
                V[j*p + k, col] += coeff
            for (k, i), coeff in term['W'].items():
                W[k*n + i, col] += coeff
            col += 1
    
    # Add Strassen for A00*B00 -> C00 (cols 0-1)
    add_strassen_term(0, 0, 0, 0, 
                     m1=1, m2=1, m3=1, m4=1, m5=1, m6=1, m7=1, 
                     C_col_offset=0)
    
    # Add Strassen for A01*B10 -> C00 (cols 0-1)
    # A01 is at rows 0-1, cols 2-3
    # B10 is at rows 2-3, cols 0-1
    add_strassen_term(0, 2, 2, 0, 
                     m1=1, m2=1, m3=1, m4=1, m5=1, m6=1, m7=1, 
                     C_col_offset=0)
    
    # Add Strassen for A00*B01 -> C01 (cols 2-3)
    # A00 is at rows 0-1, cols 0-1
    # B01 is at rows 0-1, cols 2-3
    add_strassen_term(0, 0, 0, 2, 
                     m1=1, m2=1, m3=1, m4=1, m5=1, m6=1, m7=1, 
                     C_col_offset=2)
    
    # Add Strassen for A01*B11 -> C01 (cols 2-3)
    # A01 is at rows 0-1, cols 2-3
    # B11 is at rows 2-3, cols 2-3
    add_strassen_term(0, 2, 2, 2, 
                     m1=1, m2=1, m3=1, m4=1, m5=1, m6=1, m7=1, 
                     C_col_offset=2)
    
    # Part 2: A * B2 where B2 is the last 3 columns of B (4x3)
    # For each column c in {0, 1, 2}, compute A * B[:, c+4]
    # This is 2x4 * 4x1, which has rank 4.
    # C[i, c+4] = sum_j A[i, j] * B[j, c+4]
    # This is a sum of 4 rank-1 terms (one for each j).
    
    for c in range(3):
        for j in range(4):
            U[0*m + j, col] = 1
            U[1*m + j, col] = 1
            V[j*p + (c+4), col] = 1
            W[(c+4)*n + 0, col] = 1
            W[(c+4)*n + 1, col] = 1
            col += 1
    
    # Verify that col == rank
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
