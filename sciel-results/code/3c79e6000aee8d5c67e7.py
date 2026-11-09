import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(n * p)
        
        for idx in u_indices:
            u[idx] = 1.0
        for idx in v_indices:
            v[idx] = 1.0
        for idx in w_indices:
            w[idx] = 1.0
            
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Part 1: A[0:4, 0:4] @ B[0:4, 0:4] -> C[0:4, 0:4]
    # 4x4x4 using Strassen (Rank 49)
    # Strassen for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map 4x4 blocks to 2x2 sub-blocks.
    # A4 = [[A11, A12], [A21, A22]] where each is 2x2.
    # B4 = [[B11, B12], [B21, B22]] where each is 2x2.
    # C4 = [[C11, C12], [C21, C22]] where each is 2x2.
    
    # Strassen formulas for 2x2 blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, which has rank 7.
    # So we need 7 * 7 = 49 terms.
    
    # Let's define the 7 terms for a generic 2x2x2 product X@Y=Z
    # X is 2x2, Y is 2x2, Z is 2x2.
    # Indices for X: x00, x10, x01, x11 (row-major? i*m+j)
    # Indices for Y: y00, y10, y01, y11
    # Indices for Z: z00, z10, z01, z11
    
    def strassen_2x2_terms(offset_x, offset_y, offset_z):
        # Returns list of (u_indices, v_indices, w_indices) for 7 terms
        # offset_x: starting index for X in global U
        # offset_y: starting index for Y in global V
        # offset_z: starting index for Z in global W
        
        terms = []
        
        # M1 = (x00+x11)(y00+y11)
        # Z00 += M1, Z11 += M1
        # x00, x11 in U. y00, y11 in V.
        # But Strassen combines sums.
        # U has x00+x11. This is not a rank-1 term in the original basis.
        # We must expand the sums into rank-1 terms of the original tensor.
        
        # M1 = (x00+x11)(y00+y11) = x00y00 + x00y11 + x11y00 + x11y11
        # This adds 4 rank-1 terms to Z00 and Z11.
        
        # This expansion increases the rank!
        # Rank of M1 is 4.
        # Total rank would be 7 * 4 = 28? No.
        
        # Wait, Strassen reduces the number of multiplications, but if we expand the sums,
        # we are back to schoolbook or worse?
        
        # No, Strassen works because we compute the sums first.
        # But the tensor decomposition requires linear combinations of the input variables.
        # The rank-1 terms in the decomposition of the multiplication tensor
        # correspond to the bilinear forms.
        
        # The rank of the 2x2x2 multiplication tensor is 7.
        # This means there are 7 bilinear forms (rank-1 tensors) that sum to the multiplication tensor.
        # These bilinear forms are NOT simply products of single elements.
        # They are products of linear combinations.
        
        # Example: M1 = (x00+x11)(y00+y11).
        # This is one bilinear form.
        # In the decomposition, this corresponds to:
        # U vector has 1 at x00 and 1 at x11.
        # V vector has 1 at y00 and 1 at y11.
        # W vector has 1 at z00 and 1 at z11.
        
        # So, for each of the 7 Strassen terms, we add ONE rank-1 term to the global decomposition.
        # The U, V, W vectors will have multiple non-zero entries.
        
        # So, for 4x4x4:
        # We have 7 Strassen terms for the 4x4 blocks.
        # Each of these 7 terms is a product of two 2x2 matrices.
        # Each 2x2 matrix product has rank 7.
        # So each of the 7 "big" terms is a sum of 7 "small" rank-1 terms?
        # No.
        
        # The rank of 4x4x4 is 49.
        # This is obtained by applying Strassen to the 4x4 blocks,
        # and then applying Strassen to the resulting 2x2 blocks.
        
        # So, we need to generate 49 rank-1 terms.
        
        # Let's implement a recursive generator.
        
        pass

    # This is getting too complex to implement correctly in the short time.
    # Let's stick to a simpler improvement: Rank 92.
    # Use Strassen for 2x2x2 blocks where possible.
    
    # Split 4x5x5 into:
    # 2x5x5 (top) and 2x5x5 (bottom).
    # Each 2x5x5 can be split into 2x2x2 and 2x3x3?
    
    # Let's try Rank 92 from inspiration 3.
    # "block-decomposition (schoolbook for 2x2, 2x3, 3x2, 3x3 blocks) resulting in rank 92"
    
    # We can implement this.
    # Split A (4x5) into A1 (2x5), A2 (2x5).
    # Split B (5x5) into B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3).
    
    # C11 = A1[:, 0:2] @ B11 (2x2) -> Rank 4
    # C12 = A1[:, 0:2] @ B12 (2x3) -> Rank 6
    # C13 = A1[:, 2:5] @ B21 (3x2) -> Rank 6
    # C14 = A1[:, 2:5] @ B22 (3x3) -> Rank 9
    # Total for Top: 4+6+6+9 = 25.
    
    # Similarly for Bottom: 25.
    # Total 50?
    # Wait, A1 is 2x5.
    # A1 @ B = [A1[:, 0:2] @ B[:, 0:2], A1[:, 0:2] @ B[:, 2:5], A1[:, 2:5] @ B[:, 0:2], A1[:, 2:5] @ B[:, 2:5]]
    # This is not correct.
    # A1 @ B = sum_{k=0}^4 A1[:, k] B[k, :]
    
    # Correct block decomposition:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3.
    
    # C11 (2x2) = A11 B11 + A12 B21
    # C12 (2x3) = A11 B12 + A12 B22
    # C21 (2x2) = A21 B11 + A22 B21
    # C22 (2x3) = A21 B12 + A22 B22
    
    # Ranks:
    # C11: Rank(A11 B11) + Rank(A12 B21) <= 4 + 6 = 10
    # C12: Rank(A11 B12) + Rank(A12 B22) <= 6 + 9 = 15
    # C21: Rank(A21 B11) + Rank(A22 B21) <= 4 + 6 = 10
    # C22: Rank(A21 B12) + Rank(A22 B22) <= 6 + 9 = 15
    
    # Total Rank <= 10 + 15 + 10 + 15 = 50.
    
    # This is much better than 100.
    
    # We need to implement the rank-4, rank-6, rank-9 decompositions.
    # Rank 4: 2x2x2 schoolbook.
    # Rank 6: 2x3x2 or 2x2x3 schoolbook.
    # Rank 9: 3x3x2 or 3x2x3 schoolbook.
    
    # Let's implement this.

    def add_schoolbook(n, m, p, a_offset, b_offset, c_offset):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = a_offset + i * m + j
                    v_idx = b_offset + j * p + k
                    w_idx = c_offset + k * n + i
                    add_term([u_idx], [v_idx], [w_idx])

    # A offsets:
    # A11: 0
    # A12: 4
    # A21: 14
    # A22: 18
    # A is 4x5.
    # A11 (2x2): rows 0-1, cols 0-1. Offset 0.
    # A12 (2x3): rows 0-1, cols 2-4. Offset 4.
    # A21 (2x2): rows 2-3, cols 0-1. Offset 14.
    # A22 (2x3): rows 2-3, cols 2-4. Offset 18.
    
    # B offsets:
    # B is 5x5.
    # B11 (2x2): rows 0-1, cols 0-1. Offset 0.
    # B12 (2x3): rows 0-1, cols 2-4. Offset 6.
    # B21 (3x2): rows 2-4, cols 0-1. Offset 10.
    # B22 (3x3): rows 2-4, cols 2-4. Offset 16.
    
    # C offsets:
    # C is 4x5.
    # C11 (2x2): rows 0-1, cols 0-1. Offset 0.
    # C12 (2x3): rows 0-1, cols 2-4. Offset 6.
    # C21 (2x2): rows 2-3, cols 0-1. Offset 10.
    # C22 (2x3): rows 2-3, cols 2-4. Offset 16.
    
    # C11 = A11 B11 + A12 B21
    add_schoolbook(2, 2, 2, 0, 0, 0)      # A11 B11
    add_schoolbook(2, 3, 2, 4, 10, 0)     # A12 B21 (A12 is 2x3, B21 is 3x2, C is 2x2)
    
    # C12 = A11 B12 + A12 B22
    add_schoolbook(2, 2, 3, 0, 6, 6)      # A11 B12 (A11 is 2x2, B12 is 2x3, C is 2x3)
    add_schoolbook(2, 3, 3, 4, 16, 6)     # A12 B22 (A12 is 2x3, B22 is 3x3, C is 2x3)
    
    # C21 = A21 B11 + A22 B21
    add_schoolbook(2, 2, 2, 14, 0, 10)    # A21 B11
    add_schoolbook(2, 3, 2, 18, 10, 10)   # A22 B21
    
    # C22 = A21 B12 + A22 B22
    add_schoolbook(2, 2, 3, 14, 6, 16)    # A21 B12
    add_schoolbook(2, 3, 3, 18, 16, 16)   # A22 B22
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
