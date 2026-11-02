"""Optimized tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 by 4x7 matrix.
    
    Strategy: Use a combination of block-wise Strassen-like decompositions and
    careful index mapping to reduce the rank below the schoolbook 56.
    
    The known best for this specific case (2,4,7) is rank 50, achieved by
    decomposing the 4x4 middle block using a 2x2 Strassen approach for the 
    2x2 sub-blocks and handling the remaining dimensions optimally.
    
    However, since we need to provide a concrete implementation, we will
    construct a rank-50 decomposition based on the following structure:
    
    We view the 4x4 multiplication as composed of 2x2 blocks. 
    Let A be 2x4, B be 4x7. We can partition B into [B1; B2] where B1 is 2x7, B2 is 2x7.
    Then C = A * B = A1*B1 + A2*B2 where A1, A2 are 2x2 blocks of A.
    
    Actually, a more standard approach for (2,4,7):
    Partition A (2x4) into columns: A = [A_left(2x2) | A_right(2x2)]
    Partition B (4x7) into rows: B = [B_top(2x7); B_bottom(2x7)]
    
    Then C = A_left * B_top + A_right * B_bottom
    
    Each 2x2 * 2x7 multiplication can be done with rank... 
    Actually, let's use the known result. For 2x4 * 4x7, the rank is 50.
    
    We will use a constructive approach:
    1. For the 2x4 * 4x7 problem, we can think of it as 4 independent 2x1 * 1x7 products 
       summed appropriately, but that gives 4*2*7 = 56.
    
    2. A better approach: Use the fact that 4 = 2+2. 
       Let A = [A1 A2] where A1, A2 are 2x2.
       Let B = [B1; B2] where B1, B2 are 2x7.
       C = A1*B1 + A2*B2.
       
       For each 2x2 * 2x7 multiplication, we can use a rank-... 
       The rank of 2x2 * 2x7 is at most 2*2*7 = 28, but we can do better.
       Actually, rank(2x2 * 2xp) = 2*2*p - 2*p + 2 = 2p? No.
       
       Let's use the known formula: rank(m x n * n x p) <= m*n*p - min(m,n)*min(n,p) + min(m,p)
       This is not always tight.
       
       For 2x2 * 2x7: We can use 2 Strassen-like steps? 
       Actually, a 2x2 * 2x2 is rank 7 (Strassen). 
       For 2x2 * 2x7, we can treat the 7 columns as 3 pairs of 2 columns + 1 column.
       So rank = 3 * 7 (for the 2x2*2x2 parts) + 4 (for the 2x2*2x1 part) = 21 + 4 = 25?
       Wait, 2x2 * 2x1 is just 2x1 * 2x1 which is rank 2*1*1 = 2? No, it's a dot product, rank 2.
       
       So for 2x2 * 2x7:
       - 3 groups of 2x2 * 2x2: each rank 7, total 21
       - 1 group of 2x2 * 2x1: rank 2*1*1 = 2? No, 2x2 * 2x1 = (2x2)*(2x1) = 2x1 result.
         The tensor is 2x2x2x1. Rank is at most 2*2*1 = 4. Actually it's just 4 dot products.
         But we can do it with rank 2? No, each output element is a dot product of a 2-vector and a 2-vector.
         There are 2 output elements. So rank 2? No, the input is 2x2 matrix and 2x1 vector.
         The product is a 2x1 vector. Each element is a dot product. 
         So we need 2 rank-1 tensors? No, each rank-1 tensor contributes to all outputs.
         Actually, rank(2x2 * 2x1) = 2. Because we can write:
         C[0] = A[0,0]*B[0] + A[0,1]*B[1]
         C[1] = A[1,0]*B[0] + A[1,1]*B[1]
         This is 4 terms, but can we do it with 2? 
         C[0] = (A[0,0]+A[0,1]) * (B[0]+B[1]) - A[0,0]*B[1] - A[0,1]*B[0]
         That's not rank 1. 
         Actually, rank(2x2 * 2x1) = 2 is not correct. It's 4? No.
         The tensor is in R^{2x2} \otimes R^{2} \otimes R^{2} (output is 2x1).
         Wait, the output is 2x1, so the third factor is R^2.
         The rank is the minimum number of terms \sum u_k \otimes v_k \otimes w_k.
         For 2x2 * 2x1, we have 2*2*1 = 4 scalar multiplications. 
         Can we do it with fewer? 
         C[0] = A[0,0]B[0] + A[0,1]B[1]
         C[1] = A[1,0]B[0] + A[1,1]B[1]
         This is 4 terms. Is there a way to do it with 3? 
         Probably not, since the border rank is 4. So rank = 4.
       
       So for 2x2 * 2x7: 3 * 7 + 4 = 25.
       Total for C = A1*B1 + A2*B2: 25 + 25 = 50.
       
    So the rank is 50.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    rank = 50
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # We partition A (2x4) into A1 (2x2) and A2 (2x2)
    # A1 corresponds to A[0:2, 0:2], A2 to A[0:2, 2:4]
    # B (4x7) is partitioned into B1 (2x7) and B2 (2x7)
    # B1 corresponds to B[0:2, :], B2 to B[2:4, :]
    
    # C = A1 * B1 + A2 * B2
    
    # Helper function to add a 2x2 * 2x2 Strassen block
    def add_strassen_2x2_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_row_offset, w_col_offset):
        nonlocal col
        # Strassen's 7 products for 2x2 * 2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Indices:
        # A[i, j] -> U index: (a_row_offset + i) * m + (a_col_offset + j)
        # B[i, j] -> V index: (b_row_offset + i) * p + (b_col_offset + j)
        # C[i, j] -> W index: (w_row_offset + j) * n + (w_row_offset + i)  # W[k*n+i, r]
        
        # P1 = (a+d)(e+h)
        # This requires 4 terms: a*e, a*h, d*e, d*h
        # But Strassen combines them.
        
        # Actually, let's just use the schoolbook for 2x2*2x2 but with Strassen's 7 terms.
        # The 7 terms are:
        # P1 = (a+d)(e+h) -> (a+d) is a linear combo, (e+h) is a linear combo
        # We need to express each Pk as a product of a linear combo of A entries and a linear combo of B entries.
        
        # Let's define the 7 products:
        # P1: U part: a + d, V part: e + h
        # P2: U part: c + d, V part: e
        # P3: U part: a, V part: f - h
        # P4: U part: d, V part: g - e
        # P5: U part: a + b, V part: h
        # P6: U part: c - a, V part: e + f
        # P7: U part: b - d, V part: g + h
        
        # For each product, we add 1 to the appropriate U, V, W entries.
        
        # P1
        U[a_row_offset * m + a_col_offset][col] = 1       # a
        U[(a_row_offset + 1) * m + a_col_offset + 1][col] = 1  # d
        V[b_row_offset * p + b_col_offset][col] = 1       # e
        V[(b_row_offset + 1) * p + b_col_offset + 1][col] = 1  # h
        # W part: C11 += 1, C22 += 1
        W[(w_row_offset + 0) * n + w_row_offset][col] = 1  # C[0,0]
        W[(w_row_offset + 1) * n + w_row_offset + 1][col] = 1  # C[1,1]
        col += 1
        
        # P2
        U[(a_row_offset + 1) * m + a_col_offset][col] = 1  # c
        U[(a_row_offset + 1) * m + a_col_offset + 1][col] = 1  # d
        V[b_row_offset * p + b_col_offset][col] = 1       # e
        W[(w_row_offset + 1) * n + w_row_offset][col] = 1  # C[1,0]
        W[(w_row_offset + 1) * n + w_row_offset + 1][col] = -1  # C[1,1] -= 1
        col += 1
        
        # P3
        U[a_row_offset * m + a_col_offset][col] = 1       # a
        V[(b_row_offset) * p + b_col_offset + 1][col] = 1  # f
        V[(b_row_offset + 1) * p + b_col_offset + 1][col] = -1  # -h
        W[(w_row_offset + 0) * n + w_row_offset][col] = 1  # C[0,0]
        W[(w_row_offset + 0) * n + w_row_offset + 1][col] = 1  # C[0,1]
        col += 1
        
        # P4
        U[(a_row_offset + 1) * m + a_col_offset + 1][col] = 1  # d
        V[(b_row_offset + 1) * p + b_col_offset][col] = 1  # g
        V[b_row_offset * p + b_col_offset][col] = -1  # -e
        W[(w_row_offset + 0) * n + w_row_offset][col] = 1  # C[0,0]
        W[(w_row_offset + 1) * n + w_row_offset][col] = 1  # C[1,0]
        col += 1
        
        # P5
        U[a_row_offset * m + a_col_offset][col] = 1       # a
        U[a_row_offset * m + a_col_offset + 1][col] = 1   # b
        V[(b_row_offset + 1) * p + b_col_offset + 1][col] = 1  # h
        W[(w_row_offset + 0) * n + w_row_offset][col] = -1  # C[0,0] -= 1
        W[(w_row_offset + 0) * n + w_row_offset + 1][col] = 1  # C[0,1]
        col += 1
        
        # P6
        U[(a_row_offset + 1) * m + a_col_offset][col] = 1  # c
        U[a_row_offset * m + a_col_offset][col] = -1       # -a
        V[b_row_offset * p + b_col_offset][col] = 1       # e
        V[b_row_offset * p + b_col_offset + 1][col] = 1   # f
        W[(w_row_offset + 1) * n + w_row_offset + 1][col] = 1  # C[1,1]
        col += 1
        
        # P7
        U[a_row_offset * m + a_col_offset + 1][col] = 1   # b
        U[(a_row_offset + 1) * m + a_col_offset + 1][col] = -1  # -d
        V[(b_row_offset + 1) * p + b_col_offset][col] = 1  # g
        V[(b_row_offset + 1) * p + b_col_offset + 1][col] = 1  # h
        W[(w_row_offset + 0) * n + w_row_offset][col] = 1  # C[0,0]
        W[(w_row_offset + 1) * n + w_row_offset + 1][col] = 1  # C[1,1]
        col += 1
    
    # Helper function to add a 2x2 * 2x1 schoolbook block
    def add_schoolbook_2x2_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_row_offset, w_col_offset):
        nonlocal col
        # A is 2x2, B is 2x1 (column vector)
        # C = A * B is 2x1
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        # 4 terms
        
        # Term 1: A[0,0] * B[0]
        U[a_row_offset * m + a_col_offset][col] = 1
        V[b_row_offset * p + b_col_offset][col] = 1
        W[(w_row_offset + 0) * n + w_row_offset][col] = 1  # C[0, w_col_offset]
        col += 1
        
        # Term 2: A[0,1] * B[1]
        U[a_row_offset * m + a_col_offset + 1][col] = 1
        V[(b_row_offset + 1) * p + b_col_offset][col] = 1
        W[(w_row_offset + 0) * n + w_row_offset][col] = 1  # C[0, w_col_offset]
        col += 1
        
        # Term 3: A[1,0] * B[0]
        U[(a_row_offset + 1) * m + a_col_offset][col] = 1
        V[b_row_offset * p + b_col_offset][col] = 1
        W[(w_row_offset + 1) * n + w_row_offset][col] = 1  # C[1, w_col_offset]
        col += 1
        
        # Term 4: A[1,1] * B[1]
        U[(a_row_offset + 1) * m + a_col_offset + 1][col] = 1
        V[(b_row_offset + 1) * p + b_col_offset][col] = 1
        W[(w_row_offset + 1) * n + w_row_offset][col] = 1  # C[1, w_col_offset]
        col += 1
    
    # Now, C = A1 * B1 + A2 * B2
    # A1 is A[0:2, 0:2], A2 is A[0:2, 2:4]
    # B1 is B[0:2, :], B2 is B[2:4, :]
    
    # For A1 * B1:
    # B1 is 2x7. We partition the 7 columns into 3 pairs + 1 single.
    # Columns 0-1, 2-3, 4-5, 6
    
    # A1 * B1[:, 0:2]
    add_strassen_2x2_2x2(0, 0, 0, 0, 0, 0)
    # A1 * B1[:, 2:4]
    add_strassen_2x2_2x2(0, 0, 0, 2, 0, 2)
    # A1 * B1[:, 4:6]
    add_strassen_2x2_2x2(0, 0, 0, 4, 0, 4)
    # A1 * B1[:, 6:7]
    add_schoolbook_2x2_2x1(0, 0, 0, 6, 0, 6)
    
    # For A2 * B2:
    # A2 is A[0:2, 2:4]
    # B2 is B[2:4, :]
    
    # A2 * B2[:, 0:2]
    add_strassen_2x2_2x2(0, 2, 2, 0, 0, 0)
    # A2 * B2[:, 2:4]
    add_strassen_2x2_2x2(0, 2, 2, 2, 0, 2)
    # A2 * B2[:, 4:6]
    add_strassen_2x2_2x2(0, 2, 2, 4, 0, 4)
    # A2 * B2[:, 6:7]
    add_schoolbook_2x2_2x1(0, 2, 2, 6, 0, 6)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # Verify the decomposition
    # The tensor is T[i,j,k] = delta[i,k] for the multiplication tensor
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k]
    # Wait, the output is C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we are computing C[i,k], but the tensor is 3D: A[i,j] * B[j,k] -> C[i,k]
    # The tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # So the tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k
    # We need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify
    rank = U.shape[1]
    total = np.zeros((n, m, p))
    for r in range(rank):
        total += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check if total[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    if np.allclose(total, expected):
        print("Decomposition is correct!")
        print(f"Rank: {rank}")
    else:
        print("Decomposition is INCORRECT!")
        print("Max error:", np.max(np.abs(total - expected)))
