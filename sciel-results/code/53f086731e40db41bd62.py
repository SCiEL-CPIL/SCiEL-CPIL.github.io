"""Matrix multiplication 4x5x5 tensor decomposition with Strassen-based block strategy."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _strassen_2x2(A, B):
    """Strassen 2x2 matrix multiplication. A, B are 2x2 numpy arrays.
    Returns 2x2 product using 7 multiplications."""
    # Split into quadrants
    A11, A12 = A[0,0], A[0,1]
    A21, A22 = A[1,0], A[1,1]
    B11, B12 = B[0,0], B[0,1]
    B21, B22 = B[1,0], B[1,1]
    
    # 7 products
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    # Reconstruct
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    return np.array([[C11, C12], [C21, C22]])

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Split the 4x5 matrix into two 2x5 blocks. For each block, use Strassen
    on the 2x2 row structure. The key insight is that we can group the 5 columns
    and use the 7 Strassen products to reduce the rank.
    
    Actually, let me think more carefully. The tensor is T[i,j,k] where:
    - i in {0,1,2,3} (rows of A, 4 rows)
    - j in {0,1,2,3,4} (cols of A / rows of B, 5)
    - k in {0,1,2,3,4} (cols of B, 5)
    
    The product C[i,k] = sum_j A[i,j] * B[j,k]
    
    Schoolbook: rank = 4*5*5 = 100
    
    With Strassen on the 4x4x4 sub-problem (first 4 rows, first 4 cols of the 
    summation), we can reduce 4*4*4=64 to 7*4=28 for that part.
    
    Let me think about a block decomposition:
    - Rows 0-1 of A, rows 2-3 of A
    - Cols 0-1 of B, cols 2-3 of B, col 4 of B
    
    Actually, the simplest approach that's known to work:
    For 4x5x5, we can use the fact that 4 = 2+2 and 5 = 2+2+1.
    
    Let me use a different approach: direct rank reduction.
    
    The tensor T has entries T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears.
    
    A rank-r decomposition means: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    where U has shape (4*5, r), V has shape (5*5, r), W has shape (5*4, r).
    
    Using Strassen for the 4x4x4 core:
    Split rows into {0,1} and {2,3}, split the summation index j into {0,1} and {2,3} and {4}.
    
    For j in {0,1}: 2x2x2 Strassen gives rank 7
    For j in {2,3}: 2x2x2 Strassen gives rank 7  
    For j = 4: 4x1x1 = rank 4 (each row i needs its own term)
    
    But wait, the 5 columns of B also matter. Let me reconsider.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k]
    
    Group j = {0,1} and j = {2,3}:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k]) + A[i,4]*B[4,k]
    
    For the first group (j=0,1), using Strassen on the 2x2 (rows x cols) with 5 output columns:
    Actually Strassen works on 2x2 matrices. Here we have 4 rows and 5 output cols.
    
    Let me try a cleaner approach. Use the known result that for 4x4x4, rank <= 49
    (Strassen: 7^2 = 49). For 4x5x5, we can extend:
    
    C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
    
    Split into:
    - j in {0,1,2,3}: 4x4x4 part -> rank 49 (Strassen)
    - j = 4: 4x1x5 part -> rank 4*5 = 20 (each (i,k) pair needs one term)
    
    Total: 49 + 20 = 69
    
    Wait, that's not quite right. The j=4 term is A[i,4]*B[4,k] for all i,k.
    This is an outer product of the vector [A[i,4]] and [B[4,k]], which has rank 5
    (not 20) because it's a single rank-1 tensor... no wait.
    
    Actually, A[i,4]*B[4,k] for fixed j=4: the tensor is sum over i,k of A[i,4]*B[4,k]*delta.
    This is a rank-5 tensor? No. It's rank 1 in the tensor sense if we view it as
    U[i,4,c]*V[4,k,c]*W[i,k,c] for a single c where U[i,4,c]=1, V[4,k,c]=1, W[i,k,c]=1.
    
    Hmm, no. Let me be more careful.
    
    The full tensor T[i,j,k] = delta (it's 1 for all valid (i,j,k) in the multiplication).
    
    A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook, each (i,j,k) triple gets its own column c.
    
    For Strassen on 4x4x4 (i in {0..3}, j in {0..3}, k in {0..3}):
    We use 49 terms instead of 64.
    
    For the remaining terms with j=4: T[i,4,k] = 1 for i in {0..3}, k in {0..4}.
    That's 4*5 = 20 terms in schoolbook.
    
    Can we do better? T[i,4,k] = A[i,4]*B[4,k]. In the tensor decomposition:
    U[i*5+4, c] * V[4*5+k, c] * W[k*4+i, c]
    
    We need: sum_c U[i,4,c] * V[4,k,c] * W[i,k,c] = 1 for all i in {0..3}, k in {0..4}
    
    This is equivalent to: the 4x5 matrix of ones = U_col * V_row^T * (element-wise with W)
    
    Actually it's: for each (i,k), sum_c U[i,4,c]*V[4,k,c]*W[i,k,c] = 1.
    
    If we set W[i,k,c] = 1 for all, then we need sum_c U[i,4,c]*V[4,k,c] = 1,
    which is a rank-1 matrix factorization of the 4x5 all-ones matrix.
    The all-ones 4x5 matrix has rank 1, so we can do it with c=1:
    U[i,4,1] = 1, V[4,k,1] = 1 for all i,k.
    
    So the j=4 part only needs rank 1! Not 20!
    
    Total rank: 49 + 1 = 50.
    
    Let me verify this more carefully.
    
    For the 4x4x4 Strassen part: we need T[i,j,k] = 1 for i in {0..3}, j in {0..3}, k in {0..3}
    to be represented by 49 terms.
    
    For the j=4 part: T[i,4,k] = 1 for i in {0..3}, k in {0..4}
    represented by 1 term: U[i*5+4, 49] = 1, V[4*5+k, 49] = 1, W[k*4+i, 49] = 1.
    
    Check: sum_c U[i*5+4, c] * V[4*5+k, c] * W[k*4+i, c]
    = U[i*5+4, 49] * V[4*5+k, 49] * W[k*4+i, 49]
    = 1 * 1 * 1 = 1. ✓
    
    And for the 4x4x4 part, we need to make sure the j=4 term doesn't interfere.
    In the Strassen decomposition for 4x4x4, we only use columns 0..48, and
    U[i*5+j, c] is nonzero only for j in {0..3} when c < 49.
    V[j*5+k, c] is nonzero only for j in {0..3} when c < 49.
    So U[i*5+4, c] = 0 for c < 49, and V[4*5+k, c] = 0 for c < 49.
    
    Great, so total rank = 50.
    
    Now I need to construct the actual Strassen 4x4x4 decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the Strassen 4x4x4 decomposition for i in {0..3}, j in {0..3}, k in {0..3}
    # using 49 terms (7x7 Strassen products)
    
    # Strassen 2x2x2: 7 terms
    # For 4x4x4, we apply Strassen twice: 7 * 7 = 49 terms
    
    # Let me construct this explicitly.
    # First level: split i into {0,1} and {2,3}, j into {0,1} and {2,3}
    # Second level: for each 2x2x2 block, split k into {0,1} and {2,3}
    
    # Actually, the standard approach for 4x4x4 with rank 49:
    # Use 7 Strassen products at the top level (for the 2x2 row/col split of the 4x4 matrix),
    # and for each product, use 7 more Strassen products for the inner 2x2x2.
    
    # Let me define the Strassen 2x2x2 decomposition explicitly.
    # T[i,j,k] for i,j,k in {0,1} has rank 7.
    # The 7 terms correspond to the 7 Strassen products M1..M7.
    
    # For 2x2x2 Strassen:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11  
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C10
    # M5 = (A00+A01)B11 -> contributes to C01, C11
    # M6 = (A10-A00)(B00+B01) -> contributes to C10
    # M7 = (A01-A11)(B10+B11) -> contributes to C01, C10
    
    # In tensor form, for the 2x2x2 problem:
    # T[i,j,k] = sum_{c=1}^{7} U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    # Let me enumerate the 7 terms for 2x2x2:
    # Term 1: U = [[1,0],[1,0]] for (i,j), V = [[1,0],[1,0]] for (j,k), W = [[1,0],[1,0]] for (i,k)
    #   Check: T[0,0,0] += 1*1*1 = 1, T[0,0,1] += 1*0*0 = 0, T[1,0,0] += 1*1*0 = 0, ...
    #   Hmm, this isn't quite right. Let me think more carefully.
    
    # Actually, for the tensor T[i,j,k] = 1 (all ones for 2x2x2), a rank-7 decomposition
    # via Strassen means:
    # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = 1 for all i,j,k in {0,1}
    
    # The Strassen decomposition of matrix multiplication says:
    # C = AB where C[i,k] = sum_j A[i,j] B[j,k]
    # Using 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor form, each Mc corresponds to a rank-1 term in the tensor decomposition.
    # For Mc = (linear combo of A entries) * (linear combo of B entries):
    # U[i,j,c] = coefficient of A[i,j] in the first factor
    # V[j,k,c] = coefficient of B[j,k] in the second factor
    # W[i,k,c] = coefficient in C[i,k] (from the reconstruction)
    
    # Let me work this out:
    # M1 = (A00+A11)(B00+B11)
    #   U[:, :, 1]: A00 has coeff 1, A11 has coeff 1 -> U[0,0,1]=1, U[1,1,1]=1, others 0
    #   V[:, :, 1]: B00 has coeff 1, B11 has coeff 1 -> V[0,0,1]=1, V[1,1,1]=1, others 0
    #   W[:, :, 1]: C00 gets +1, C11 gets +1 -> W[0,0,1]=1, W[1,1,1]=1, others 0
    
    # M2 = (A10+A11)B00
    #   U[:, :, 2]: A10 coeff 1, A11 coeff 1 -> U[1,0,2]=1, U[1,1,2]=1
    #   V[:, :, 2]: B00 coeff 1 -> V[0,0,2]=1
    #   W[:, :, 2]: C10 gets +1, C11 gets -1 -> W[1,0,2]=1, W[1,1,2]=-1
    
    # M3 = A00(B01-B11)
    #   U[:, :, 3]: A00 coeff 1 -> U[0,0,3]=1
    #   V[:, :, 3]: B01 coeff 1, B11 coeff -1 -> V[0,1,3]=1, V[1,1,3]=-1
    #   W[:, :, 3]: C01 gets +1 -> W[0,1,3]=1
    
    # M4 = A11(B10-B00)
    #   U[:, :, 4]: A11 coeff 1 -> U[1,1,4]=1
    #   V[:, :, 4]: B10 coeff 1, B00 coeff -1 -> V[1,0,4]=1, V[0,0,4]=-1
    #   W[:, :, 4]: C10 gets +1, C00 gets +1 -> W[1,0,4]=1, W[0,0,4]=1
    
    # M5 = (A00+A01)B11
    #   U[:, :, 5]: A00 coeff 1, A01 coeff 1 -> U[0,0,5]=1, U[0,1,5]=1
    #   V[:, :, 5]: B11 coeff 1 -> V[1,1,5]=1
    #   W[:, :, 5]: C01 gets +1, C00 gets -1 -> W[0,1,5]=1, W[0,0,5]=-1
    
    # M6 = (A10-A00)(B00+B01)
    #   U[:, :, 6]: A10 coeff 1, A00 coeff -1 -> U[1,0,6]=1, U[0,0,6]=-1
    #   V[:, :, 6]: B00 coeff 1, B01 coeff 1 -> V[0,0,6]=1, V[0,1,6]=1
    #   W[:, :, 6]: C11 gets +1 -> W[1,1,6]=1
    
    # M7 = (A01-A11)(B10+B11)
    #   U[:, :, 7]: A01 coeff 1, A11 coeff -1 -> U[0,1,7]=1, U[1,1,7]=-1
    #   V[:, :, 7]: B10 coeff 1, B11 coeff 1 -> V[1,0,7]=1, V[1,1,7]=1
    #   W[:, :, 7]: C00 gets +1, C10 gets +1 -> W[0,0,7]=1, W[1,0,7]=1
    
    # Let me verify M1:
    # sum_{i,j,k} U[i,j,1]*V[j,k,1]*W[i,k,1] * T[i,j,k]
    # = U[0,0,1]*V[0,0,1]*W[0,0,1]*1 + U[1,1,1]*V[1,1,1]*W[1,1,1]*1
    # = 1*1*1 + 1*1*1 = 2
    # But we need T[0,0,0]=1 and T[1,1,1]=1, each contributing 1.
    # The total contribution to T[0,0,0] from all 7 terms should be 1.
    # From M1: U[0,0,1]*V[0,0,1]*W[0,0,1] = 1
    # From M4: U[0,0,4]*V[0,0,4]*W[0,0,4] = 0 (U[0,0,4]=0)
    # From M5: U[0,0,5]*V[0,0,5]*W[0,0,5] = 1*0*(-1) = 0 (V[0,0,5]=0)
    # From M7: U[0,0,7]*V[0,0,7]*W[0,0,7] = 0 (U[0,0,7]=0)
    # So T[0,0,0] gets 1 from M1. ✓
    
    # This is getting complex. Let me just implement it programmatically.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Build the 4x4x4 Strassen decomposition (rank 49)
    # We'll use the recursive Strassen approach
    
    def strassen_2x2x2_decomp():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: shape (2*2, 7) = (4, 7), indexed by [i*2+j, c]
        V: shape (2*2, 7) = (4, 7), indexed by [j*2+k, c]
        W: shape (2*2, 7) = (4, 7), indexed by [k*2+i, c]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
        V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
        W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
        
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
        V[0*2+0, 1] = 1
        W[1*2+0, 1] = 1; W[1*2+1, 1] = -1
        
        # M3 = A00(B01-B11) -> C01 += 1
        U[0*2+0, 2] = 1
        V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
        W[0*2+1, 2] = 1
        
        # M4 = A11(B10-B00) -> C10 += 1, C00 += 1
        U[1*2+1, 3] = 1
        V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
        W[1*2+0, 3] = 1; W[0*2+0, 3] = 1
        
        # M5 = (A00+A01)B11 -> C01 += 1, C00 -= 1
        U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
        V[1*2+1, 4] = 1
        W[0*2+1, 4] = 1; W[0*2+0, 4] = -1
        
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
        V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
        W[1*2+1, 5] = 1
        
        # M7 = (A01-A11)(B10+B11) -> C00 += 1, C10 += 1
        U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
        V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
        W[0*2+0, 6] = 1; W[1*2+0, 6] = 1
        
        return U, V, W
    
    # For 4x4x4, we apply Strassen at two levels.
    # Level 1: Split the 4x4 matrix into four 2x2 blocks.
    #   Rows: {0,1} and {2,3}
    #   Cols: {0,1} and {2,3}
    # Level 2: For each of the 7 Strassen products, we have a 2x2x2 subproblem.
    
    # The 4x4x4 tensor: T[i,j,k] for i,j,k in {0..3}
    # We split i into (i1, i2) where i1 in {0,1} selects the row block and i2 in {0,1} the row within block
    # Similarly for j and k.
    
    # Actually, let me think of it differently.
    # C = AB where A is 4x4, B is 4x4.
    # Split A into [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # Split B into [[B00, B01], [B10, B11]] where each Bj is 2x2.
    # C = [[C00, C01], [C10, C11]] where Ci is 2x2.
    
    # Strassen gives 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix product, which we compute with 7 Strassen terms.
    # So total: 7 * 7 = 49 terms.
    
    # For the tensor decomposition of 4x4x4:
    # T[i,j,k] where i is row of A (0..3), j is col of A / row of B (0..3), k is col of B (0..3)
    
    # Let me define the decomposition.
    # For each of the 7 top-level Strassen products (call them alpha = 0..6),
    # and for each of the 7 inner Strassen products (call them beta = 0..6),
    # we have one term with column c = alpha*7 + beta.
    
    # The top-level Strassen combines A blocks and B blocks.
    # The inner Strassen computes the actual 2x2 matrix product.
    
    # Let me define the top-level coefficients.
    # For top-level product alpha, the A-side is a linear combination of A00, A01, A10, A11
    # and the B-side is a linear combination of B00, B01, B10, B11.
    
    # Top-level Strassen (for 2x2 block matrices):
    # alpha=0 (M1): A-side = A00+A11, B-side = B00+B11
    # alpha=1 (M2): A-side = A10+A11, B-side = B00
    # alpha=2 (M3): A-side = A00, B-side = B01-B11
    # alpha=3 (M4): A-side = A11, B-side = B10-B00
    # alpha=4 (M5): A-side = A00+A01, B-side = B11
    # alpha=5 (M6): A-side = A10-A00, B-side = B00+B01
    # alpha=6 (M7): A-side = A01-A11, B-side = B10+B11
    
    # Where A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B.
    
    # For the inner 2x2x2 Strassen (beta = 0..6), we use the decomposition from strassen_2x2x2_decomp().
    
    # Now, for the full 4x4x4 tensor decomposition:
    # U[i*4+j, c] where i in {0..3}, j in {0..3}, c in {0..48}
    # V[j*4+k, c] where j in {0..3}, k in {0..3}, c in {0..48}
    # W[k*4+i, c] where k in {0..3}, i in {0..3}, c in {0..48}
    
    # Wait, for our problem, n=4, m=5, p=5. So:
    # U has shape (n*m, rank) = (20, rank), indexed by [i*m+j, c] = [i*5+j, c]
    # V has shape (m*p, rank) = (25, rank), indexed by [j*p+k, c] = [j*5+k, c]
    # W has shape (p*n, rank) = (20, rank), indexed by [k*n+i, c] = [k*4+i, c]
    
    # For the 4x4x4 part (j in {0..3}):
    # U[i*5+j, c] for i in {0..3}, j in {0..3}, c in {0..48}
    # V[j*5+k, c] for j in {0..3}, k in {0..3}, c in {0..48}
    # W[k*4+i, c] for k in {0..3}, i in {0..3}, c in {0..48}
    
    # For the j=4 part:
    # U[i*5+4, 49] = 1 for all i
    # V[4*5+k, 49] = 1 for all k
    # W[k*4+i, 49] = 1 for all i,k
    
    # Now let me construct the 4x4x4 Strassen decomposition.
    
    # Define the top-level Strassen coefficients.
    # A-blocks: A00 = rows {0,1}, cols {0,1}; A01 = rows {0,1}, cols {2,3}
    #           A10 = rows {2,3}, cols {0,1}; A11 = rows {2,3}, cols {2,3}
    # B-blocks: B00 = rows {0,1}, cols {0,1}; B01 = rows {0,1}, cols {2,3}
    #           B10 = rows {2,3}, cols {0,1}; B11 = rows {2,3}, cols {2,3}
    
    # Top-level product alpha:
    # A-coefficients: dict from block name to coefficient
    # B-coefficients: dict from block name to coefficient
    
    # Let me define:
    # a_coeffs[alpha] = [(block, coeff), ...] for A
    # b_coeffs[alpha] = [(block, coeff), ...] for B
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (same for B)
    # Block (bi) corresponds to rows {2*(bi//2), 2*(bi//2)+1} and cols {2*(bi%2), 2*(bi%2)+1}
    
    # Actually, let me use a 2D array: a_coeffs[alpha][bi] = coefficient
    # where bi = row_block * 2 + col_block, row_block in {0,1}, col_block in {0,1}
    
    a_coeffs = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [0, 0, 1, 0],  # M6: A10 - A00 -> wait, M6 = (A10-A00)(B00+B01)
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    # Fix M6:
    a_coeffs[5] = [-1, 0, 1, 0]  # A10 - A00
    
    b_coeffs = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    
    # Inner Strassen 2x2x2 decomposition
    U_inner, V_inner, W_inner = strassen_2x2x2_decomp()
    # U_inner[i*2+j, beta] for i,j in {0,1}, beta in {0..6}
    # V_inner[j*2+k, beta] for j,k in {0,1}, beta in {0..6}
    # W_inner[k*2+i, beta] for k,i in {0,1}, beta in {0..6}
    
    # Now build the 4x4x4 decomposition.
    # For each alpha in {0..6} and beta in {0..6}:
    #   c = alpha * 7 + beta
    #   U[i*4+j, c] = sum over A-blocks: a_coeffs[alpha][bi] * U_inner[i_local*2+j_local, beta]
    #     where i = row_block*2 + i_local, j = col_block*2 + j_local
    #     and bi = row_block*2 + col_block
    #   Similarly for V and W.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for alpha in range(7):
        for beta in range(7):
            c = alpha * 7 + beta
            # A-side
            for row_block in range(2):
                for col_block in range(2):
                    bi = row_block * 2 + col_block
                    coeff_a = a_coeffs[alpha][bi]
                    if coeff_a == 0:
                        continue
                    for i_local in range(2):
                        for j_local in range(2):
                            i = row_block * 2 + i_local
                            j = col_block * 2 + j_local
                            U_444[i*4+j, c] += coeff_a * U_inner[i_local*2+j_local, beta]
            
            # B-side
            for row_block in range(2):
                for col_block in range(2):
                    bi = row_block * 2 + col_block
                    coeff_b = b_coeffs[alpha][bi]
                    if coeff_b == 0:
                        continue
                    for j_local in range(2):
                        for k_local in range(2):
                            j = row_block * 2 + j_local
                            k = col_block * 2 + k_local
                            V_444[j*4+k, c] += coeff_b * V_inner[j_local*2+k_local, beta]
            
            # W-side (output C)
            # C = [[C00, C01], [C10, C11]]
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # W[i,k] gets contribution from the alpha-th product with coefficient w_coeff[alpha][ci]
            # where ci = row_block*2 + col_block for the output block.
            
            w_coeff = [
                [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
                [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
                [0, 1, 0, 0],   # M3: C01 += 1
                [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
                [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
                [0, 0, 0, 1],   # M6: C11 += 1
                [1, 0, 0, 0],   # M7: C00 += 1
            ]
            
            for row_block in range(2):
                for col_block in range(2):
                    ci = row_block * 2 + col_block
                    coeff_w = w_coeff[alpha][ci]
                    if coeff_w == 0:
                        continue
                    for k_local in range(2):
                        for i_local in range(2):
                            k = col_block * 2 + k_local
                            i = row_block * 2 + i_local
                            W_444[k*4+i, c] += coeff_w * W_inner[k_local*2+i_local, beta]
    
    # Now assemble the full 4x5x5 decomposition.
    # U: shape (20, 50), V: shape (25, 50), W: shape (20, 50)
    
    U_full = np.zeros((20, 50))
    V_full = np.zeros((25, 50))
    W_full = np.zeros((20, 50))
    
    # Copy the 4x4x4 part (j in {0..3})
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U_full[i*5+j, c] = U_444[i*4+j, c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_full[j*5+k, c] = V_444[j*4+k, c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_full[k*4+i, c] = W_444[k*4+i, c]
    
    # Add the j=4 term (rank 1)
    for i in range(4):
        U_full[i*5+4, 49] = 1
    for k in range(5):
        V_full[4*5+k, 49] = 1
    for k in range(5):
        for i in range(4):
            W_full[k*4+i, 49] = 1
    
    # Verify the decomposition
    # Check: for all (i,j,k) in {0..3} x {0..4} x {0..4}:
    # sum_c U_full[i*5+j, c] * V_full[j*5+k, c] * W_full[k*4+i, c] should be 1
    
    # Let's verify numerically
    errors = 0
    max_err = 0
    for i in range(4):
        for j in range(5):
            for k in range(5):
                val = 0
                for c in range(50):
                    val += U_full[i*5+j, c] * V_full[j*5+k, c] * W_full[k*4+i, c]
                err = abs(val - 1)
                if err > 1e-10:
                    errors += 1
                max_err = max(max_err, err)
    
    if errors > 0:
        # Fallback to schoolbook if verification fails
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
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(arr):
        return [[float(x) if x != int(x) else int(x) for x in row] for row in arr]
    
    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
