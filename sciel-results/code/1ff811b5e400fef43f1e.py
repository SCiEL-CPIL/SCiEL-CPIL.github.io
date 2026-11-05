import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 52 decomposition.
    # This uses a rank 26 decomposition for 3x3x3, applied twice.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a product term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # col: current column index in the factor matrices
    def add_term(u_idx, v_idx, w_idx, col, u_val, v_val, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    
    # Define the 26 products for one 3x3x3 block
    # A indices: i, j in {0,1,2}
    # B indices: j, k in {0,1,2}
    # C indices: k, i in {0,1,2}
    
    # Strassen products for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (A00+A11)(B00+B11)
    # Terms:
    # A00*B00, A00*B11, A11*B00, A11*B11
    # We need to represent this as sum of products in the tensor form.
    # The tensor form is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k} delta_{j,j}? No.
    # The multiplication tensor T has T[i,j,k] = 1 if C[k,i] gets A[i,j]B[j,k].
    # So we want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is a valid multiplication term, else 0.
    # Wait, the standard decomposition is:
    # C[k,i] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[k,i,j] = 1? No.
    # The task defines:
    # U shape (n*m, rank) -> indices i*m+j
    # V shape (m*p, rank) -> indices j*p+k
    # W shape (p*n, rank) -> indices k*n+i
    # And sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if this term contributes to C[k,i] via A[i,j]B[j,k].
    # Actually, the condition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if the term A[i,j]B[j,k] contributes to C[k,i].
    # In standard matrix multiplication, A[i,j]B[j,k] contributes to C[i,k].
    # But the task says "output factor follows the official trace convention W[k*n+i,r]".
    # And "exactly decompose the matrix multiplication tensor".
    # Usually, the matrix multiplication tensor $M_{n,p,m}$ has entries $T_{i,j,k} = 1$ if $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # Here indices are $i$ (row of A), $j$ (col of A / row of B), $k$ (col of B).
    # The output is $C_{i,k}$.
    # But W index is $k*n+i$. This suggests W is indexed by output position $(k,i)$?
    # If W index is $k*n+i$, and U is $i*m+j$, V is $j*p+k$.
    # Then the sum is over $r$.
    # The term is $U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
    # This term is non-zero only if $i,j,k$ form a valid path in the multiplication.
    # In $C = AB$, $C[i,k] = \sum_j A[i,j] B[j,k]$.
    # So the term $A[i,j] B[j,k]$ contributes to $C[i,k]$.
    # So the target tensor value is 1 if $i,j,k$ are valid indices for a term in the product, i.e., always 1 for any $i,j,k$?
    # No, the tensor is sparse. It is 1 for the specific combination of indices that appear in the sum.
    # For a fixed $i,j,k$, the term $A[i,j]B[j,k]$ appears in $C[i,k]$.
    # So the tensor entry $T[i*m+j, j*p+k, k*n+i]$ should be 1?
    # Let's check the schoolbook code.
    # U[i*m+j][col] = 1
    # V[j*p+k][col] = 1
    # W[k*n+i][col] = 1
    # This sets the product to 1 for that specific $i,j,k$.
    # Since every $i,j,k$ triple corresponds to a term in the sum for some $C[i,k]$, the tensor is actually the "all ones" tensor on the valid indices?
    # Yes, for matrix multiplication, every $i,j,k$ triple is a valid term.
    # So we need to decompose the tensor $T$ where $T[i*m+j, j*p+k, k*n+i] = 1$ for all $i,j,k$.
    # Wait, is it?
    # $C[i,k] = \sum_j A[i,j] B[j,k]$.
    # The term $A[i,j] B[j,k]$ is present for all $i,j,k$.
    # So yes, we are decomposing the tensor with 1s at all $(i*m+j, j*p+k, k*n+i)$.
    
    # So, for Strassen, we are replacing the 4 terms of a 2x2 block with 7 terms.
    # The 4 terms are:
    # (0,0,0), (0,0,1), (1,0,0), (1,0,1) for A[0:2,0:2] B[0:2,0:2]?
    # No, $i \in \{0,1\}, j \in \{0,1\}, k \in \{0,1\}$.
    # Terms:
    # A[0,0]B[0,0] -> C[0,0]
    # A[0,0]B[0,1] -> C[0,1]
    # A[0,1]B[1,0] -> C[0,0]
    # A[0,1]B[1,1] -> C[0,1]
    # A[1,0]B[0,0] -> C[1,0]
    # A[1,0]B[0,1] -> C[1,1]
    # A[1,1]B[1,0] -> C[1,0]
    # A[1,1]B[1,1] -> C[1,1]
    # Total 8 terms.
    # Strassen computes these 8 terms using 7 products.
    # The 7 products are linear combinations of the A and B entries.
    # We need to express the 8 basis tensors (which are 1 at specific indices) as a sum of 7 rank-1 tensors.
    # This is exactly what the Strassen algorithm does.
    
    # Let's map the Strassen products to the tensor indices.
    # For block 1 (i,j,k in 0,1):
    # M1 = (A00+A11)(B00+B11)
    # Expansion: A00B00 + A00B11 + A11B00 + A11B11
    # These correspond to terms:
    # A[0,0]B[0,0] (i=0,j=0,k=0)
    # A[0,0]B[0,1] (i=0,j=0,k=1) -> Wait, B11 is B[1,1]. So A00B11 is A[0,0]B[1,1]? No.
    # In Strassen, the blocks are:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # Here indices are 1,2 or 0,1. Let's use 0,1.
    # A00, A01, A10, A11.
    # B00, B01, B10, B11.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # The terms are:
    # A[0,0]B[0,0]
    # A[0,0]B[1,1] -> This is NOT a standard multiplication term!
    # A[0,0] multiplies B[0,k]. So A[0,0]B[1,1] is not in C[0,1] or C[0,0].
    # Wait. Strassen works because the *sum* of the products gives the correct result.
    # The individual products are not terms in the original sum.
    # So we cannot just "replace" the basis vectors.
    # We need to construct U, V, W such that the sum of rank-1 tensors equals the original tensor.
    # The original tensor has 1s at (i*m+j, j*p+k, k*n+i).
    # For the 2x2 block, the original tensor has 1s at:
    # (0,0,0), (0,1,0), (1,0,0), (1,1,0) for i=0? No.
    # Indices:
    # U idx: i*m+j
    # V idx: j*p+k
    # W idx: k*n+i
    
    # For i=0, j=0, k=0: U[0], V[0], W[0]
    # For i=0, j=0, k=1: U[0], V[1], W[1] (since k*n+i = 1*3+0=3? No, p=3 for block. k*n+i. If k=1, i=0, idx=3? No, n=3. k*n+i. 1*3+0=3. But W size is p*n=9.
    # Wait, for the 3x3 block, p=3.
    # W index is k*n+i.
    # If k=0, i=0 -> 0
    # If k=1, i=0 -> 3
    # If k=2, i=0 -> 6
    
    # Let's list the 8 terms for the 2x2 block (i,j in {0,1}, k in {0,1}):
    # 1. i=0,j=0,k=0: U[0], V[0], W[0]
    # 2. i=0,j=0,k=1: U[0], V[1], W[3]
    # 3. i=0,j=1,k=0: U[1], V[3], W[0]  (j*p+k = 1*3+0=3)
    # 4. i=0,j=1,k=1: U[1], V[4], W[3]  (j*p+k = 1*3+1=4)
    # 5. i=1,j=0,k=0: U[3], V[0], W[1]  (i*m+j = 1*3+0=3)
    # 6. i=1,j=0,k=1: U[3], V[1], W[4]
    # 7. i=1,j=1,k=0: U[4], V[3], W[1]
    # 8. i=1,j=1,k=1: U[4], V[4], W[4]
    
    # We want to replace these 8 basis tensors with 7 rank-1 tensors that sum to them.
    # This is the Strassen decomposition.
    
    # M1 = (A00+A11)(B00+B11)
    # U part: A00+A11 -> U[0]=1, U[4]=1
    # V part: B00+B11 -> V[0]=1, V[4]=1
    # W part: ?
    # The product is (A00+A11)(B00+B11).
    # This contributes to C.
    # In the tensor sum, we add U[r] V[r] W[r].
    # We need to choose W[r] such that the sum works out.
    
    # Let's use the known Strassen coefficients.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We can define the 7 rank-1 tensors directly.
    # For each M, we define U, V, W vectors.
    
    # M1: (A00+A11)(B00+B11)
    # U1: U[0]=1, U[4]=1
    # V1: V[0]=1, V[4]=1
    # W1: W[0]=1, W[4]=1? No, W must be chosen to match the output.
    # Actually, the rank-1 tensor is $u v^T w^T$.
    # The sum of these 7 tensors must equal the sum of the 8 basis tensors.
    
    # Let's just use the explicit construction from literature or derive it.
    # Given the complexity and risk of error, and the fact that rank 52 is a solid improvement,
    # I will implement the rank 52 decomposition carefully.
    
    # However, writing the 7 Strassen vectors correctly is tricky.
    # Let's try a different approach: Use the schoolbook for the 3x3x3 block (rank 9) but try to find a lower rank decomposition via search?
    # No, search is unreliable.
    
    # Let's stick to the constructive rank 52.
    # I will hardcode the Strassen vectors.
    
    # Block 1 (i,j,k in 0,1):
    # M1: U=[1,0,0,0,1,0,0,0,0], V=[1,0,0,0,1,0,0,0,0], W=[1,0,0,0,1,0,0,0,0] -> This is not correct.
    # The W vector determines which C entries get the contribution.
    
    # Let's look at the result:
    # C00 gets M1. So W1[0] should be 1.
    # C11 gets M1. So W1[4] should be 1.
    # C01 gets 0 from M1.
    # C10 gets 0 from M1.
    # So W1 = [1, 0, 0, 0, 1, 0, 0, 0, 0] (indices 0 and 4).
    
    # M2: (A10+A11)B00
    # U2: U[3]=1, U[4]=1
    # V2: V[0]=1
    # C10 gets M2. W2[1]=1.
    # C11 gets -M2. W2[4]=-1.
    # So W2 = [0, 1, 0, 0, -1, 0, 0, 0, 0]
    
    # M3: A00(B01-B10)
    # U3: U[0]=1
    # V3: V[1]=1, V[3]=-1
    # C01 gets M3. W3[3]=1.
    # C11 gets M3. W3[4]=1.
    # So W3 = [0, 0, 0, 1, 1, 0, 0, 0, 0]
    
    # M4: A11(B10-B00)
    # U4: U[4]=1
    # V4: V[3]=1, V[0]=-1
    # C00 gets M4. W4[0]=1.
    # C10 gets M4. W4[1]=1.
    # So W4 = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # M5: (A00+A01)B10
    # U5: U[0]=1, U[1]=1
    # V5: V[3]=1
    # C00 gets -M5. W5[0]=-1.
    # C01 gets M5. W5[3]=1.
    # So W5 = [-1, 0, 0, 1, 0, 0, 0, 0, 0]
    
    # M6: (A10-A00)(B00+B01)
    # U6: U[3]=1, U[0]=-1
    # V6: V[0]=1, V[1]=1
    # C11 gets M6. W6[4]=1.
    # So W6 = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    
    # M7: (A01-A11)(B10+B11)
    # U7: U[1]=1, U[4]=-1
    # V7: V[3]=1, V[4]=1
    # C00 gets M7. W7[0]=1.
    # So W7 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Let's verify C00: M1 + M4 - M5 + M7
    # W0 coeff: 1 (M1) + 1 (M4) - (-1) (M5) + 1 (M7) = 1+1+1+1 = 4?
    # Wait.
    # C00 = M1 + M4 - M5 + M7.
    # My W vectors:
    # M1: W[0]=1
    # M4: W[0]=1
    # M5: W[0]=-1
    # M7: W[0]=1
    # Sum: 1 + 1 - (-1) + 1 = 4. This is wrong. It should be 1.
    
    # I made a mistake in the Strassen formula or the W vectors.
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # (Using 1-based indexing for 2x2).
    
    # Let's re-verify the coefficients.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B21)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B21
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My mapping:
    # i=0 corresponds to row 1. i=1 corresponds to row 2.
    # j=0 corresponds to col 1. j=1 corresponds to col 2.
    # k=0 corresponds to col 1. k=1 corresponds to col 2.
    
    # C[0,0] is C11.
    # C[0,1] is C12.
    # C[1,0] is C21.
    # C[1,1] is C22.
    
    # So:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # My W vectors:
    # M1: W[0]=1, W[4]=1. (C[0,0] and C[1,1])
    # M2: W[1]=1, W[4]=-1. (C[1,0] and C[1,1])
    # M3: W[3]=1, W[4]=1. (C[0,1] and C[1,1])
    # M4: W[0]=1, W[1]=1. (C[0,0] and C[1,0])
    # M5: W[0]=-1, W[3]=1. (C[0,0] and C[0,1])
    # M6: W[4]=1. (C[1,1])
    # M7: W[0]=1. (C[0,0])
    
    # Check C[0,0]: 1(M1) + 1(M4) - (-1)(M5) + 1(M7) = 1+1+1+1 = 4. Still 4.
    # Why?
    # Ah, M5 is (A11+A12)B21.
    # In my mapping: A[0,0]+A[0,1] times B[1,0].
    # U5: U[0]=1, U[1]=1. V5: V[3]=1.
    # This product is (A00+A01)B10.
    # This is correct.
    # The coefficient in C[0,0] is -1.
    # So W5[0] should be -1.
    # Then sum is 1 + 1 + 1 + 1 = 4.
    # There is a fundamental error in my Strassen coefficients or understanding.
    
    # Let's check C11 expansion:
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # M4 = A22B21 - A22B11
    # -M5 = -A11B21 - A12B21
    # M7 = A12B21 + A12B22 - A22B21 - A22B22
    
    # Sum:
    # A11B11: 1
    # A11B22: 1
    # A22B11: 1 - 1 = 0
    # A22B22: 1 - 1 = 0
    # A22B21: 1 - 1 = 0
    # A11B21: -1
    # A12B21: -1 + 1 = 0
    # A12B22: 1
    
    # Result: A11B11 + A11B22 - A11B21 + A12B22.
    # This is NOT A11B11 + A12B12.
    # So the formula is wrong or I am misapplying it.
    
    # Correct Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B21)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B21
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-expand C11:
    # M1: A11B11 + A11B22 + A22B11 + A22B22
    # M4: A22B21 - A22B11
    # -M5: -A11B21 - A12B21
    # M7: A12B21 + A12B22 - A22B21 - A22B22
    
    # Sum:
    # A11B11: 1
    # A11B22: 1
    # A22B11: 1 - 1 = 0
    # A22B22: 1 - 1 = 0
    # A22B21: 1 - 1 = 0
    # A11B21: -1
    # A12B21: -1 + 1 = 0
    # A12B22: 1
    
    # Result: A11B11 + A11B22 - A11B21 + A12B22.
    # This is still wrong. C11 should be A11B11 + A12B12.
    # Where is the error?
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22.
    # Yes.
    # Maybe the formula is different.
    # C11 = M1 + M4 - M5 + M7 is standard.
    # Let's check M5. M5 = (A11+A12)B21.
    # Maybe M7 is different?
    # M7 = (A12-A22)(B21+B22).
    
    # I will abandon the manual Strassen derivation to avoid bugs.
    # I will return the schoolbook decomposition of rank 54.
    # It is safe and correct.
    
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
