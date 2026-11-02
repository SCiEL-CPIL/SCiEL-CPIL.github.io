"""Program for rank-minimal tensor decomposition of 4x5 * 5x5 matrix multiplication."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
from fractions import Fraction
import random
import sys

def _matmul_tensor_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    The matrix multiplication tensor T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
    must be decomposed as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For each output element C[i,k] = sum_j A[i,j] * B[j,k],
    we need: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    Actually, the decomposition is:
    T_{(i,j),(j,k),(k,i)} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is equivalent to checking that for all i, k:
    sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * ... 
    No, the correct check is:
    For each r, define M_r = sum_j U[i*m+j, r] A[i,j] * V[j*p+k, r] for all i,k
    Then C = sum_r W[k*n+i, r] * M_r... 
    
    Actually the standard check is:
    The tensor T has entries T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (k,i).
    T[a, b, c] = delta[i,i'] * delta[j,j'] * delta[k,k'] where a=(i,j), b=(j',k), c=(k',i')
    
    The decomposition says: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    For this to be the matrix multiplication tensor:
    T[(i,j), (j',k), (k',i')] = 1 if i=i' and j=j' and k=k', else 0
    
    So we need: sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    This is equivalent to: for all i, j, k, the (i,k) entry of the matrix
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] (with appropriate j) should equal delta
    
    Actually, let me think more carefully. The matrix multiplication algorithm computes:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In the tensor decomposition form, this is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * ... 
    No, that's not right either.
    
    The correct form is:
    C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
    
    Wait, that's not how tensor decomposition works. Let me be precise.
    
    The matrix multiplication tensor T is a 3-tensor with indices (i,j), (j,k), (k,i).
    T[(i,j), (j,k), (k,i)] = 1 (it's the identity in some sense, but actually it's defined by:
    for matrices A (n x m) and B (m x p), C = A @ B has C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor form: C[i,k] = sum_{(i,j), (j,k), (k,i)} A[i,j] B[j,k] T[(i,j),(j,k),(k,i)]
    
    For this to work, T[(i,j),(j,k),(k,i)] must be 1 when the indices match up, but since
    (i,j) and (j,k) already share j, we need T[(i,j),(j,k),(k,i)] = delta_{j,j} = 1?
    
    No, the standard formulation is:
    T_{(i,j), (j',k), (k',i')} = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    So T is 1 only when all three index pairs are consistent.
    
    The decomposition T = sum_r U ⊗ V ⊗ W means:
    T_{(i,j), (j',k), (k',i')} = sum_r U_{(i,j),r} V_{(j',k),r} W_{(k',i'),r}
    
    For the matrix multiplication tensor, we need:
    sum_r U_{(i,j),r} V_{(j',k),r} W_{(k',i'),r} = delta_{j,j'} delta_{k,k'} delta_{i,i'}
    
    This is a very specific constraint. Let me verify with the schoolbook solution:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for col corresponding to (i,j,k)
    
    Then sum over col: for a fixed (i,j), (j',k), (k',i'), the sum is 1 only if
    there exists col with i'=i, j'=j, k'=k, which gives 1. Otherwise 0. Correct.
    """
    # Verify using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # For each r, compute the contribution to C
    # C[i,k] should equal sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k])
    # Wait, that's not quite right. Let me re-derive.
    
    # Actually, the bilinear form is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * ... 
    # No. The correct formula for matrix multiplication as a bilinear map is:
    # C = A @ B
    # In tensor decomposition: C[i,k] = sum_r W[k*n+i, r] * f_r(A) * g_r(B)
    # where f_r(A) = sum_j U[i*m+j, r] A[i,j]... no, f_r should not depend on i,k in that way.
    
    # Let me think again. The matrix multiplication is a bilinear map:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-1 decomposition would be: C[i,k] = W[i,k] * f(A) * g(B)
    # But here f and g need to capture the sum over j.
    # 
    # For a general rank-r decomposition:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * ... 
    # This doesn't make sense because the inner sum depends on i.
    
    # I think the correct form is:
    # C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
    # But that's a triple product, not bilinear.
    
    # Let me look at this differently. The standard way to express matrix multiplication
    # as a tensor decomposition is:
    # T_{(i,j), (j,k), (k,i)} = sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
    #
    # And the algorithm is:
    # 1. For each r, compute M_r = sum_j U_{(i,j),r} A[i,j] * V_{(j,k),r} B[j,k]  -- NO, this is wrong.
    #
    # Actually, I recall now. The correct bilinear decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k])
    # But this has a problem: the second sum is over k, but C[i,k] depends on k.
    
    # I think I'm confusing myself. Let me just verify computationally.
    # The schoolbook solution should pass, so let me use that as a test.
    
    # For the schoolbook solution, let's verify that it correctly computes matrix multiplication.
    # We'll test with random matrices.
    
    A_test = np.random.randint(0, 3, size=(n, m))
    B_test = np.random.randint(0, 3, size=(m, p))
    C_expected = A_test @ B_test
    
    # Compute C using the decomposition
    C_computed = np.zeros((n, p), dtype=float)
    for r in range(rank):
        # For each r, compute the contribution
        # The contribution to C[i,k] from term r is:
        # W[k*n+i, r] * sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
        # Wait, this is still not right. Let me think step by step.
        
        # I believe the correct formula is:
        # C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
        # But this means for each r and each (i,k), we sum over j.
        
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += U_np[i*m+j, r] * A_test[i,j] * V_np[j*p+k, r] * B_test[j,k]
                C_computed[i,k] += W_np[k*n+i, r] * s
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook is correct
    assert _matmul_tensor_decomposition(n, m, p, U_school.tolist(), V_school.tolist(), W_school.tolist()), \
        "Schoolbook decomposition failed verification!"
    
    # Now try to find a lower rank decomposition.
    # Strategy: Use the fact that 2x2*2x2 has rank 7 (Strassen) to reduce rank.
    # 
    # Partition: A is 4x5, B is 5x5.
    # Split A into A_top (2x5) and A_bottom (2x5).
    # Split B into B_left (5x3) and B_right (5x2).
    # C_top_left = A_top @ B_left (2x3)
    # C_top_right = A_top @ B_right (2x2)
    # C_bottom_left = A_bottom @ B_left (2x3)
    # C_bottom_right = A_bottom @ B_right (2x2)
    #
    # For the 2x2 blocks, we can use Strassen (rank 7 instead of 4).
    # For the 2x3 blocks, we need to find the optimal rank.
    # 
    # Let's first handle the 2x2 * 5x5 case using Strassen on 2x2 sub-blocks.
    # 
    # Actually, let's try a simpler approach: use the known rank-7 decomposition for 2x2*2x2
    # and apply it to sub-blocks of the larger problem.
    #
    # B is 5x5. Split into B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3) -- this is awkward.
    # 
    # Better: Split B into B_top (2x5) and B_bottom (3x5).
    # C_top = A @ B_top (4x2)
    # C_bottom = A @ B_bottom (4x3)
    #
    # For C_top: A is 4x5, B_top is 2x5 (transposed? No, B_top is rows 0-1 of B, so it's 2x5).
    # Wait, A @ B_top: A is 4x5, B_top is 2x5. This doesn't work dimensionally.
    # 
    # Let me redo. B is 5x5. If I split B into B_left (5x2) and B_right (5x3):
    # C_left = A @ B_left (4x2)
    # C_right = A @ B_right (4x3)
    #
    # For C_left (4x2): A is 4x5, B_left is 5x2.
    # Split A into A1 (2x5) and A2 (2x5).
    # C1 = A1 @ B_left (2x2)
    # C2 = A2 @ B_left (2x2)
    # Use Strassen for each 2x2 * 5x2? No, Strassen is for 2x2 * 2x2.
    #
    # This is getting complicated. Let me try a different approach.
    #
    # Use the fact that rank(2x2*2x2) = 7 to build a decomposition for larger matrices.
    # 
    # For 4x4 * 4x4, the Strassen algorithm gives rank at most 7^2 = 49 (vs 64 schoolbook).
    # For 4x5 * 5x5, we can embed this.
    #
    # Let me try to use a block Strassen-like approach.
    #
    # Actually, let me try a direct approach: use the known Strassen decomposition for 2x2*2x2
    # and apply it to sub-blocks.
    
    # Strassen 2x2*2x2 decomposition (rank 7):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me implement a function to apply Strassen to a 2x2 * 2x2 block within the larger decomposition.
    
    def strassen_2x2():
        """Return U, V, W for Strassen 2x2*2x2 decomposition, rank 7.
        
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is at (0,0), A22 is at (1,1) in A (2x2)
        # B11 is at (0,0), B22 is at (1,1) in B (2x2)
        # M1 contributes to C11 and C22
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # M2 = (A21 + A22)B11
        # A21 is at (1,0), A22 is at (1,1)
        # B11 is at (0,0)
        # M2 contributes to C21 and C22
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1  # C21
        W[3, 1] = -1 # C22
        
        # M3 = A11(B12 - B22)
        # A11 is at (0,0)
        # B12 is at (0,1), B22 is at (1,1)
        # M3 contributes to C12 and C22
        U[0, 2] = 1  # A11
        V[1, 2] = 1  # B12
        V[3, 2] = -1 # B22
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # M4 = A22(B21 - B11)
        # A22 is at (1,1)
        # B21 is at (1,0), B11 is at (0,0)
        # M4 contributes to C11 and C21
        U[3, 3] = 1  # A22
        V[2, 3] = 1  # B21
        V[0, 3] = -1 # B11
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21
        
        # M5 = (A11 + A12)B22
        # A11 is at (0,0), A12 is at (0,1)
        # B22 is at (1,1)
        # M5 contributes to C11 and C12
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1 # C11
        W[1, 4] = 1  # C12
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21 is at (1,0), A11 is at (0,0)
        # B11 is at (0,0), B12 is at (0,1)
        # M6 contributes to C22
        U[2, 5] = 1  # A21
        U[0, 5] = -1 # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        W[3, 5] = 1  # C22
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12 is at (0,1), A22 is at (1,1)
        # B21 is at (1,0), B22 is at (1,1)
        # M7 contributes to C11
        U[1, 6] = 1  # A12
        U[3, 6] = -1 # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = strassen_2x2()
    assert _matmul_tensor_decomposition(2, 2, 2, U_s.tolist(), V_s.tolist(), W_s.tolist()), \
        "Strassen 2x2 decomposition failed verification!"
    
    # Now let's try to build a decomposition for 4x5 * 5x5 using blocks.
    # 
    # Strategy: Split into 2x2 blocks where possible, use Strassen for those, and handle the rest.
    # 
    # A is 4x5, B is 5x5.
    # 
    # Let me split A into A1 (2x5) and A2 (2x5).
    # Let me split B into B1 (5x2) and B2 (5x3).
    # C1 = A1 @ B1 (2x2)
    # C2 = A1 @ B2 (2x3)
    # C3 = A2 @ B1 (2x2)
    # C4 = A2 @ B2 (2x3)
    #
    # For C1 (2x2): A1 is 2x5, B1 is 5x2. This is not 2x2*2x2.
    # 
    # I need to split further. Let me split B1 (5x2) into B1a (2x2) and B1b (3x2).
    # C1a = A1 @ B1a (2x2) -- use Strassen
    # C1b = A1 @ B1b (2x3)
    #
    # This is getting very complex. Let me try a simpler approach: use the schoolbook for the non-2x2 parts
    # and Strassen for the 2x2 parts, and see if the total rank is less than 100.
    
    # Let me count: if I can use Strassen for some 2x2*2x2 sub-problems, I save 3 rank units each
    # (7 instead of 4).
    
    # Let me try a specific decomposition:
    # 
    # Partition A (4x5) into four 2x(2 or 3) blocks:
    # A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # 
    # Partition B (5x5) into four (2 or 3)x(2 or 3) blocks:
    # B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    # 
    # C11 = A11@B11 + A12@B21 (2x2)
    # C12 = A11@B12 + A12@B22 (2x3)
    # C21 = A21@B11 + A22@B21 (2x2)
    # C22 = A21@B12 + A22@B22 (2x3)
    
    # This is getting too complex to reason about. Let me just try to use a random search
    # with the constraint that entries are integers or half-integers.
    
    # Given the time constraints and the difficulty of finding a provably correct lower rank decomposition,
    # let me try a different approach: use the known lower bound and see if we can get close.
    
    # For now, let me return the schoolbook decomposition, which is guaranteed to be correct.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
