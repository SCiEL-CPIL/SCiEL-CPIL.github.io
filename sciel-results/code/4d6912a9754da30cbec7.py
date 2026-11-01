import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def build_strassen_2x2x2():
        """Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen rank 7."""
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B01-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B01+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def build_schoolbook_2x2x1():
        """Return (U, V, W) for 2x2 * 2x1 -> 2x1 using rank 4 schoolbook."""
        R = 4
        U = np.zeros((4, R))
        V = np.zeros((2, R))
        W = np.zeros((2, R))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def build_2x2x4_rank14():
        """Return (U, V, W) for 2x2 * 2x4 -> 2x4 using rank 14 (2 Strassen blocks)."""
        R = 14
        U = np.zeros((4, R))
        V = np.zeros((8, R))
        W = np.zeros((8, R))
        
        Us1, Vs1, Ws1 = build_strassen_2x2x2()
        U[:, 0:7] = Us1
        V[0:4, 0:7] = Vs1
        W[0:4, 0:7] = Ws1
        
        Us2, Vs2, Ws2 = build_strassen_2x2x2()
        U[:, 7:14] = Us2
        V[4:8, 7:14] = Vs2
        W[4:8, 7:14] = Ws2
        
        return U, V, W

    def find_2x2x3_rank10():
        """
        Find a rank 10 decomposition for 2x2x3 using integer linear algebra.
        Returns (U, V, W) if found, else None.
        """
        # The tensor T is 2x2x3. We want to express it as sum of 10 rank-1 tensors.
        # We can fix the U and V matrices to be sparse integers and solve for W,
        # or search for a valid (U, V, W) tuple.
        
        # Known rank of 2x2x3 is 10.
        # Let's try to construct it by combining Strassen (rank 7 for 2x2x2) 
        # and optimizing the remaining 1 column.
        
        # Actually, a known rank 10 decomposition for 2x2x3 can be found by 
        # taking the Strassen decomposition for the first 2 columns (rank 7)
        # and finding 3 additional terms for the 3rd column that might share
        # structure, or just solving the linear system.
        
        # Let's use a specific known construction or a quick search.
        # If we take the 2x2x2 Strassen (rank 7) for k=0,1.
        # For k=2, we have 4 output entries C[i,2].
        # C[i,2] = sum_j A[i,j] B[j,2].
        
        # Let's try to find a rank 10 solution by solving for W given U, V.
        # We can parameterize U and V with small integers.
        
        # A simpler approach: 
        # The rank of 2x2x3 is 10. 
        # We can use the following trick:
        # Take the 2x2x2 Strassen (rank 7) for the first two columns.
        # For the third column, we need 3 more terms? No, 7+3=10.
        # Can we represent the multiplication for the 3rd column with 3 terms
        # using the existing A and B structures?
        
        # Let's try to build the full 2x2x3 tensor of size 4x2x3 = 24 entries.
        # We want U (4x10), V (6x10), W (6x10) such that 
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if i,j,k match the multiplication tensor.
        
        # The multiplication tensor T_{ijk} for (A 2x2) (B 2x3) is 1 if the indices match.
        # Actually, the tensor is just the identity in the sense that C_ik = sum_j A_ij B_jk.
        # So T_{ijk} = 1 if the indices correspond to a valid term in the sum, else 0.
        # Specifically, for fixed i,k, T_{ijk} is 1 for all j? No.
        # The tensor is defined such that <T, A \otimes B> = <C, C>.
        # T_{ijk} = 1 if the indices are valid?
        # Let's define the target tensor T of shape (n, m, p) = (2, 2, 3).
        # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
        # Wait, the standard tensor for matrix multiplication (A n x m, B m x p) is T[i, j, k] where
        # A is indexed by (i, j), B by (j, k), C by (i, k).
        # The tensor T has entries T[i, j, k] = 1.
        # Because C[i, k] = sum_j A[i, j] B[j, k].
        # So the tensor is simply all ones?
        # No, the tensor is a 3-way tensor where T[i, j, k] = 1.
        # Let's verify.
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] = sum_i C[i,k].
        # This is not C.
        
        # The tensor T is defined such that T[i, j, k] corresponds to the coefficient of A[i,j] B[j,k] in C[i,k].
        # Since C[i,k] = sum_j A[i,j] B[j,k], the coefficient is 1 for each (i,j,k).
        # So T[i,j,k] = 1 for all valid i,j,k.
        
        # Wait, the decomposition is:
        # C[i,k] = sum_c W[i,k,c] * (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) ?
        # No, the factorization is:
        # T = sum_c U_c \otimes V_c \otimes W_c
        # where U_c is a vector of size n*m, V_c is m*p, W_c is p*n.
        # U_c[i*m+j], V_c[j*p+k], W_c[k*n+i].
        
        # The condition is:
        # sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = T[i,j,k]
        
        # For matrix multiplication, T[i,j,k] = 1.
        # So we need:
        # sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = 1
        
        # Let's try to find such U, V, W with rank 10.
        
        # We can fix U and V to be sparse and solve for W.
        # Let's try a structured approach.
        # Use Strassen for the first 2 columns (k=0,1). This gives 7 terms.
        # For the 3rd column (k=2), we need to add 3 terms.
        # Let's see if we can reuse the U and V vectors from the first 7 terms.
        
        # Let U_1..U_7, V_1..V_7, W_1..W_7 be the Strassen terms for 2x2x2.
        # They satisfy: sum_{c=1}^7 U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = 1 for k=0,1.
        # For k=2, the term V_c[j*p+2] is 0 in the standard Strassen setup (since p=2).
        # So we need new terms for k=2.
        
        # Let's add 3 new terms c=8,9,10.
        # We need sum_{c=8}^{10} U_c[i*m+j] V_c[j*p+2] W_c[2*n+i] = 1 for all i,j.
        # Note: V_c[j*p+2] is the component for B[j,2].
        # Let's denote V_c2[j] = V_c[j*p+2].
        # We need sum_{c=8}^{10} U_c[i*m+j] V_c2[j] W_c[2*n+i] = 1.
        
        # This is equivalent to:
        # sum_{c=8}^{10} (U_c \otimes V_c2) [i*m+j] * W_c[2*n+i] = 1.
        
        # Let's choose V_c2 to be standard basis vectors.
        # Let V_82 = [1, 0], V_92 = [0, 1], V_102 = [1, 1].
        # Then we need:
        # U_8[i*m+j] * 1 * W_8[2n+i] + U_9[i*m+j] * 0 * ... + U_10[i*m+j] * 1 * W_10[2n+i] = 1 (if j=0)
        # U_8[i*m+1] * 0 * ... + U_9[i*m+1] * 1 * W_9[2n+i] + U_10[i*m+1] * 1 * W_10[2n+i] = 1 (if j=1)
        
        # This looks complicated. Let's just use a numerical solver or a known solution.
        # A known rank 10 solution for 2x2x3 exists.
        # Let's try to construct it using the "border rank" or specific algebraic identities.
        
        # Actually, I can just hardcode a known rank 10 solution if I can find one.
        # Or I can use the fact that rank(2x2x3) <= rank(2x2x2) + rank(2x2x1) = 7 + 4 = 11.
        # Is it 10? Yes, it is known to be 10.
        
        # Let's try to find it by solving a linear system.
        # We will fix U and V to have small integer entries and solve for W.
        
        # Let's try to extend the Strassen decomposition.
        # We have 7 terms for k=0,1.
        # We need 3 more terms for k=2.
        # Let's try to set U_8, U_9, U_10 to be related to A.
        
        # Let's try a specific known decomposition from literature.
        # One such decomposition is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # M8 = A00 B02
        # M9 = A01 B12
        # M10 = (A10+A11) B02 + A11 B12 ? No.
        
        # Let's try to solve for the last 3 terms.
        # We need to cover C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2].
        # Let's define the new terms to cover this.
        
        # Term 8: U_8 has 1 at (0,0), V_8 has 1 at (0,2), W_8 has 1 at (2,0).
        # Term 9: U_9 has 1 at (0,1), V_9 has 1 at (1,2), W_9 has 1 at (2,0).
        # Term 10: U_10 has 1 at (1,0), V_10 has 1 at (0,2), W_10 has 1 at (2,1).
        # Term 11: U_11 has 1 at (1,1), V_11 has 1 at (1,2), W_11 has 1 at (2,1).
        
        # This would be rank 7 + 4 = 11.
        # To get rank 10, we need to combine two of these.
        # Can we combine Term 8 and Term 10?
        # U_8 = [1,0,0,0], V_8 = [0,0,1,0,0,0] (for B[0,2]), W_8 = [0,0,1,0] (for C[0,2])
        # U_10 = [0,0,1,0], V_10 = [0,0,1,0,0,0] (for B[0,2]), W_10 = [0,0,0,1] (for C[1,2])
        
        # If we use a single term for B[0,2], we can only multiply by one linear combination of A.
        # Let M8 = (A00 + A10) B02.
        # Then C[0,2] gets A00 B02. C[1,2] gets A10 B02.
        # We still need A01 B12 and A11 B12.
        # Let M9 = A01 B12.
        # Let M10 = A11 B12.
        # This is 7 + 3 = 10 terms.
        
        # Let's verify this decomposition.
        # Terms 1-7: Strassen for k=0,1.
        # Term 8: U_8[0,0]=1, U_8[2,0]=1. V_8[0,2]=1. W_8[2,0]=1, W_8[2,1]=1.
        #   Contribution: A00 B02 -> C02, A10 B02 -> C12.
        # Term 9: U_9[1,0]=1. V_9[1,2]=1. W_9[2,0]=1.
        #   Contribution: A01 B12 -> C02.
        # Term 10: U_10[3,0]=1. V_10[1,2]=1. W_10[2,1]=1.
        #   Contribution: A11 B12 -> C12.
        
        # Let's check C[0,2]:
        # From Term 8: A00 B02.
        # From Term 9: A01 B12.
        # Total: A00 B02 + A01 B12. Correct.
        
        # Let's check C[1,2]:
        # From Term 8: A10 B02.
        # From Term 10: A11 B12.
        # Total: A10 B02 + A11 B12. Correct.
        
        # This seems valid!
        # Let's construct the matrices.
        
        R = 10
        U = np.zeros((4, R))
        V = np.zeros((6, R))
        W = np.zeros((6, R))
        
        Us1, Vs1, Ws1 = build_strassen_2x2x2()
        U[:, 0:7] = Us1
        V[0:4, 0:7] = Vs1
        W[0:4, 0:7] = Ws1
        
        # Term 8 (index 7)
        U[0, 7] = 1
        U[2, 7] = 1
        V[0, 7] = 1  # B[0,2] is at index 0*3+2 = 2? No.
        # V index is j*p+k. p=3.
        # B[0,2] is j=0, k=2. Index 0*3+2 = 2.
        # B[1,2] is j=1, k=2. Index 1*3+2 = 5.
        
        V[2, 7] = 1
        # W index is k*n+i. n=2.
        # C[0,2] is k=2, i=0. Index 2*2+0 = 4.
        # C[1,2] is k=2, i=1. Index 2*2+1 = 5.
        
        W[4, 7] = 1
        W[5, 7] = 1
        
        # Term 9 (index 8)
        U[1, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1
        
        # Term 10 (index 9)
        U[3, 9] = 1
        V[5, 9] = 1
        W[5, 9] = 1
        
        # Verify
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T_reconstructed[i, j, k] = val
        
        if np.allclose(T_reconstructed, 1.0, atol=1e-9):
            return U, V, W
        else:
            return None

    def build_2x2x3_rank11():
        """Return (U, V, W) for 2x2 * 2x3 -> 2x3 using rank 11 (1 Strassen + 1 schoolbook)."""
        R = 11
        U = np.zeros((4, R))
        V = np.zeros((6, R))
        W = np.zeros((6, R))
        
        Us1, Vs1, Ws1 = build_strassen_2x2x2()
        U[:, 0:7] = Us1
        V[0:4, 0:7] = Vs1
        W[0:4, 0:7] = Ws1
        
        Us2, Vs2, Ws2 = build_schoolbook_2x2x1()
        U[:, 7:11] = Us2
        V[4:6, 7:11] = Vs2
        W[4:6, 7:11] = Ws2
        
        return U, V, W

    # Strategy: 15 = 4 + 4 + 4 + 3
    # Try to use rank 10 for the 2x2x3 block.
    # If successful, total rank = 14 + 14 + 14 + 10 = 52.
    # If not, fallback to rank 11, total rank = 53.
    
    U_2x2x3, V_2x2x3, W_2x2x3 = find_2x2x3_rank10()
    
    if U_2x2x3 is not None:
        rank_2x2x3 = 10
    else:
        U_2x2x3, V_2x2x3, W_2x2x3 = build_2x2x3_rank11()
        rank_2x2x3 = 11
        
    total_rank = 14 + 14 + 14 + rank_2x2x3
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Block 1: 2x2x4
    Us, Vs, Ws = build_2x2x4_rank14()
    U_full[:, col_offset:col_offset+14] = Us
    V_full[0:4, col_offset:col_offset+14] = Vs[0:4, :]
    V_full[4:8, col_offset:col_offset+14] = Vs[4:8, :]
    W_full[0:8, col_offset:col_offset+14] = Ws
    col_offset += 14
    
    # Block 2: 2x2x4
    Us, Vs, Ws = build_2x2x4_rank14()
    U_full[:, col_offset:col_offset+14] = Us
    V_full[0*p+4:0*p+8, col_offset:col_offset+14] = Vs[0:4, :]
    V_full[1*p+4:1*p+8, col_offset:col_offset+14] = Vs[4:8, :]
    W_full[4*n+0:6*n, col_offset:col_offset+14] = Ws[0:4, :]
    W_full[6*n+0:8*n, col_offset:col_offset+14] = Ws[4:8, :]
    col_offset += 14
    
    # Block 3: 2x2x4
    Us, Vs, Ws = build_2x2x4_rank14()
    U_full[:, col_offset:col_offset+14] = Us
    V_full[0*p+8:0*p+12, col_offset:col_offset+14] = Vs[0:4, :]
    V_full[1*p+8:1*p+12, col_offset:col_offset+14] = Vs[4:8, :]
    W_full[8*n+0:10*n, col_offset:col_offset+14] = Ws[0:4, :]
    W_full[10*n+0:12*n, col_offset:col_offset+14] = Ws[4:8, :]
    col_offset += 14
    
    # Block 4: 2x2x3
    U_full[:, col_offset:col_offset+rank_2x2x3] = U_2x2x3
    V_full[0*p+12:0*p+15, col_offset:col_offset+rank_2x2x3] = V_2x2x3[0:3, :]
    V_full[1*p+12:1*p+15, col_offset:col_offset+rank_2x2x3] = V_2x2x3[3:6, :]
    W_full[12*n+0:15*n, col_offset:col_offset+rank_2x2x3] = W_2x2x3
    
    # Verify
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(total_rank):
                    val += U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c]
                T_reconstructed[i, j, k] = val
    
    if np.allclose(T_reconstructed, 1.0, atol=1e-9):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
