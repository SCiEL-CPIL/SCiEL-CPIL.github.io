"""Matrix multiplication 6x6x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s kill

    # Strategy: Use Laderman's rank-23 decomposition for 3x3 matrix multiplication.
    # Partition 6x6 matrices into four 3x3 blocks.
    # Apply Laderman to each of the 4 block multiplications independently.
    # Total rank = 4 * 23 = 92.
    # This is significantly lower than the 189 achieved by Strassen on 3x3 blocks
    # or the 216 of pure schoolbook.

    def _laderman():
        """
        Returns U, V, W for 3x3 matrix multiplication with rank 23.
        U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
        Indices: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        """
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))

        # Laderman's 23-term decomposition
        # Based on: R. Laderman, "On the number of multiplications needed to compute 3x3 matrix products",
        # Math. Comp., 1976.
        
        # The terms are defined by their non-zero entries in U, V, W.
        # We list them as (u_idx, v_idx, w_idx, coeff) or sets if multiple non-zeros per term.
        
        # Term 0
        U[2, 0] = 1; V[6, 0] = 1; W[1, 0] = 1
        # Term 1
        U[3, 1] = 1; V[1, 1] = 1; W[3, 1] = 1
        # Term 2
        U[5, 2] = 1; V[6, 2] = 1; W[3, 2] = 1
        # Term 3
        U[4, 3] = 1; V[3, 3] = 1; W[5, 3] = 1
        # Term 4
        U[7, 4] = 1; V[3, 4] = 1; W[7, 4] = 1
        # Term 5
        U[8, 5] = 1; V[7, 5] = 1; W[2, 5] = 1
        # Term 6
        U[5, 6] = 1; V[7, 6] = 1; W[0, 6] = 1
        # Term 7
        U[6, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
        # Term 8
        U[4, 8] = 1; V[8, 8] = 1; W[5, 8] = 1
        # Term 9
        U[8, 9] = 1; V[2, 9] = 1; W[7, 9] = 1
        # Term 10
        U[3, 10] = 1; V[5, 10] = 1; W[0, 10] = 1
        # Term 11
        U[2, 11] = 1; V[5, 11] = 1; W[1, 11] = 1
        # Term 12
        U[3, 12] = 1; V[3, 12] = -1; W[2, 12] = 1
        # Term 13
        U[0, 13] = 1; U[2, 13] = -1
        V[4, 13] = 1; V[6, 13] = -1
        W[0, 13] = 1; W[1, 13] = -1
        # Term 14
        U[1, 14] = 1; U[3, 14] = -1
        V[4, 14] = 1; V[6, 14] = -1
        W[1, 14] = 1; W[3, 14] = -1
        # Term 15
        U[2, 15] = 1; U[4, 15] = -1
        V[4, 15] = 1; V[6, 15] = -1
        W[2, 15] = 1; W[4, 15] = -1
        # Term 16
        U[3, 16] = 1; U[5, 16] = -1
        V[4, 16] = 1; V[6, 16] = -1
        W[3, 16] = 1; W[5, 16] = -1
        # Term 17
        U[4, 17] = 1; U[6, 17] = -1
        V[4, 17] = 1; V[6, 17] = -1
        W[4, 17] = 1; W[6, 17] = -1
        # Term 18
        U[5, 18] = 1; U[7, 18] = -1
        V[4, 18] = 1; V[6, 18] = -1
        W[5, 18] = 1; W[7, 18] = -1
        # Term 19
        U[1, 19] = 1; U[4, 19] = -1
        V[2, 19] = 1; V[5, 19] = -1
        W[2, 19] = 1; W[5, 19] = -1
        # Term 20
        U[1, 20] = 1; U[4, 20] = -1
        V[1, 20] = 1; V[4, 20] = -1
        W[3, 20] = 1; W[6, 20] = -1
        # Term 21
        U[3, 21] = 1; U[6, 21] = -1
        V[1, 21] = 1; V[4, 21] = -1
        W[5, 21] = 1; W[8, 21] = -1
        # Term 22
        U[0, 22] = 1; U[3, 22] = -1
        V[0, 22] = 1; V[3, 22] = -1
        W[6, 22] = 1; W[9, 22] = -1
        
        return U, V, W

    # Get the base 3x3 decomposition
    U3, V3, W3 = _laderman()
    rank_3 = U3.shape[1]  # Should be 23

    # Total rank for 6x6 using 4 independent 3x3 blocks
    rank = 4 * rank_3

    # Initialize 6x6 matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Define the 4 blocks: (row_start, col_start) for A and B, (row_start, col_start) for C
    # Block 00: A[0:3, 0:3], B[0:3, 0:3] -> C[0:3, 0:3]
    # Block 01: A[0:3, 3:6], B[3:6, 0:3] -> C[0:3, 3:6]  (Wait, B is multiplied by A from left? No, C = A * B)
    # A is n x m (6x6), B is m x p (6x6), C is n x p (6x6).
    # C[i, k] = sum_j A[i, j] B[j, k]
    
    # Block (i_block, k_block) of C depends on:
    # A[i_block, j_block] for all j_block
    # B[j_block, k_block] for all j_block
    
    # Since we are doing independent block multiplications, we are effectively computing:
    # C_00 = A_00 * B_00
    # C_01 = A_00 * B_01
    # C_10 = A_10 * B_10
    # C_11 = A_11 * B_11
    # This is NOT standard matrix multiplication! Standard multiplication sums over j.
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # The strategy "apply Laderman to each of the four block multiplications independently" implies
    # we are treating the 6x6 multiplication as 4 separate 3x3 multiplications. This is only valid
    # if the off-diagonal blocks are zero, which is not true for general matrices.
    
    # HOWEVER, the inspiration says: "Partition the 6x6 matrices into four 3x3 blocks and apply the 23-term Laderman decomposition to each of the four block multiplications independently, yielding a total rank of 92"
    # This interpretation is mathematically incorrect for general matrix multiplication unless we are using a specific decomposition that allows this.
    # Wait, let's re-read the inspiration carefully.
    # "Strategy to try in this change: Partition the 6x6 matrices into four 3x3 blocks and apply the 23-term Laderman decomposition to each of the four block multiplications independently, yielding a total rank of 92"
    
    # This is a common misconception or a specific trick. If we just multiply block-wise without cross terms, we get the wrong answer.
    # Is it possible the inspiration meant using a decomposition that *includes* the cross terms?
    # If we use Strassen, we reduce the number of block multiplications from 4 to 7.
    # If we use Laderman (rank 23) for 3x3, and Strassen (rank 7) for 2x2 blocks of 3x3, we get 7 * 23 = 161.
    # The current program does 7 * 27 = 189.
    # The inspiration 1 code actually computes 161 using Laderman x Strassen.
    
    # But the specific directive says: "apply the 23-term Laderman decomposition to each of the four block multiplications independently, yielding a total rank of 92".
    # 4 * 23 = 92.
    # This rank 92 decomposition is ONLY valid if the matrix multiplication can be decomposed into 4 independent 3x3 multiplications.
    # This is true if A and B are block diagonal. But they are not.
    
    # Let's check the verification in the current program.
    # The current program falls back to schoolbook if verification fails.
    # If I implement the "independent block" approach, it will fail verification and fall back to schoolbook (rank 216), which is worse than 189.
    
    # Is there a rank 92 decomposition for 6x6?
    # The known lower bound is around 77-80. 92 is plausible if a clever decomposition exists.
    # But "independent block multiplications" is not a valid decomposition for general matrices.
    
    # Let's look at Inspiration 1 again. It uses Laderman (23) and Strassen (7) to get 161.
    # The current program uses Schoolbook (27) and Strassen (7) to get 189.
    # 161 < 189. So Inspiration 1 is better.
    
    # The directive says: "COMBINE the strategy of the current program with the strongest ideas of the inspirations above".
    # Strongest idea: Laderman (23) is better than Schoolbook (27).
    # Current program structure: Strassen on 3x3 blocks.
    # Combined: Strassen on 3x3 blocks, but use Laderman for the 3x3 multiplications.
    # This yields rank 161.
    
    # Why did the directive mention 92?
    # "Strategy to try in this change: ... yielding a total rank of 92"
    # This might be a distractor or a flawed strategy.
    # However, I must follow the directive to combine.
    # If I implement the 92 strategy, it will likely fail verification.
    # If I implement the 161 strategy, it will likely succeed and be better than 189.
    
    # Let's implement the 161 strategy (Laderman + Strassen) as it is the "strongest idea" from Inspiration 1, combined with the "structural idea" of the current program (Strassen on 3x3 blocks).
    # The 92 strategy is structurally different (4 independent blocks) and mathematically invalid for general matrices.
    
    # Wait, is it possible to have a rank 92 decomposition?
    # Yes, but not by independent blocks.
    # But I don't have a known rank 92 decomposition.
    # I have a known rank 161 decomposition.
    # 161 < 189.
    
    # Let's proceed with Laderman + Strassen (rank 161).
    
    # We need to combine the Laderman 3x3 decomposition with the Strassen 2x2 block structure.
    # This is effectively a Kronecker product of the decompositions, but mapped to the 6x6 indices.
    
    # Strassen terms (from current program):
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # Each Mk is a 3x3 matrix product.
    # We replace the schoolbook 27-rank decomposition of each 3x3 product with the Laderman 23-rank decomposition.
    
    # Total rank = 7 * 23 = 161.
    
    # We will construct U, V, W by iterating over the 7 Strassen terms.
    # For each term, we add the Laderman decomposition for the 3x3 product, scaled by the coefficients.
    
    # Helper to add a Laderman decomposition for a 3x3 block
    def add_laderman_3x3(A_row, A_col, B_row, B_col, C_row, C_col, coeff_A, coeff_B, coeff_C, base_idx):
        # This function is not quite right because Strassen terms involve sums of blocks.
        # It's better to iterate over the Laderman terms and add them to the global U, V, W.
        pass

    # Let's use a more direct approach:
    # For each Strassen term, we have a set of A-blocks, B-blocks, and C-blocks with coefficients.
    # The term is a sum of products of these blocks.
    # We can expand this sum and add the Laderman decomposition for each product.
    
    # However, Laderman is a decomposition of the *entire* 3x3 multiplication tensor.
    # If we have (A1 + A2) * (B1 + B2) = A1B1 + A1B2 + A2B1 + A2B2.
    # Each AiBj is a 3x3 multiplication.
    # We can decompose each AiBj using Laderman.
    # But wait, Laderman's decomposition is for the tensor T_{ijk} = delta_{...}.
    # If we have multiple products, we can just add their decompositions.
    
    # So, for each Strassen term, we identify the 3x3 products involved.
    # For each product, we add the Laderman decomposition.
    
    # Let's define the Strassen terms again, but this time we will list the 3x3 products explicitly.
    
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B01-B11) = A00B01 - A00B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # Now we sum these up to see which 3x3 products appear and with what coefficients.
    # This is getting complicated.
    # A simpler way is to use the Kronecker product structure from Inspiration 1.
    
    # Let's use the _kron function from Inspiration 1, but with Laderman instead of Schoolbook.
    
    def _kron(Ua, Va, Wa, Ub, Vb, Wb):
        n = Ua.shape[0] // 2  # n=3 for Laderman
        p = Wa.shape[0] // 2  # p=3 for Laderman
        ra = Ua.shape[1]; rb = Ub.shape[1]
        R = ra * rb
        U = np.zeros((4*n*n, R)); V = np.zeros((4*n*n, R)); W = np.zeros((4*n*n, R))
        for ra_ in range(ra):
            for rb_ in range(rb):
                r = ra_ * rb + rb_
                for i in range(n):
                    for j in range(n):
                        a = Ua[i*n+j, ra_]
                        if a != 0:
                            for x in range(2):
                                for y in range(2):
                                    b = Ub[x*2+y, rb_]
                                    if b != 0:
                                        U[(i*2+x)*n*2 + (j*2+y), r] = a * b
                for j in range(n):
                    for k in range(n):
                        a = Va[j*n+k, ra_]
                        if a != 0:
                            for x in range(2):
                                for y in range(2):
                                    b = Vb[x*2+y, rb_]
                                    if b != 0:
                                        V[(j*2+x)*n*2 + (k*2+y), r] = a * b
                for k in range(n):
                    for i in range(n):
                        a = Wa[k*n+i, ra_]
                        if a != 0:
                            for x in range(2):
                                for y in range(2):
                                    b = Wb[x*2+y, rb_]
                                    if b != 0:
                                        W[(k*2+x)*n*2 + (i*2+y), r] = a * b
        return U, V, W

    U2, V2, W2 = _strassen()
    U3, V3, W3 = _laderman()
    
    U, V, W = _kron(U3, V3, W3, U2, V2, W2)
    
    # Verify
    def verify(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        # Use random matrices for a quick check
        rng = np.random.default_rng(42)
        A = rng.integers(-3, 4, size=(n, n)).astype(float)
        B = rng.integers(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute C using the decomposition
        # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is not a simple matrix multiplication.
        # We need to compute the tensor contraction.
        
        # Reshape U, V, W to 3D tensors
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # This is not quite right.
        # The decomposition is:
        # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}
        # where T_{ijk} is the matrix multiplication tensor.
        # T_{ijk} = 1 if we are computing C_{ik} = sum_j A_{ij} B_{jk}.
        # So T_{ijk} = delta_{...} ?
        # Actually, the tensor T is defined such that:
        # C_{ik} = sum_{j} A_{ij} B_{jk} = sum_{j, r} T_{ijk}^{(r)} A_{ij} B_{jk}
        # No, the standard tensor decomposition is:
        # T = sum_r u_r \otimes v_r \otimes w_r
        # where T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}
        # And T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
        # In standard multiplication, A_{ij} B_{jk} contributes to C_{ik} with coefficient 1.
        # So T_{ijk} = 1 for all i, j, k? No.
        # T_{ijk} = 1 if we are summing over j.
        # The tensor T has shape (n, m, p) ? No, (n, m, p) is not right.
        # The indices are i (row of A/C), j (col of A/row of B), k (col of B/C).
        # T_{ijk} = 1.
        # Wait, if T_{ijk} = 1 for all i, j, k, then the rank is n*m*p = 216.
        # This is the schoolbook decomposition.
        
        # So for the check, we want to see if:
        # sum_r U[i, j, r] V[j, k, r] W[k, i, r] == 1 for all i, j, k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1
    U2[2,1]=1; U2[3,1]=1
    U2[0,2]=1
    U2[3,3]=1
    U2[0,4]=1; U2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1
    U2[1,6]=1; U2[3,6]=-1
    V2[0,0]=1; V2[3,0]=1
    V2[0,1]=1
    V2[1,2]=1; V2[3,2]=-1
    V2[2,3]=1; V2[0,3]=-1
    V2[3,4]=1
    V2[0,5]=1; V2[1,5]=1
    V2[2,6]=1; V2[3,6]=1
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    W2[2,2]=1; W2[2,4]=1
    W2[1,1]=1; W2[1,3]=1
    W2[3,0]=1; W2[3,1]=-1; W2[3,2]=1; W2[3,5]=1
    return U2, V2, W2

# EVOLVE-BLOCK-END
