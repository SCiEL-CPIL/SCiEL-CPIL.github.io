"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Partition intermediate dimension j into two sets: {0,1} and {2,3}
    # Each sub-tensor: C[i,k] = A[i,2a] * B[2a,k] + A[i,2a+1] * B[2a+1,k]
    # This is a tensor product of:
    #   - 2x2x2 multiplication (rank 7 via Strassen) over (i, a_local, k_local)
    #   - 1x2x2 "broadcast" (rank 2) over (dummy, a_local, k_global)
    # Total rank per sub-tensor: 7 * 2 = 14
    # Total rank: 2 * 14 = 28

    # Strassen 2x2 decomposition:
    # P1 = (A11 + A22)(B11 + B22) -> C11 += 1, C22 += 1
    # P2 = (A21 + A22) B11       -> C21 += 1, C22 -= 1
    # P3 = A11 (B12 - B22)       -> C12 += 1, C22 += 1
    # P4 = A22 (B21 - B11)       -> C11 += 1, C21 += 1
    # P5 = (A11 + A12) B22       -> C11 -= 1, C12 += 1
    # P6 = (A21 - A11)(B11 + B12) -> C22 += 1
    # P7 = (A12 - A22)(B21 + B22) -> C11 += 1

    # Strassen factors: (U_local, V_local, W_local)
    # U_local: (i, a_local) -> coeff
    # V_local: (a_local, k_local) -> coeff  
    # W_local: (k_local, i) -> coeff
    strassen_factors = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Broadcast factors for the k dimension (2 terms):
    # Term 0: B[2a, k] for all k -> contributes to C[i, k]
    # Term 1: B[2a+1, k] for all k -> contributes to C[i, k]
    # Actually, the broadcast is: for each k_local in {0,1}, we have B[a_local, k_global]
    # where k_global = 4*a_local + k_local... no wait.
    # 
    # Let me reconsider. The sub-tensor for intermediate set {2a, 2a+1}:
    # C[i, k] = A[i, 2a] * B[2a, k] + A[i, 2a+1] * B[2a+1, k]
    # 
    # This is: sum_{a_local=0}^{1} A[i, 2a+a_local] * B[2a+a_local, k]
    # 
    # The tensor is: T[i, a_local, k] = A[i, 2a+a_local] * B[2a+a_local, k] * C[i,k]
    # 
    # This factors as: (2x2x2 mult over i, a_local, k_local) tensor (broadcast over a_local, k_global)
    # 
    # Wait, I need to be more careful. Let me think of it as:
    # For fixed a (the sub-tensor index), the contribution is:
    #   C[i,k] += A[i, 2a] * B[2a, k] + A[i, 2a+1] * B[2a+1, k]
    #
    # This is a 2x2x8 tensor (i in {0,1}, a_local in {0,1}, k in {0..7})
    # which equals the sum over a_local of A[i, 2a+a_local] * B[2a+a_local, k]
    #
    # We can write this as a tensor product of:
    #   - A 2x2x2 tensor S[i, a_local, k_local] (the Strassen-decomposable part)
    #   - A broadcast tensor B[a_local, k] 
    #
    # But that's not quite right either. Let me use the standard tensor product approach.
    #
    # The key insight: C[i,k] = sum_{j} A[i,j] B[j,k]
    # For sub-tensor a: C[i,k] = sum_{a_local} A[i, 2a+a_local] B[2a+a_local, k]
    #
    # This is the same as matrix multiplication of a 2x2 matrix (rows i, cols a_local)
    # by a 2x8 matrix (rows a_local, cols k).
    #
    # We can decompose the 2x2x2 part using Strassen (rank 7), and the "broadcast" 
    # from 2 to 8 columns is just a direct sum of 4 copies of the 2x1x2 part.
    #
    # Actually, the cleanest way: the 2x2x8 tensor is the tensor product of:
    #   - 2x2x2 multiplication tensor (rank 7)
    #   - 1x2x2 identity/broadcast tensor (rank 2)
    #
    # No wait. Let me just directly construct it.
    #
    # For sub-tensor a, the tensor is:
    #   T[i, j_local, k] = A[i, 2a+j_local] * B[2a+j_local, k] * C[i, k]
    # where j_local in {0,1} and k in {0..7}
    #
    # This can be written as: sum over 4 output column pairs, each being a 2x2x2 tensor
    #   For k_pair = 0..3: T[i, j_local, 2*k_pair + k_local]
    #     = A[i, 2a+j_local] * B[2a+j_local, 2*k_pair+k_local] * C[i, 2*k_pair+k_local]
    #
    # Each such 2x2x2 tensor can be decomposed with Strassen (rank 7).
    # Total: 4 * 7 = 28 per sub-tensor, 56 total. That's the same as before!
    #
    # Hmm, so the tensor product approach doesn't help directly. Let me reconsider.
    #
    # Actually, the correct tensor product decomposition is:
    # The 2x2x8 tensor = (2x2x2 tensor) x (1x2x4 tensor) where the 1x2x4 tensor
    # has entries T[0, a_local, k] = delta_{a_local, k mod 2} * (k // 2)
    # 
    # No, that's getting complicated. Let me just use the direct Strassen approach
    # but be smarter about it.
    #
    # Actually, I realize the original approach already does this correctly with rank 56.
    # The proposed improvement to rank 28 requires a different structure.
    #
    # Let me think again. The tensor T_{ijk} = A_{ij} B_{jk} C_{ki} for 2x4x8.
    # 
    # We can write: T = T_0 + T_1 where
    # T_a[i, j, k] = A[i, 2a+j] B[2a+j, k] C[i,k] for j in {0,1}
    #
    # T_a is a 2x2x8 tensor. Its rank is at most 16 (schoolbook).
    # Using Strassen on the 2x2 part: for each pair of k values, we get a 2x2x2 tensor.
    # There are 4 such pairs, so rank 4*7 = 28 per sub-tensor.
    # Total: 2 * 28 = 56. Same as before.
    #
    # To get rank 28 total, we need each sub-tensor to have rank 14.
    # This requires a rank-14 decomposition of the 2x2x8 multiplication tensor.
    #
    # The 2x2x8 tensor can be seen as: multiply 2x2 by 2x8.
    # This is equivalent to 4 independent 2x1x2 multiplications (one for each column of B).
    # Each 2x1x2 multiplication has rank 2 (schoolbook is optimal for this).
    # Total: 4 * 2 = 8. But we can do better using the structure.
    #
    # Actually, 2x2x8 = (2x2x2) tensor (1x1x4)? No...
    #
    # Let me just keep the current rank-56 solution which is correct, and note that
    # finding a rank-28 solution would require a novel decomposition.

    # Keep the original Strassen-based rank-56 solution
    strassen_products = [
        # P1 = (A11 + A22)(B11 + B22)
        # contributes to C11 with +1, C22 with +1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A21 + A22) B11
        # contributes to C21 with +1, C22 with -1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3 = A11 (B12 - B22)
        # contributes to C12 with +1, C22 with +1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4 = A22 (B21 - B11)
        # contributes to C11 with +1, C21 with +1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5 = (A11 + A12) B22
        # contributes to C11 with -1, C12 with +1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6 = (A21 - A11) (B11 + B12)
        # contributes to C22 with +1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A12 - A22) (B21 + B22)
        # contributes to C11 with +1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    rank = 7 * 8  # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(8):
        qc = block // 2
        ar = block % 2
        a_col0 = 2 * ar
        b_row0 = 2 * ar
        b_col0 = 2 * qc
        c_col0 = 2 * qc
        
        for u_coeff, v_coeff, w_coeff in strassen_products:
            for (li, lj), coeff in u_coeff.items():
                gi = li
                gj = a_col0 + lj
                U[gi * m + gj][col] += coeff
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row0 + lj
                gk = b_col0 + lk
                V[gj * p + gk][col] += coeff
            for (lk, li), coeff in w_coeff.items():
                gk = b_col0 + lk
                gi = li
                W[gk * n + gi][col] += coeff
            col += 1
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
