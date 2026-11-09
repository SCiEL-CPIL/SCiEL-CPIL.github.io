"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # We need to decompose the matrix multiplication tensor for C = A * B
    # where A is 4x5 and B is 5x5.
    # C[i, k] = sum_{j=0}^{4} A[i, j] * B[j, k]
    #
    # The tensor is T_{i, j, k} = delta_{i, k? No, T is the structure tensor of matrix mult}
    # Actually, the structure tensor T has entries T[i, j, k] which is 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 for all i in 0..3, j in 0..4, k in 0..4.
    # Wait, that's not right. The tensor T is defined such that the bilinear form sum_{i,j,k} T[i,j,k] x_i y_j z_k equals the product.
    # Standard definition: T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # So T[i, j, k] = 1 for all valid i, j, k.
    #
    # We can use Strassen's algorithm for 2x2 blocks.
    # A is 4x5, B is 5x5.
    # We can split A into four 2x2 blocks plus a 2x1 block? Or four 2x2 and a 2x1?
    # Let's split A (4x5) into:
    # A00 (2x2), A01 (2x2), A02 (2x1)
    # A10 (2x2), A11 (2x2), A12 (2x1)
    #
    # B (5x5) into:
    # B00 (2x2), B01 (2x2), B02 (2x1)
    # B10 (2x2), B11 (2x2), B12 (2x1)
    #
    # C (4x5) into:
    # C00 (2x2), C01 (2x2), C02 (2x1)
    # C10 (2x2), C11 (2x2), C12 (2x1)
    #
    # Let's use Strassen on the 2x2 blocks.
    # For the 2x2 blocks, we can use the 7-term Strassen decomposition.
    # For the 2x1 blocks, we need a decomposition for 2x1 * 1x1 -> 2x1? No, it's 2x1 * 1x5? No.
    #
    # Let's try a different approach. Use the known rank for 4x5*5x5.
    # According to the knowledge, replacing 4 disjoint 2x2x2 standard sub-problems with Strassen reduces rank from 100 to 96.
    # This suggests that the base decomposition was 100, and using Strassen on four 2x2 blocks saves 4*1 = 4? No, Strassen saves 1 per 2x2 block (8 -> 7).
    # So 4 blocks * 1 saving = 4 savings. 100 - 4 = 96.
    #
    # How to construct this?
    # We can view the 4x5 * 5x5 multiplication as involving several 2x2 * 2x2 multiplications.
    # Let's partition the indices.
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4}.
    #
    # Let's group i into {0,1} and {2,3}.
    # Let's group k into {0,1}, {2,3}, {4}.
    # Let's group j into {0,1}, {2,3}, {4}.
    #
    # We have blocks:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2 * 2x2 -> 2x2)
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2 * 2x2 -> 2x2)
    # A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (2x2 * 2x1 -> 2x1)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2 * 2x2 -> 2x2)
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (2x2 * 2x2 -> 2x2)
    # A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] (2x2 * 2x1 -> 2x1)
    # A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2] (2x1 * 1x2 -> 2x2)
    # A[0:2, 4:5] * B[4:5, 2:4] -> C[0:2, 2:4] (2x1 * 1x2 -> 2x2)
    # A[0:2, 4:5] * B[4:5, 4:5] -> C[0:2, 4:5] (2x1 * 1x1 -> 2x1)
    # And similarly for A[2:4, ...]
    #
    # This is getting complicated. Let's use the known result: rank 96.
    # We can construct the 96-term decomposition by using Strassen on the 2x2 blocks and standard multiplication for the rest.
    #
    # Let's define the Strassen decomposition for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We will apply this to each 2x2 * 2x2 block.
    # For the non-2x2 blocks, we use standard multiplication.
    #
    # Let's count the terms:
    # There are 4 blocks of 2x2 * 2x2 for the main part? No.
    # Let's list all the block multiplications:
    # For i-block {0,1} and {2,3}, and k-block {0,1}, {2,3}, {4}, and j-block {0,1}, {2,3}, {4}:
    #
    # i-block 0 ({0,1}):
    #   j-block 0 ({0,1}):
    #     k-block 0 ({0,1}): 2x2 * 2x2 -> 2x2 (Strassen: 7 terms)
    #     k-block 1 ({2,3}): 2x2 * 2x2 -> 2x2 (Strassen: 7 terms)
    #     k-block 2 ({4}):   2x2 * 2x1 -> 2x1 (Standard: 2*2*1 = 4 terms)
    #   j-block 1 ({2,3}):
    #     k-block 0 ({0,1}): 2x2 * 2x2 -> 2x2 (Strassen: 7 terms)
    #     k-block 1 ({2,3}): 2x2 * 2x2 -> 2x2 (Strassen: 7 terms)
    #     k-block 2 ({4}):   2x2 * 2x1 -> 2x1 (Standard: 4 terms)
    #   j-block 2 ({4}):
    #     k-block 0 ({0,1}): 2x1 * 1x2 -> 2x2 (Standard: 2*1*2 = 4 terms)
    #     k-block 1 ({2,3}): 2x1 * 1x2 -> 2x2 (Standard: 4 terms)
    #     k-block 2 ({4}):   2x1 * 1x1 -> 2x1 (Standard: 2*1*1 = 2 terms)
    #
    # i-block 1 ({2,3}):
    #   (Same as i-block 0)
    #   j-block 0: 7 + 7 + 4 = 18
    #   j-block 1: 7 + 7 + 4 = 18
    #   j-block 2: 4 + 4 + 2 = 10
    #
    # Total for i-block 0: 18 + 18 + 10 = 46
    # Total for i-block 1: 46
    # Total rank: 46 + 46 = 92?
    #
    # Wait, the knowledge says 96. Let's re-check.
    # Maybe the 2x2 * 2x1 block is not 4 terms?
    # A (2x2) * B (2x1) -> C (2x1)
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is 4 terms. Correct.
    #
    # A (2x1) * B (1x2) -> C (2x2)
    # C[0,0] = A[0]*B[0,0]
    # C[0,1] = A[0]*B[0,1]
    # C[1,0] = A[1]*B[0,0]
    # C[1,1] = A[1]*B[0,1]
    # This is 4 terms. Correct.
    #
    # A (2x1) * B (1x1) -> C (2x1)
    # C[0] = A[0]*B
    # C[1] = A[1]*B
    # This is 2 terms. Correct.
    #
    # So why 96? Maybe the partition is different.
    # Let's try partitioning j into {0,1,2,3,4} and k into {0,1,2,3,4}.
    # Maybe we use Strassen on 2x2 blocks of A and B, but the dimensions are 4x5 and 5x5.
    #
    # Let's just use the 96-term decomposition from the literature or construct it carefully.
    # Actually, the knowledge says "replacing 4 disjoint 2x2x2 standard sub-problems with Strassen's 7-term decomposition reduces the tensor rank from 100 to 96".
    # This implies the base was 100, and 4 blocks were changed from 8 terms to 7 terms each.
    # 100 - 4*1 = 96.
    #
    # So we need to find a decomposition of rank 100 that has 4 disjoint 2x2x2 sub-problems.
    # The standard decomposition has rank 4*5*5 = 100.
    # Can we find 4 disjoint 2x2x2 sub-problems in the standard decomposition?
    # Yes, for example, take i in {0,1}, j in {0,1}, k in {0,1}. This is a 2x2x2 sub-problem.
    # Similarly, i in {0,1}, j in {2,3}, k in {2,3}.
    # i in {2,3}, j in {0,1}, k in {0,1}.
    # i in {2,3}, j in {2,3}, k in {2,3}.
    # These 4 sub-problems are disjoint and cover i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}.
    # The remaining terms involve j=4 or k=4.
    #
    # So we can use Strassen on these 4 sub-problems, and standard multiplication for the rest.
    #
    # Let's implement this.
    n, m, p = DIMENSIONS
    rank = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    # We'll use a helper to add a term
    def add_term(i, j, k, val_u=1, val_v=1, val_w=1):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = val_u
        V[j * p + k][col] = val_v
        W[k * n + i][col] = val_w
        col += 1
    #
    # Define the 4 Strassen blocks:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 3: i in {2,3}, j in {0,1}, k in {0,1}
    # Block 4: i in {2,3}, j in {2,3}, k in {2,3}
    #
    # Strassen terms for a block with i0, j0, k0 (top-left indices):
    # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    # M3: A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
    # M4: A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
    # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
    #
    # Let's add these terms for each block.
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1
        add_term(i0, j0, k0, 1, 1, 1)
        add_term(i0, j0, k0+1, 1, 1, 1)
        add_term(i0+1, j0+1, k0, 1, 1, 1)
        add_term(i0+1, j0+1, k0+1, 1, 1, 1)
        # M2
        add_term(i0+1, j0, k0, 1, 1, 1)
        add_term(i0+1, j0+1, k0, 1, 1, 1)
        # M3
        add_term(i0, j0, k0+1, 1, -1, 1)
        add_term(i0, j0, k0, 1, 1, 1) # Wait, M3 is A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
        # I need to be careful with the signs and indices.
        # Let's use a more systematic approach.
        pass
    #
    # This is getting messy. Let's just use the standard decomposition but replace the 4 blocks with Strassen.
    # I'll write a function to add a standard term and a function to add a Strassen block.
    #
    col = 0
    def add_std(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    #
    # Strassen block for i0, j0, k0
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        add_std(i0, j0, k0)
        add_std(i0, j0, k0+1)
        add_std(i0+1, j0+1, k0)
        add_std(i0+1, j0+1, k0+1)
        # M2 = (A10+A11)B00
        add_std(i0+1, j0, k0)
        add_std(i0+1, j0+1, k0)
        # M3 = A00(B10-B11)
        add_std(i0, j0, k0+1)
        U[i0 * m + j0][col-1] = 1
        V[j0 * p + k0+1][col-1] = 1
        W[k0+1 * n + i0][col-1] = 1
        # Wait, I'm overwriting. Let's do it properly.
        pass
    #
    # I'll just implement the 96-term decomposition directly using the known formula.
    # Since I don't have the exact formula, I'll use a simple approach:
    # Use standard multiplication for all terms, but for the 4 blocks, use Strassen.
    # I'll write a clean implementation.
    #
    col = 0
    # Initialize U, V, W
    U = [[0] * 96 for _ in range(20)]
    V = [[0] * 96 for _ in range(25)]
    W = [[0] * 96 for _ in range(20)]
    #
    # Add Strassen blocks
    # Block 1: i0=0, j0=0, k0=0
    # Block 2: i0=0, j0=2, k0=2
    # Block 3: i0=2, j0=0, k0=0
    # Block 4: i0=2, j0=2, k0=2
    #
    # I'll use a helper function to add a Strassen block.
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m+j0][col] = 1; V[j0*p+k0][col] = 1; W[k0*n+i0][col] = 1; col += 1
        U[i0*m+j0][col] = 1; V[j0*p+k0+1][col] = 1; W[k0+1*n+i0][col] = 1; col += 1
        U[i0+1*m+j0+1][col] = 1; V[j0*p+k0][col] = 1; W[k0*n+i0+1][col] = 1; col += 1
        U[i0+1*m+j0+1][col] = 1; V[j0*p+k0+1][col] = 1; W[k0+1*n+i0+1][col] = 1; col += 1
        # M2
        U[i0+1*m+j0][col] = 1; V[j0*p+k0][col] = 1; W[k0*n+i0+1][col] = 1; col += 1
        U[i0+1*m+j0+1][col] = 1; V[j0*p+k0][col] = 1; W[k0*n+i0+1][col] = 1; col += 1
        # M3
        U[i0*m+j0][col] = 1; V[j0*p+k0+1][col] = -1; W[k0+1*n+i0][col] = 1; col += 1
        U[i0*m+j0][col] = 1; V[j0*p+k0][col] = 1; W[k0*n+i0][col] = 1; col += 1
        # M4
        U[i0+1*m+j0+1][col] = 1; V[j0*p+k0+1][col] = 1; W[k0+1*n+i0+1][col] = 1; col += 1
        U[i0+1*m+j0+1][col] = 1; V[j0*p+k0][col] = -1; W[k0*n+i0+1][col] = 1; col += 1
        # M5
        U[i0*m+j0][col] = 1; V[j0+1*p+k0+1][col] = 1; W[k0+1*n+i0][col] = 1; col += 1
        U[i0*m+j0+1][col] = 1; V[j0+1*p+k0+1][col] = 1; W[k0+1*n+i0][col] = 1; col += 1
        # M6
        U[i0+1*m+j0][col] = 1; V[j0*p+k0][col] = 1; W[k0*n+i0+1][col] = 1; col += 1
        U[i0*m+j0][col] = -1; V[j0*p+k0][col] = 1; W[k0*n+i0][col] = 1; col += 1
        U[i0+1*m+j0][col] = 1; V[j0*p+k0+1][col] = 1; W[k0+1*n+i0+1][col] = 1; col += 1
        U[i0*m+j0][col] = -1; V[j0*p+k0+1][col] = 1; W[k0+1*n+i0][col] = 1; col += 1
        # M7
        U[i0*m+j0+1][col] = 1; V[j0+1*p+k0][col] = 1; W[k0*n+i0][col] = 1; col += 1
        U[i0*m+j0+1][col] = 1; V[j0+1*p+k0+1][col] = 1; W[k0+1*n+i0][col] = 1; col += 1
        U[i0+1*m+j0+1][col] = -1; V[j0+1*p+k0][col] = 1; W[k0*n+i0+1][col] = 1; col += 1
        U[i0+1*m+j0+1][col] = -1; V[j0+1*p+k0+1][col] = 1; W[k0+1*n+i0+1][col] = 1; col += 1
    #
    # This is 28 terms per block, which is too many. Strassen is 7 terms.
    # I made a mistake in the implementation. Each "term" in Strassen is a product of sums, which expands to multiple standard terms.
    # But in the tensor decomposition, each "term" is a single rank-1 tensor.
    # So M1 = (A00+A11)(B00+B11) is not a single rank-1 tensor. It's a sum of 4 rank-1 tensors.
    # So Strassen's 7 terms are 7 bilinear forms, each of which is a product of two linear forms.
    # In the tensor decomposition, each bilinear form corresponds to one rank-1 tensor.
    # So M1 corresponds to one rank-1 tensor with U, V, W entries that are sums.
    #
    # Ah, I see. The U, V, W matrices can have entries that are not just 0 or 1. They can be any integers.
    # So for M1, U[i0,j0] = 1, U[i0+1,j0+1] = 1, V[j0,k0] = 1, V[j0+1,k0+1] = 1, W[k0,i0] = 1, W[k0+1,i0+1] = 1, etc.
    #
    # Let's redo this.
    col = 0
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col] = 1
        U[i0+1*m+j0+1][col] = 1
        V[j0*p+k0][col] = 1
        V[j0+1*p+k0+1][col] = 1
        W[k0*n+i0][col] = 1
        W[k0+1*n+i0+1][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[i0+1*m+j0][col] = 1
        U[i0+1*m+j0+1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i0+1][col] = 1
        col += 1
        # M3 = A00(B10-B11)
        U[i0*m+j0][col] = 1
        V[j0+1*p+k0][col] = 1
        V[j0+1*p+k0+1][col] = -1
        W[k0*n+i0][col] = 1
        W[k0+1*n+i0][col] = -1
        col += 1
        # M4 = A11(B01-B00)
        U[i0+1*m+j0+1][col] = 1
        V[j0*p+k0+1][col] = 1
        V[j0*p+k0][col] = -1
        W[k0+1*n+i0+1][col] = 1
        W[k0*n+i0+1][col] = -1
        col += 1
        # M5 = (A00+A01)B11
        U[i0*m+j0][col] = 1
        U[i0*m+j0+1][col] = 1
        V[j0+1*p+k0+1][col] = 1
        W[k0+1*n+i0][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[i0+1*m+j0][col] = 1
        U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1
        V[j0*p+k0+1][col] = 1
        W[k0*n+i0+1][col] = 1
        W[k0+1*n+i0+1][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[i0*m+j0+1][col] = 1
        U[i0+1*m+j0+1][col] = -1
        V[j0+1*p+k0][col] = 1
        V[j0+1*p+k0+1][col] = 1
        W[k0*n+i0][col] = 1
        W[k0+1*n+i0][col] = 1
        col += 1
    #
    # Now add the 4 Strassen blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(2, 0, 0)
    add_strassen_block(2, 2, 2)
    #
    # Now add the remaining standard terms
    # The remaining terms are those where j=4 or k=4, or i is not in {0,1,2,3} (but i is always in {0,1,2,3}).
    # So we need to add terms for j=4 and k=4.
    #
    # For i in 0..3, j=4, k in 0..4:
    for i in range(4):
        for k in range(5):
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    #
    # For i in 0..3, j in 0..3, k=4:
    for i in range(4):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    #
    # Total terms: 4*7 + 4*5 + 4*4 = 28 + 20 + 16 = 64? No.
    # Wait, the Strassen blocks cover i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}.
    # The remaining terms are:
    # 1. j=4, k in 0..4: 4*5 = 20 terms
    # 2. j in 0..3, k=4: 4*4 = 16 terms
    # Total: 28 + 20 + 16 = 64. But we need 96.
    #
    # I see the mistake. The Strassen blocks only cover 2x2x2 sub-problems, but the standard terms for j=4 and k=4 are not all covered.
    # Actually, the standard decomposition has 100 terms. The Strassen blocks replace 4*8=32 terms with 4*7=28 terms.
    # So the total should be 100 - 32 + 28 = 96.
    #
    # So I need to add all the standard terms except the ones covered by the Strassen blocks.
    #
    # Let's redo this.
    col = 0
    U = [[0] * 96 for _ in range(20)]
    V = [[0] * 96 for _ in range(25)]
    W = [[0] * 96 for _ in range(20)]
    #
    # Add Strassen blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(2, 0, 0)
    add_strassen_block(2, 2, 2)
    #
    # Add remaining standard terms
    # For i in 0..3, j in 0..3, k in 0..3, skip the 4 blocks
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Skip if (i,j,k) is in one of the 4 blocks
                if (i < 2 and j < 2 and k < 2) or (i < 2 and j >= 2 and k >= 2) or (i >= 2 and j < 2 and k < 2) or (i >= 2 and j >= 2 and k >= 2):
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    #
    # Add terms for j=4
    for i in range(4):
        for k in range(5):
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    #
    # Add terms for k=4 (and j in 0..3, already handled above? No, j=4 is handled, k=4 is new)
    for i in range(4):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    #
    # Total: 28 + (4*4*4 - 4*4*4) + 20 + 16 = 28 + 0 + 20 + 16 = 64? No.
    # The loop for i,j,k in 0..3 skips the 4 blocks, so it adds 64 - 32 = 32 terms.
    # Total: 28 + 32 + 20 + 16 = 96. Correct.
    #
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
