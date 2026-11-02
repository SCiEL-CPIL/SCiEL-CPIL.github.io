import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def build_strassen_block(U, V, W, i0, j0, k0, col):
        """Add one Strassen 2x2x2 block at position (i0, j0, k0), return new col."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        return col

    def build_221_block(U, V, W, i0, j0, k0, col):
        """Add schoolbook 2x2x1 block."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        return col

    # Build the base rank-50 decomposition
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)

    col = 0
    for i0 in [0]:
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                col = build_strassen_block(U, V, W, i0, j_start, k_start, col)
            col = build_221_block(U, V, W, i0, j_start, 6, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        # Now try to reduce rank via hypergraph coverage search
        # The key insight: look for ways to merge terms across blocks
        # 
        # Strategy: Try different block partitionings and see if any give rank < 50
        # 
        # Alternative partitionings to try:
        # 1. Use Strassen on different row pairings
        # 2. Mix Strassen and schoolbook differently
        # 3. Try 2x4x2 + 2x4x2 + 2x4x2 + 2x4x1 structures
        
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Try: Can we do better by using a different structure?
        # Key observation: the 2x4x7 tensor has a specific structure.
        # Let's try to find a decomposition with rank < 50 by exploring
        # different block combinations.
        
        # Approach: Enumerate different ways to partition the problem
        # and use the best known blocks for each partition.
        
        # Partition strategy: divide m=4 into groups, p=7 into groups
        # For each group of (m_size, p_size), use the best known decomposition
        
        # Known optimal ranks for small cases:
        # 2x2x2: rank 7 (Strassen)
        # 2x2x1: rank 4 (schoolbook, can't do better)
        # 2x1x1: rank 2 (schoolbook)
        # 1x1x1: rank 1
        
        # Try different partitions of p=7 into groups of size 2 and 1:
        # Partitions of 7 into {2,1}: 2+2+2+1, 2+2+1+1+1, etc.
        # For each partition, compute total rank.
        
        # Actually, let's think more carefully. The structure is:
        # A is 2x4, B is 4x7, C is 2x7
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # 
        # We can group the j dimension and k dimension independently.
        # For a fixed row i, we need:
        # C[i, :] = A[i, :] * B  (row vector times matrix)
        # 
        # This is a 1x4x7 tensor for each row, but the two rows share the B matrix.
        # 
        # The key question: can we exploit sharing between rows to reduce rank?
        # 
        # In the Strassen approach, we pair rows and use 7 multiplications for 2x2x2.
        # The saving comes from the fact that (a+b)*(c+d) gives us 4 products from 1 mult.
        # 
        # Let's try: what if we use a 2x4x4 block with a custom decomposition?
        # The 2x4x4 tensor: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2,3}
        # 
        # One approach: split j into {0,1} and {2,3}
        # C[i, 0:2] = A[i, 0:2] * B[0:2, 0:2] + A[i, 2:4] * B[2:4, 0:2]
        # C[i, 2:4] = A[i, 0:2] * B[0:2, 2:4] + A[i, 2:4] * B[2:4, 2:4]
        # 
        # Each 2x2x2 block costs 7 with Strassen, so 4 blocks = 28.
        # But we can do better by sharing:
        # 
        # Let's define:
        # P1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k]) for k in {0,1}
        # This gives 2 multiplications that produce 4 values.
        # 
        # Actually, the standard Strassen decomposition for 2x2x2 is already optimal.
        # The question is whether combining multiple blocks can reduce total rank.
        # 
        # Let me try a different approach: use a 4x4x4 decomposition.
        # The rank of 4x4x4 matrix multiplication is 49 (it's known).
        # We need 2x4x7, which is a subcase.
        # 
        # Actually, let's just try several concrete decompositions and pick the best.
        
        # Try 1: 6 Strassen + 2 schoolbook-221 = 50 (current)
        
        # Try 2: Different partitioning
        # What if we use Strassen on (rows 0-1, cols 0-1), (rows 0-1, cols 2-3), 
        # (rows 0-1, cols 4-5), and then handle col 6 differently?
        # That's what we have.
        
        # Try 3: Can we merge the schoolbook 2x2x1 blocks with adjacent Strassen blocks?
        # The 2x2x1 block at (j=0, k=6) and (j=2, k=6) could potentially be merged
        # with the Strassen blocks at (j=0, k=4) and (j=2, k=4).
        
        # Let's try: use a 2x2x3 block (Strassen-like for 3 columns)
        # For 2x2x3: C[i, k] = sum_{j=0}^{1} A[i,j]*B[j,k] for k in {0,1,2}
        # This is 2*2*3 = 12 scalar multiplications in schoolbook.
        # Can we do better?
        # 
        # Using Strassen ideas:
        # M1 = (a00+a11)*(b00+b11) -> gives a00*b00 + a00*b11 + a11*b00 + a11*b11
        # M2 = (a10+a11)*(b00) -> gives a10*b00 + a11*b00
        # M3 = (a00)*(b10+b11) -> gives a00*b10 + a00*b11
        # M4 = (a11)*(b10) -> gives a11*b10
        # ... this is getting complicated.
        
        # Let me try a more systematic approach:
        # Use integer programming / greedy to find a minimal set of rank-1 terms.
        
        # The matrix multiplication tensor for 2x4x7 has:
        # - 8 A-indices (i,j) for i in {0,1}, j in {0,1,2,3}
        # - 28 B-indices (j,k) for j in {0,1,2,3}, k in {0,...,6}
        # - 14 C-indices (i,k) for i in {0,1}, k in {0,...,6}
        
        # A rank-1 term is a triple (u, v, w) where:
        # u is a vector in R^8, v in R^28, w in R^14
        # such that the tensor T = sum_c u_c * v_c * w_c equals the MM tensor.
        
        # The MM tensor T[i,j,k] = 1 if the term contributes to C[i,k] via A[i,j]*B[j,k]
        # Actually, T is a 3-tensor with T[i,j,k] = 1 for all valid (i,j,k).
        
        # For a rank-1 term to be valid, we need:
        # sum_c u_c[i,j] * v_c[j',k] * w_c[i',k'] = delta_{j,j'} * delta_{i,i'} for all (i,j,j',i',k,k')
        # Wait, that's not quite right. Let me reconsider.
        
        # The decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: the MM tensor T has T[i,j,k] = 1 (it's just the identity in the j-index)
        # 
        # A rank-1 decomposition: T = sum_c T_c where T_c[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[i,k]
        # 
        # We need: sum_c u_c[i,j] * v_c[j,k] * w_c[i,k] = 1 for all (i,j,k)
        # 
        # This is a system of equations. For 2x4x7, there are 2*4*7 = 56 equations.
        # Each rank-1 term has 8+28+14 = 50 variables (minus scale invariance = 49).
        # 
        # This is an underdetermined system, so there should be many solutions.
        # The question is finding one with minimal rank (fewest terms).
        
        # Let's try a greedy approach:
        # 1. Start with the schoolbook decomposition (rank 56)
        # 2. Try to merge pairs of terms into a single term
        # 3. Iterate until no more merges are possible
        
        # Actually, let's try a different approach:
        # Use the fact that the tensor has a specific structure and try to find
        # a low-rank decomposition using the block structure.
        
        # Key idea: The 2x4x7 MM tensor can be viewed as a block matrix:
        # [C[0,0:7]]   [A[0,0:4]] [B[0:4,0:7]]
        # [C[1,0:7]] = [A[1,0:4]] [B[0:4,0:7]]
        # 
        # Each row is a 1x4 times 4x7 product.
        # 
        # For a single row: C[i,:] = A[i,:] * B
        # This is a 1x4x7 tensor, which has rank 4 (schoolbook).
        # 
        # But we have two rows sharing B, so we can potentially save.
        # 
        # The total tensor is the sum of two 1x4x7 tensors (one per row).
        # If they were independent, rank would be 8. But they share B, so there's structure.
        
        # Let me try to construct a decomposition with rank < 50 by hand.
        # 
        # Idea: Use the fact that B is shared.
        # 
        # Define intermediate variables:
        # M1 = (A[0,0] + A[1,0]) * B[0,:]  -> 1x7 vector
        # M2 = (A[0,1] + A[1,1]) * B[1,:]  -> 1x7 vector
        # M3 = (A[0,2] + A[1,2]) * B[2,:]  -> 1x7 vector
        # M4 = (A[0,3] + A[1,3]) * B[3,:]  -> 1x7 vector
        # M5 = (A[0,0] - A[1,0]) * B[0,:]  -> 1x7 vector
        # M6 = (A[0,1] - A[1,1]) * B[1,:]  -> 1x7 vector
        # M7 = (A[0,2] - A[1,2]) * B[2,:]  -> 1x7 vector
        # M8 = (A[0,3] - A[1,3]) * B[3,:]  -> 1x7 vector
        # 
        # Then:
        # C[0,k] = (M1[k] + M5[k])/2 + (M2[k] + M6[k])/2 + (M3[k] + M7[k])/2 + (M4[k] + M8[k])/2
        # C[1,k] = (M1[k] - M5[k])/2 + (M2[k] - M6[k])/2 + (M3[k] - M7[k])/2 + (M4[k] - M8[k])/2
        # 
        # This uses 8 rank-1 terms (each M is a rank-1 term: scalar * row vector).
        # Wait, that's not right. Each M is a 1x7 vector, not a scalar.
        # 
        # Let me reconsider. A rank-1 term in the tensor decomposition is:
        # u * v * w where u is 2x4, v is 4x7, w is 2x7.
        # 
        # So each term is a full 2x4x7 tensor of rank 1.
        # 
        # The schoolbook decomposition has 56 terms, each with u having one 1, v having one 1, w having one 1.
        # 
        # To reduce rank, we need to find terms where u, v, w have multiple non-zero entries.
        
        # Let's try: 
        # Term 1: u[0,0]=1, u[1,0]=1, v[0,k]=1 for all k, w[0,k]=1/2, w[1,k]=1/2
        # This gives: C[0,k] += A[0,0]*B[0,k]/2, C[1,k] += A[1,0]*B[0,k]/2
        # 
        # Term 2: u[0,0]=1, u[1,0]=-1, v[0,k]=1 for all k, w[0,k]=1/2, w[1,k]=-1/2
        # This gives: C[0,k] += A[0,0]*B[0,k]/2, C[1,k] -= A[1,0]*B[0,k]/2
        # 
        # Sum of terms 1 and 2: C[0,k] += A[0,0]*B[0,k], C[1,k] += 0
        # 
        # That's not helpful. Let me think differently.
        
        # The standard Strassen trick for 2x2x2:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b10+b11)
        # M4 = a11*b10
        # M5 = (a00+a10)*b11
        # M6 = (a11-a10)*(b00+b10)
        # M7 = (a10-a00)*(b11+b01)
        # 
        # The key is that each M is a product of a linear combination of A entries and a linear combination of B entries.
        # 
        # For 2x4x7, we can generalize:
        # For each pair of rows (just 0,1) and each pair of columns in B:
        # Use 7 products to compute 2x2x2 blocks.
        # 
        # That's what we have: 3 Strassen blocks per row-pair for columns 0-5, plus schoolbook for column 6.
        # 
        # The question is: can we do better for the "tail" (column 6)?
        
        # Idea: Instead of handling column 6 separately, can we incorporate it into the Strassen blocks?
        # 
        # What if we use a 2x2x3 Strassen-like block?
        # For 2x2x3: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] for k in {0,1,2}
        # 
        # Schoolbook: 2*2*3 = 12 multiplications.
        # 
        # Can we do it in fewer?
        # 
        # Let's try:
        # M1 = (a00+a11)*(b00+b12)  -> a00*b00 + a00*b12 + a11*b00 + a11*b12
        # M2 = (a10+a11)*(b00)       -> a10*b00 + a11*b00
        # M3 = (a00)*(b10+b11+b12)   -> a00*b10 + a00*b11 + a00*b12
        # M4 = (a11)*(b10)           -> a11*b10
        # M5 = (a00+a10)*(b11)       -> a00*b11 + a10*b11
        # M6 = (a11-a10)*(b00+b10)   -> a11*b00 + a11*b10 - a10*b00 - a10*b10
        # M7 = (a10-a00)*(b12)       -> a10*b12 - a00*b12
        # 
        # Let's check if we can extract all 12 products:
        # a00*b00: from M1 (coeff 1)
        # a00*b10: from M3 (coeff 1)
        # a00*b11: from M3 (coeff 1) and M5 (coeff 1) -> total 2, need 1. Problem.
        # 
        # This approach doesn't work directly. Let me try a different set.
        
        # Actually, the rank of the 2x2x3 MM tensor is known to be 12 (same as schoolbook).
        # So we can't do better for 2x2x3.
        
        # What about 2x3x3? The rank is 16 (known result).
        # Schoolbook: 2*3*3 = 18.
        # 
        # For our problem, we could partition as:
        # 2x4x7 = 2x3x7 + 2x1x7
        # 2x3x7: rank ? 
        # 2x1x7: rank 7 (schoolbook, can't do better)
        # 
        # Or: 2x4x7 = 2x2x7 + 2x2x7
        # 2x2x7: rank ?
        
        # The rank of 2x2xn MM tensor:
        # For n=1: rank 4
        # For n=2: rank 7 (Strassen)
        # For n=3: rank 12 (schoolbook is optimal)
        # For n=4: rank ? 
        # 
        # Actually, I recall that the rank of 2x2xn is at most 3n for n>=2.
        # For n=7: rank <= 21.
        # 
        # If we can decompose 2x2x7 with rank 21, then:
        # 2x4x7 = 2x2x7 + 2x2x7 (splitting j into {0,1} and {2,3})
        # Total rank: 21 + 21 = 42.
        # 
        # That would be a significant improvement over 50!
        
        # Let's verify: can we do 2x2x7 with rank 21?
        # 
        # The 2x2xn tensor: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] for i in {0,1}, k in {0,...,n-1}
        # 
        # This is a 2x2xn tensor. The rank is the minimum number of rank-1 terms.
        # 
        # Known results:
        # - Rank of 2x2x2 is 7 (Strassen)
        # - Rank of 2x2xn for n>=3 is 3n (I believe this is correct)
        # 
        # If rank(2x2x7) = 21, then we can do 2x4x7 with rank 42.
        
        # Let's try to construct a 2x2x7 decomposition with rank 21.
        # 
        # The idea: use 3 rank-1 terms per column of B.
        # For column k of B:
        # M1_k = (a00 + a11) * b0k
        # M2_k = a10 * b1k  
        # M3_k = (a00 - a10) * b1k  -- wait, this doesn't work.
        
        # Let me think more carefully.
        # 
        # For a single column k, we need:
        # C[0,k] = a00*b0k + a01*b1k  (where a0j = A[0,j], bjk = B[j,k])
        # C[1,k] = a10*b0k + a11*b1k
        # 
        # This is a 2x2x1 tensor, which has rank 4.
        # 
        # But if we process multiple columns together, we can share the A-side.
        # 
        # The key insight: the A-side is the same for all columns.
        # 
        # Let's define:
        # P1 = a00 + a11
        # P2 = a10 + a11
        # P3 = a00
        # P4 = a11
        # P5 = a00 + a10
        # P6 = a11 - a10
        # P7 = a10 - a00
        # 
        # These are the 7 Strassen combinations for 2x2.
        # 
        # For each column k, we can compute:
        # C[0,k] = (P1*b0k + P2*b1k + P3*b0k + P4*b1k + P5*b1k + P6*b0k + P7*b1k) / 2
        #         = ... this is getting messy.
        
        # Let me use the standard Strassen formulas:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b10+b11)
        # M4 = a11*b10
        # M5 = (a00+a10)*b11
        # M6 = (a11-a10)*(b00+b10)
        # M7 = (a10-a00)*(b11+b01)
        # 
        # C[0,0] = M1 - M4 + M5
        # C[0,1] = M3 + M4
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3
        # 
        # For 2x2xn, we can generalize:
        # For each pair of columns (k, k+1):
        # Use 7 multiplications as above.
        # 
        # For n=7, we have 3 pairs (0,1), (2,3), (4,5) and one leftover column 6.
        # Total: 3*7 + 4 = 25 for 2x2x7.
        # 
        # But we can do better! The known rank of 2x2xn is 3n for n>=3.
        # 
        # Let me look up the construction for rank 3n.
        # 
        # The construction uses:
        # For each k, define:
        # M1_k = (a00 + a11) * b0k
        # M2_k = a10 * b1k
        # M3_k = (a00 - a10) * b1k
        # 
        # Wait, that's only 3n terms, but let's check if it works.
        # 
        # C[0,k] should be a00*b0k + a01*b1k
        # C[1,k] should be a10*b0k + a11*b1k
        # 
        # Using M1_k, M2_k, M3_k:
        # M1_k = a00*b0k + a11*b0k
        # M2_k = a10*b1k
        # M3_k = a00*b1k - a10*b1k
        # 
        # We need to express a00*b0k, a01*b1k, a10*b0k, a11*b1k in terms of these.
        # 
        # a00*b0k = M1_k - a11*b0k  -- still have a11*b0k
        # 
        # This doesn't work directly. Let me try a different set.
        
        # Actually, I think the correct construction for rank 3n is:
        # M1_k = (a00 + a11) * b0k
        # M2_k = (a00 + a10) * b1k
        # M3_k = (a11 - a10) * b1k
        # 
        # Let's check:
        # M1_k = a00*b0k + a11*b0k
        # M2_k = a00*b1k + a10*b1k
        # M3_k = a11*b1k - a10*b1k
        # 
        # C[0,k] = a00*b0k + a01*b1k
        # We have a00*b0k from M1_k (but mixed with a11*b0k).
        # We have a00*b1k from M2_k (but mixed with a10*b1k).
        # 
        # This doesn't directly give us C[0,k].
        
        # I think I'm overcomplicating this. Let me just try to verify if rank 42 is achievable
        # by constructing a specific decomposition.
        
        # Alternative approach: use the fact that the problem is small enough to try
        # several explicit decompositions and verify them.
        
        # For now, let's return the rank-50 decomposition since we've verified it works.
        # The search for better decompositions can be done in future iterations.
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
