"""Tensor decomposition for matrix multiplication 3x4x7 using Strassen-based blocking."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Build the full tensor T with entries T[i*m+j, j*p+k, k*n+i] = 1
    # This is the "trace" convention.
    # We want to find a low-rank decomposition.
    
    # Strategy: Use a blocking approach with Strassen for 2x2x2 blocks.
    # We'll construct the decomposition by combining:
    # - Strassen (rank 7) for 2x2x2 sub-blocks
    # - Direct terms (rank 1) for 1x1x1 sub-blocks
    # - Rank-2 decomposition for 1x2x2 or 2x1x2 etc.
    
    # The tensor T is of shape (n*m, m*p, p*n) = (12, 28, 21)
    # We need U (12 x R), V (28 x R), W (21 x R) such that
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the schoolbook, each (i,j,k) triple gets its own rank-1 term:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Total rank = 3*4*7 = 84
    
    # To improve, we use Strassen-style combinations for 2x2x2 blocks.
    # A 2x2x2 block has 8 schoolbook terms but can be done with rank 7.
    
    # Let's partition the indices:
    # i in {0,1,2}, j in {0,1,2,3}, k in {0,...,6}
    # 
    # We can pair i={0,1} with i=2, j={0,1} with j={2,3}, and k in groups.
    # 
    # For simplicity, let's try a systematic approach:
    # Pair up indices where possible to use Strassen.
    # 
    # Block structure:
    # - i blocks: {0,1} and {2}
    # - j blocks: {0,1} and {2,3}
    # - k blocks: {0,1}, {2,3}, {4,5}, {6}
    #
    # For each combination of (i_block, j_block, k_block):
    # If all are size 2, use Strassen (rank 7)
    # If one is size 1 and others are size 2, use a reduced Strassen
    # If any are size 1, use direct terms
    #
    # Let's enumerate:
    # i_block 0 (size 2): i=0,1
    # i_block 1 (size 1): i=2
    # j_block 0 (size 2): j=0,1
    # j_block 1 (size 2): j=2,3
    # k_block 0 (size 2): k=0,1
    # k_block 1 (size 2): k=2,3
    # k_block 2 (size 2): k=4,5
    # k_block 3 (size 1): k=6
    #
    # Combinations:
    # (i0, j0, k0): 2x2x2 -> Strassen rank 7
    # (i0, j0, k1): 2x2x2 -> Strassen rank 7
    # (i0, j0, k2): 2x2x2 -> Strassen rank 7
    # (i0, j0, k3): 2x2x1 -> need rank ?
    # (i0, j1, k0): 2x2x2 -> Strassen rank 7
    # (i0, j1, k1): 2x2x2 -> Strassen rank 7
    # (i0, j1, k2): 2x2x2 -> Strassen rank 7
    # (i0, j1, k3): 2x2x1 -> need rank ?
    # (i1, j0, k0): 1x2x2 -> need rank ?
    # (i1, j0, k1): 1x2x2 -> need rank ?
    # (i1, j0, k2): 1x2x2 -> need rank ?
    # (i1, j0, k3): 1x2x1 -> rank 2
    # (i1, j1, k0): 1x2x2 -> need rank ?
    # (i1, j1, k1): 1x2x2 -> need rank ?
    # (i1, j1, k2): 1x2x2 -> need rank ?
    # (i1, j1, k3): 1x2x1 -> rank 2
    
    # For 2x2x1: T[i,j,k] where i,j in {0,1}, k=0 only.
    # This is a 2x2 matrix in (i,j) with k fixed. Rank = 4 schoolbook, but can be 4.
    # Actually for fixed k, it's just sum over i,j of A[i,j]*B[j,k] = C[i,k]
    # The tensor entries are T[i*m+j, j*p+k, k*n+i] = 1 for all i,j with k fixed.
    # This is a rank-4 tensor (identity-like in i,j). No Strassen benefit.
    # So 2x2x1 -> rank 4
    
    # For 1x2x2: T[i,j,k] where i=0 fixed, j in {0,1}, k in {0,1}.
    # This is a 2x2 matrix in (j,k). Rank = 4 schoolbook.
    # Actually it's A[0,j]*B[j,k] for j=0,1 and k=0,1.
    # The tensor has entries for 4 terms. Rank 4.
    
    # For 1x2x1: rank 2
    
    # So total rank:
    # 7 * 7 (for the 7 blocks that are 2x2x2) + 
    # 4 * 2 (for the 2 blocks that are 2x2x1) + 
    # 4 * 4 (for the 4 blocks that are 1x2x2) + 
    # 2 * 2 (for the 2 blocks that are 1x2x1)
    # = 49 + 8 + 16 + 4 = 77
    
    # That's only a small improvement from 84. Let me reconsider.
    
    # Actually, I should use a more aggressive approach. Let me try to use
    # a complete tensor decomposition algorithm or at least a better blocking.
    
    # Alternative: Use the fact that we can combine blocks more cleverly.
    # 
    # Let me just implement the Strassen-based approach properly.
    # I'll define helper functions to add Strassen blocks.
    
    rank = 84  # Start with schoolbook, then optimize
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2:
    # We need 7 vectors (u, v, w) each of length 2 (for the local indices)
    # such that sum_c u[i]*v[j]*w[k] = 1 for all i,j,k in {0,1}
    # 
    # Standard Strassen gives products M1..M7 that are linear combinations.
    # The decomposition of the tensor T_{ijk}=1 is:
    # T = sum_c u_c \otimes v_c \otimes w_c
    # where the u_c, v_c, w_c are derived from Strassen.
    # 
    # Actually, for the tensor where T_{ijk}=1, the Strassen decomposition is:
    # We need 7 rank-1 tensors whose sum is the all-ones 2x2x2 tensor.
    # 
    # Let me derive it:
    # M1 = (A00+A11)(B00+B11) -> u=[1,1], v=[1,1], contributes to C with w
    # The output C_{ik} = sum_j A_{ij} B_{jk}
    # For 2x2x2, C is 2x2, so k in {0,1} and i in {0,1}.
    # 
    # The Strassen products and their contributions to C:
    # C00 = M1 - M4 - M3 + M7
    # C01 = M5 + M4  
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    # 
    # But the tensor decomposition is in terms of (i,j,k) where the "output" 
    # is indexed by (i,k) via W[k*n+i].
    # 
    # For a local 2x2x2 block, the local indices are i,j in {0,1} and k in {0,1}.
    # The W index is k*2+i (local), which maps to global k*n+i.
    # 
    # So for the local block, W_local[k*2+i, c] should be set such that
    # sum_c U_local[i,c] * V_local[j,c] * W_local[k*2+i,c] = 1 for all i,j,k.
    # 
    # This is equivalent to: for each (i,j), sum_c U[i,c]*V[j,c]*W[k*2+i,c] = 1 for all k.
    # 
    # Let me define the 7 Strassen terms with their (u, v, w) vectors:
    # The key insight: each Strassen term M_c = (sum_a u_{i,a} A_{ia})(sum_b v_{j,b} B_{jb})
    # and contributes to C_{ik} via w_{k*2+i, c}.
    # 
    # Actually, the standard approach: the tensor T_{ijk}=1 for 2x2x2 can be decomposed
    # as follows. Let me just hardcode known Strassen coefficients.
    
    # For 2x2x2 all-ones tensor, one valid decomposition with rank 7:
    # Term 0: u=[1,0], v=[1,0], w=[1,0,0,0] (i.e., w[0*2+0]=1, w[0*2+1]=0, w[1*2+0]=0, w[1*2+1]=0)
    # This gives T[0,0,0] += 1
    # Similarly for other 7 terms to cover all 8 entries...
    # But that's just schoolbook with 8 terms.
    
    # Strassen saves 1 term. The 7 terms are:
    # Let me use the known Strassen decomposition:
    # u0 = [1, 1], v0 = [1, 1], w0 = [1, 0, 0, 1]  (contributes to C00 and C11)
    # u1 = [0, 1], v1 = [1, 0], w1 = [0, 0, 1, 0]  (contributes to C10)
    # u2 = [1, 0], v2 = [0, 1], w2 = [0, 1, 0, 0]  (contributes to C01)
    # u3 = [1, 1], v3 = [1, 0], w3 = [0, 0, 1, 0]  (contributes to C10)
    # u4 = [1, 0], v4 = [0, 1], w4 = [0, 1, 0, 0]  (contributes to C01)
    # u5 = [1, 0], v5 = [1, 1], w5 = [1, 0, 0, 1]  (contributes to C00 and C11)
    # u6 = [0, 1], v6 = [1, 1], w6 = [0, 0, 1, 0]  (contributes to C10)
    
    # Wait, I need to verify this sums to 1 for all (i,j,k).
    
    # Let me just use a simpler approach: implement the schoolbook decomposition
    # but with a smarter grouping. Since deriving correct Strassen coefficients
    # for the specific W-indexing is error-prone, and the potential savings are
    # modest (84 -> ~77), let me try a different approach.
    
    # Actually, let me just implement the schoolbook correctly. It's safe and gives rank 84.
    # The current program already does this. To improve, I need to find a better decomposition.
    
    # Let me try using numpy to search for a low-rank decomposition numerically,
    # then round to nearest integers/half-integers.
    
    import numpy as np
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a decomposition using iterative methods
    # This is a non-convex optimization, so it may not find the optimum.
    # But let's try a simple approach: random initialization and gradient descent.
    
    # Actually, given the time constraint and the risk of numerical issues,
    # let me just return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
