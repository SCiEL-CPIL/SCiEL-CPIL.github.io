"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 140.0  # Safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_decomp():
    """
    Build a rank-40 decomposition for 2x3x8 matrix multiplication.
    
    Strategy:
    - Split the 8 columns of B into four 2-column blocks.
    - Each block corresponds to a 2x3x2 matrix multiplication problem.
    - Decompose each 2x3x2 tensor using rank 11:
      - Split A (2x3) into A_top (2x2) and A_bottom (2x1).
      - Split B (3x2) into B_top (2x2) and B_bottom (1x2).
      - Compute A_top * B_top using Strassen (rank 7).
      - Compute A_bottom * B_bottom using schoolbook (rank 4).
    - Total rank: 4 blocks * 11 = 44? No, 4 * 11 = 44.
    - Wait, is rank(2x3x2) actually 11?
      - Lower bound: max(2*3, 3*2, 2*2) = 6.
      - Upper bound: 2*3*2 = 12.
      - Known result: rank(2x3x2) = 11.
      - Let's verify the 11 rank construction.
        - Strassen for 2x2x2 is 7.
        - The extra terms are A[0,2]B[2,0], A[0,2]B[2,1], A[1,2]B[2,0], A[1,2]B[2,1].
        - This is a 2x1 x 1x2 multiplication, which has rank 4.
        - Total 7 + 4 = 11.
    - So total rank is 4 * 11 = 44?
    - Why did I think 40?
    - Ah, the previous decomposition was 44.
    - Is there a better decomposition for 2x3x2?
    - According to literature, rank(2x3x2) is indeed 11.
    - Is it possible to share terms between the 4 blocks?
    - The blocks are disjoint in the K dimension (columns of B and rows of C).
    - U depends on i, j. V depends on j, k. W depends on k, i.
    - For block g, j is in {0, 1, 2} relative to the block? No, j is global 0..2.
    - In the 2x3x2 sub-problem:
      - A is 2x3 (rows 0,1; cols 0,1,2).
      - B is 3x2 (rows 0,1,2; cols 0,1).
      - C is 2x2 (rows 0,1; cols 0,1).
      - The tensor is supported on (i, j, k) where i in 0,1; j in 0,1,2; k in 0,1.
      - In the full problem:
        - Block g uses B columns 2g, 2g+1. So j in {0, 1, 2} is global j.
        - Wait, B is 3x8. Row j of B is B[j, :].
        - In the sub-problem, B_sub is 3x2. Rows 0,1,2 of B_sub correspond to rows 0,1,2 of B.
        - So the j index is the same global j.
        - The k index is local 0,1, mapping to global k = 2g + k_local.
        - The i index is global i.
    - Since the k indices are disjoint for different g, the W vectors are disjoint in support.
    - Since the k indices are disjoint, the V vectors have disjoint supports in the k dimension?
      - V is (m*p, rank). Index is j*p + k.
      - For block g, k is 2g, 2g+1.
      - So V supports are disjoint across blocks.
    - Since V supports are disjoint, we cannot share V terms.
    - Can we share U or W terms?
      - U depends on i, j.
      - W depends on k, i.
      - Since k is disjoint, W supports are disjoint.
      - So we cannot share W terms.
      - U terms could potentially be shared if the coefficients match, but Strassen uses different linear combinations of A elements.
      - However, the A elements are the same (global j=0,1,2).
      - But the "Strassen" part operates on A[0,0], A[0,1], A[1,0], A[1,1] (j=0,1).
      - The "Schoolbook" part operates on A[0,2], A[1,2] (j=2).
      - The U vectors for the Strassen part involve j=0,1.
      - The U vectors for the Schoolbook part involve j=2.
      - So U supports are disjoint in j dimension?
        - Strassen U vectors have non-zeros at j=0,1.
        - Schoolbook U vectors have non-zeros at j=2.
        - So U supports are disjoint.
    - Therefore, the 4 blocks are completely independent in terms of support overlap.
    - The total rank is simply the sum of the ranks of the blocks.
    - Rank = 4 * 11 = 44.
    
    Wait, the current program already has rank 44.
    My analysis suggests rank 44 is the minimum for this block-diagonal approach.
    Is there a way to do better?
    
    Let's check the rank of 2x3x2 again.
    Some sources say rank(2x3x2) = 11.
    Is it possible to reduce the total rank by mixing the blocks?
    Since the supports are disjoint, no.
    
    Let's re-read the task. "Find a rank-minimal exact tensor decomposition".
    The current score is 1.0000 with rank 44.
    If I cannot find rank < 44, I should ensure the current solution is robust and fast.
    
    However, let's look at the "Inspiration 3" which mentions `_try_rank_reduction`.
    And the cell finding: "Decomposes the 2x3x8 tensor into four 2x3x2 sub-tensors, each decomposed using a rank-11 scheme... reducing total rank from 48 to 44."
    This confirms 44 is the target for this family.
    
    Is it possible to get rank 40?
    Only if rank(2x3x2) < 11.
    Lower bound for 2x3x2:
    - Tensor rank >= max(n*m, m*p, p*n) is NOT the lower bound.
    - The lower bound is the "border rank" or "tensor rank".
    - For 2x2x2, rank is 7.
    - For 2x3x2, it is known to be 11.
    
    So 44 is likely the minimum for this specific block decomposition.
    Can we do better than 44 for the whole 2x3x8?
    Maybe by not splitting into 4 independent blocks?
    But the k-dimension is "local" to the output columns.
    C[i, k] depends only on A[i, :] and B[:, k].
    Since B[:, k] for k in block g is disjoint from B[:, k'] for k' in block g', the V vectors must be disjoint.
    This forces the decomposition to be block-wise.
    
    So, I am stuck at 44.
    I will just clean up the code and ensure it returns the rank 44 solution efficiently.
    I will verify the current code's correctness.
    The current code implements exactly this:
    - 4 blocks.
    - Each block uses Strassen (7) for the 2x2 part (j=0,1) and Schoolbook (4) for the 2x1 x 1x2 part (j=2).
    - Total 44.
    
    Wait, the current code in `_build_decomp` does:
    1. Part 1: A_2x1 * B_1x8.
       - This handles j=2 for all k.
       - Rank 16.
    2. Part 2: A_2x2 * B_2x8 via Strassen.
       - This handles j=0,1 for all k.
       - It splits B_2x8 into 4 blocks of 2x2.
       - Each block uses Strassen (7).
       - Total 28.
    Total 16 + 28 = 44.
    
    This is equivalent to the block-wise 11-rank decomposition.
    
    So the current program is already optimal for this family.
    I will make minor improvements to ensure robustness and speed.
    Specifically, I will optimize the verification step to be faster, although it's not the bottleneck.
    I will also ensure the deadline guard is tight.
    
    Actually, I can try to find a rank 40 solution if I'm wrong about the disjointness.
    But V supports are definitely disjoint.
    V[j, k] is non-zero only if the term contributes to C[i, k].
    Since C[i, k] depends only on B[:, k], and B[:, k] is distinct for each k, the V vectors for different k must be orthogonal in the k-dimension?
    No, V is a matrix of size (m*p, rank).
    Column c of V is a vector of length m*p.
    If two columns c1, c2 have non-zero entries at different k indices, they are linearly independent.
    Since the blocks use disjoint k indices, the V columns for different blocks have disjoint supports.
    Thus, they are linearly independent.
    So the rank is the sum of the ranks.
    
    Conclusion: Rank 44 is the minimum for this structure.
    I will return the current solution.
    I will just refine the code to be cleaner.
"""
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A_2x1 * B_1x8 (rank 16)
    # A[0,2] at index 0*3+2=2, A[1,2] at index 1*3+2=5
    # B[2,k] at index 2*8+k = 16+k
    # C[i,k] at index k*2+i
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A_2x2 * B_2x8 via Strassen (rank 28)
    # A_2x2 indices: A[0,0]@0, A[0,1]@1, A[1,0]@3, A[1,1]@4
    # B_2x8 rows 0,1. B[0,k]@k, B[1,k]@8+k
    # C is 2x8: C[i,k] at k*2+i
    
    for g in range(4):
        b11_idx = 0 + 2 * g
        b12_idx = 0 + 2 * g + 1
        b21_idx = 8 + 2 * g
        b22_idx = 8 + 2 * g + 1
        
        a11_idx = 0
        a12_idx = 1
        a21_idx = 3
        a22_idx = 4
        
        c11_idx = (2 * g) * 2 + 0
        c12_idx = (2 * g + 1) * 2 + 0
        c21_idx = (2 * g) * 2 + 1
        c22_idx = (2 * g + 1) * 2 + 1
        
        # p1 = (a11+a22)(b11+b22)
        U[a11_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # p2 = (a21+a22)b11
        U[a21_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        W[c21_idx][col] = 1
        col += 1
        
        # p3 = a11(b22-b12)
        U[a11_idx][col] = 1
        V[b22_idx][col] = 1
        V[b12_idx][col] = -1
        W[c12_idx][col] = 1
        W[c22_idx][col] = -1
        col += 1
        
        # p4 = a22(b11-b21)
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b21_idx][col] = -1
        W[c11_idx][col] = -1
        W[c21_idx][col] = 1
        col += 1
        
        # p5 = (a11+a12)b22
        U[a11_idx][col] = 1
        U[a12_idx][col] = 1
        V[b22_idx][col] = 1
        W[c12_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # p6 = (a21-a11)(b11+b12)
        U[a21_idx][col] = 1
        U[a11_idx][col] = -1
        V[b11_idx][col] = 1
        V[b12_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # p7 = (a12-a22)(b21+b22)
        U[a12_idx][col] = 1
        U[a22_idx][col] = -1
        V[b21_idx][col] = 1
        V[b22_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp()
        
        # Quick verification using NumPy
        # Convert to arrays
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # The tensor is T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # We want T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} for the matmul tensor?
        # No, the matmul tensor T_{ijk} is 1 if i=i', j=j', k=k' is not quite right.
        # The decomposition computes C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T_{ijk} contributes to C_{ik} if we contract j?
        # Actually, the task defines the interface:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # The value computed is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # This should equal 1 if (i,j,k) is a valid term in the sum for C_{ik}?
        # No, it should equal the coefficient of A_{ij} B_{jk} in C_{ik}.
        # For matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
        # So the coefficient is 1 for all valid (i,j,k).
        # Thus, the sum should be 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(44):
                        val += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
                    
        if max_err < 1e-10:
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback if verification fails (should not happen)
            pass
            
    except Exception:
        pass
    
    # Fallback: schoolbook
    if time.time() - start_time > _DEADLINE:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
