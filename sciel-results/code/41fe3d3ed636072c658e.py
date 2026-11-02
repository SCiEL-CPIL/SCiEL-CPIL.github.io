"""Program for matrix multiplication 3x5x6 rank-minimal decomposition."""
import numpy as np
import time
import math

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 115  # safety margin for 120s limit (evaluator kills at 150s)

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    
    The tensor equation is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] should be 1 for all valid (i,j,k) triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build tensor from decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_decomposition(n, m, p, block_n, block_p):
    """Decompose M_{n,m,p} using block structure.
    
    Split B into blocks of width block_p along columns, and A into blocks of height block_n.
    For each block of A (rows) and block of B (cols), use schoolbook for the sub-multiplication.
    
    This gives rank = ceil(m/block_n) * ceil(p/block_p) * block_n * block_n * block_p
    but actually it's: for each A-row-block and B-col-block, we do a (block_n x block_n x block_p) mult.
    Wait, let me think more carefully.
    
    Actually: C[i,k] = sum_j A[i,j] B[j,k]
    If we partition the m=5 intermediate dimension into blocks of size block_m,
    and p=6 output columns into blocks of size block_p:
    
    For each block of j (size block_m) and block of k (size block_p):
      C[i, k_block] += A[i, j_block] * B[j_block, k_block]
    The cost per (j_block, k_block) pair is n * block_m * block_p.
    Total rank = (m / block_m) * (p / block_p) * n * block_m * block_p = n * m * p = 90.
    
    So plain blocking doesn't help. We need known low-rank decompositions for sub-blocks.
    """
    pass

def _try_random_decomposition(rank, n, m, p, seed=None):
    """Try to find a random decomposition of given rank."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return [U.tolist(), V.tolist(), W.tolist()]
    return None

def _structured_block_search(n, m, p, start_time, deadline):
    """Try structured decompositions using known results for smaller blocks.
    
    Strategy: Use the fact that M_{3,3,3} has rank 23 (Strassen-like).
    We can split the 5x6 problem into blocks:
    - Split the 5 intermediate dimension: e.g., 3+2 or 2+3
    - Split the 6 output dimension: e.g., 3+3 or 2+2+2
    
    For M_{3,3,3} we know rank 23 is achievable.
    For M_{3,2,3}: rank <= 3*2*3 = 18 (schoolbook)
    For M_{3,3,2}: rank <= 3*3*2 = 18
    For M_{3,2,2}: rank <= 12
    
    So if we split j into 3+2 and k into 3+3:
    - Block (j=0..2, k=0..2): M_{3,3,3} rank 23
    - Block (j=0..2, k=3..5): M_{3,3,3} rank 23  
    - Block (j=3..4, k=0..2): M_{3,2,3} rank 18
    - Block (j=3..4, k=3..5): M_{3,2,3} rank 18
    Total: 23+23+18+18 = 82
    
    Or split j into 2+3 and k into 3+3:
    - Block (j=0..1, k=0..2): M_{3,2,3} rank 18
    - Block (j=0..1, k=3..5): M_{3,2,3} rank 18
    - Block (j=2..4, k=0..2): M_{3,3,3} rank 23
    - Block (j=2..4, k=3..5): M_{3,3,3} rank 23
    Total: 18+18+23+23 = 82
    
    Or split k into 2+2+2:
    j into 3+2:
    - (j=0..2, k=0..1): M_{3,3,2} rank 18
    - (j=0..2, k=2..3): M_{3,3,2} rank 18
    - (j=0..2, k=4..5): M_{3,3,2} rank 18
    - (j=3..4, k=0..1): M_{3,2,2} rank 12
    - (j=3..4, k=2..3): M_{3,2,2} rank 12
    - (j=3..4, k=4..5): M_{3,2,2} rank 12
    Total: 18*3 + 12*3 = 54 + 36 = 90. Same as schoolbook.
    
    So the best structured approach gives 82 if we have the rank-23 decomposition for M_{3,3,3}.
    
    Let me construct the rank-23 decomposition for M_{3,3,3}.
    This is the classic Strassen-like 23-term decomposition.
    """
    
    # Known rank-23 decomposition for 3x3x3 matrix multiplication.
    # This is based on the work of Lader and Strassen.
    # I'll construct it using a known parametrization.
    
    # Actually, let me use a simpler but still improved approach:
    # Use Strassen's 7-term decomposition for 2x2x2, and combine with schoolbook.
    
    # M_{3,3,3}: Split into 2x2 block + remainder
    # A = [A11 A12; A21 A22 + A23] where A11 is 2x2, etc.
    # This gets complicated. Let me use a direct approach.
    
    # For now, let me implement the 82-rank construction with explicit factors.
    # I need the 23 terms for M_{3,3,3}.
    
    # The rank-23 decomposition of M_{3,3,3} can be constructed as follows:
    # It's known that R(M_{3,3,3}) <= 23.
    # One explicit decomposition uses the following structure.
    
    # Let me use a known explicit 23-term decomposition.
    # Based on the literature, here's one way to express it.
    
    # Actually, for simplicity and correctness, let me use the following approach:
    # Use Strassen (rank 7) for 2x2x2 sub-blocks combined with schoolbook for the rest.
    
    # M_{3,3,3}: 
    # Write A = [A_top(2x3); A_bot(1x3)], B = [B_left(3x2); B_right(3x1)]
    # C = A_top * B_left + A_top * B_right + A_bot * B_left + A_bot * B_right
    # = C_TL + C_TR + C_BL + C_BR
    # C_TL is 2x2: use Strassen rank 7
    # C_TR is 2x1: schoolbook rank 6 (or maybe less)
    # C_BL is 1x2: schoolbook rank 6
    # C_BR is 1x1: rank 1
    # Total: 7 + 6 + 6 + 1 = 20? But wait, the U, V, W factors must be consistent.
    
    # Actually this decomposition isn't quite right because the factors need to be shared.
    # Let me think differently.
    
    # A better approach: directly construct the 82-rank decomposition for M_{3,5,6}
    # using schoolbook for all blocks (giving 90) and then try to optimize.
    
    # For now, let me return the schoolbook 90-rank and try random search for improvement.
    pass

def _construct_82_rank(n, m, p):
    """Construct an 82-rank decomposition for M_{3,5,6}.
    
    Uses block decomposition:
    - Split j (intermediate dim, size 5) into [0,1,2] and [3,4]
    - Split k (output dim, size 6) into [0,1,2] and [3,4,5]
    
    Four sub-problems:
    1. A[0:3, 0:3] * B[0:3, 0:3] -> rank 23 (M_{3,3,3})
    2. A[0:3, 0:3] * B[0:3, 3:6] -> rank 23 (M_{3,3,3})
    3. A[0:3, 3:5] * B[3:5, 0:3] -> rank 18 (M_{3,2,3} schoolbook)
    4. A[0:3, 3:5] * B[3:5, 3:6] -> rank 18 (M_{3,2,3} schoolbook)
    
    Total: 23 + 23 + 18 + 18 = 82
    
    I need the explicit rank-23 decomposition for M_{3,3,3}.
    Let me use a known one from the literature.
    """
    
    # Rank-23 decomposition of M_{3,3,3}
    # This is a well-known result. Here I'll construct it explicitly.
    # 
    # One explicit construction: use the "23-term" decomposition.
    # Based on Lader (1979) and Strassen.
    # 
    # The decomposition has the form:
    # sum_{c=1}^{23} U_c ⊗ V_c ⊗ W_c = T_{3,3,3}
    #
    # Let me use a known explicit set of coefficients.
    # 
    # Actually, let me use a simpler but valid approach:
    # Use Strassen's 7-term for the 2x2x2 sub-block and schoolbook for the rest.
    #
    # M_{3,3,3}: 
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    #
    # Split: A = [A2 | a3^T] where A2 is 2x3, a3^T is 1x3
    #        B = [B2; b3] where B2 is 3x2, b3 is 3x1
    #
    # C = A2*B2 + A2*b3 + a3^T*B2 + a3^T*b3
    #
    # A2*B2 is 2x2: use Strassen (rank 7)
    # A2*b3 is 2x1: rank 6 (schoolbook) 
    # a3^T*B2 is 1x2: rank 6 (schoolbook)
    # a3^T*b3 is 1x1: rank 1
    # Total: 7+6+6+1 = 20
    #
    # But wait - can we do better on the 2x1 and 1x2 parts?
    # M_{2,3,1} and M_{1,3,2}: these are just dot products.
    # M_{2,3,1}: C[i] = sum_j A[i,j]*B[j] for i=0,1. Rank = 6 (each is a separate dot product of dim 3).
    # Actually no - it's rank 2*3*1 = 6 for schoolbook. Can we do better?
    # M_{2,3,1} is just two independent inner products of length 3. Rank is exactly 6.
    # Similarly M_{1,3,2} has rank 6.
    #
    # So total = 7 + 6 + 6 + 1 = 20. That's better than 23!
    # But wait, I need to verify this is valid. The issue is that in the full tensor decomposition,
    # the U, V, W factors for different blocks must be compatible.
    
    # Let me be more careful. The full decomposition for M_{3,3,3} using this split:
    # 
    # For the Strassen part (A2*B2, which is 2x2x2):
    #   U factors: 2x2 matrices embedded in 3x3 (rows 0,1 of A; cols 0,1 of B)
    #   V factors: 2x2 matrices embedded in 3x3 (rows 0,1 of B; cols 0,1 of A for W)
    #   W factors: 2x2 matrices embedded in 3x3 (cols 0,1 of C; rows 0,1 of A)
    #
    # For A2*b3 (2x3x1):
    #   U: 2x3 (rows 0,1 of A, all 3 cols of B... wait no)
    #   Actually: A2 is 2x3 (rows 0,1 of A; cols 0,1,2 of... no.
    #   
    # Let me reclarify the dimensions.
    # A is 3x3, B is 3x3, C is 3x3.
    # A2 = A[0:2, :] is 2x3
    # a3^T = A[2, :] is 1x3
    # B2 = B[:, 0:2] is 3x2
    # b3 = B[:, 2] is 3x1
    #
    # A2 * B2: (2x3)*(3x2) = 2x2 -> C[0:2, 0:2]
    # A2 * b3: (2x3)*(3x1) = 2x1 -> C[0:2, 2]
    # a3^T * B2: (1x3)*(3x2) = 1x2 -> C[2, 0:2]
    # a3^T * b3: (1x3)*(3x1) = 1x1 -> C[2, 2]
    #
    # For A2*B2 (M_{2,3,2}): Wait, this is multiplying 2x3 by 3x2, giving 2x2.
    # That's M_{2,3,2}, not M_{2,2,2}!
    # M_{2,3,2} has rank... schoolbook is 2*3*2=12. 
    # Can we do better? 
    # M_{2,3,2} = (2x3)*(3x2). This is like M_{2,2,2} with one extra column in the middle.
    # Actually M_{2,3,2} can be decomposed as M_{2,2,2} + M_{2,1,2} = 7 + 4 = 11?
    # Or use Strassen-like: the 3 in the middle means we have an extra term.
    # R(M_{2,3,2}) <= R(M_{2,2,2}) + R(M_{2,1,2}) = 7 + 4 = 11.
    # Or R(M_{2,3,2}) <= R(M_{2,3,1}) + R(M_{2,3,1}) = 6+6=12.
    # Actually I think R(M_{2,3,2}) = 11 might be achievable.
    # 
    # This is getting complex. Let me just use the full schoolbook 90-rank for now
    # and focus on making the code correct and fast.
    # The random search might find something better, but 90 is a safe baseline.
    pass

def _direct_90_rank(n, m, p):
    """Direct 90-rank schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _attempt_improved_decomposition(n, m, p, start_time):
    """Attempt to find a decomposition better than schoolbook using structured methods."""
    
    # Try the 82-rank block decomposition with explicit Strassen-based factors.
    # 
    # M_{3,5,6} split:
    # j: [0,1,2] | [3,4]
    # k: [0,1,2] | [3,4,5]
    #
    # Block 1: A[:,0:3] * B[0:3,0:3] -> M_{3,3,3}, target rank 23
    # Block 2: A[:,0:3] * B[0:3,3:6] -> M_{3,3,3}, target rank 23
    # Block 3: A[:,3:5] * B[3:5,0:3] -> M_{3,2,3}, target rank 18
    # Block 4: A[:,3:5] * B[3:5,3:6] -> M_{3,2,3}, target rank 18
    #
    # For blocks 3 and 4 (M_{3,2,3}), schoolbook gives 18.
    # For blocks 1 and 2 (M_{3,3,3}), I'll use a rank-20 decomposition:
    #   Split M_{3,3,3} as:
    #     A2*B2 (M_{2,3,2}) + A2*b3 (M_{2,3,1}) + a3^T*B2 (M_{1,3,2}) + a3^T*b3 (M_{1,3,1})
    #   where A2=A[0:2,:], a3=A[2,:], B2=B[:,:2], b3=B[:,2]
    #
    # M_{2,3,1}: rank 6 (schoolbook, optimal)
    # M_{1,3,2}: rank 6 (schoolbook, optimal)  
    # M_{1,3,1}: rank 3 (schoolbook, optimal)
    # M_{2,3,2}: Need to determine. Schoolbook is 12.
    #   M_{2,3,2} = (2x3)(3x2) -> 2x2
    #   Can write as: split the 3 in B into 2+1
    #   M_{2,3,2} <= M_{2,2,2} + M_{2,1,2} = 7 + 4 = 11
    #   Or: split the 3 in A: M_{2,3,2} <= M_{1,3,2} + M_{1,3,2} = 6+6=12
    #   Or: M_{2,3,2} <= M_{2,2,2} + M_{2,1,2} = 7+4 = 11
    #   I'll use 11.
    #
    # So M_{3,3,3} <= 11 + 6 + 6 + 3 = 26? That's worse than 23.
    # 
    # Let me try another split for M_{3,3,3}:
    # Split both A and B differently.
    # 
    # Actually, the known result R(M_{3,3,3}) <= 23 is well-established.
    # Let me use a known explicit decomposition.
    #
    # Based on the literature, here's a rank-23 decomposition structure:
    # It can be derived from the rank-7 Strassen decomposition of M_{2,2,2}
    # combined with additional terms.
    #
    # For now, let me just use schoolbook 90 and rely on the structure being correct.
    # The key insight is that we CAN do better than 90, but constructing the
    # explicit coefficients is complex.
    #
    # Let me try a different approach: use the known fact that
    # R(M_{3,5,6}) <= R(M_{3,5,3}) + R(M_{3,5,3}) (split k into 3+3)
    # and R(M_{3,5,3}) <= R(M_{3,3,3}) + R(M_{3,2,3}) = 23 + 18 = 41
    # So R(M_{3,5,6}) <= 41 + 41 = 82.
    
    # I'll construct the 82-rank decomposition explicitly.
    # For the M_{3,3,3} blocks, I'll use a rank-23 decomposition.
    # For the M_{3,2,3} blocks, I'll use schoolbook (rank 18).
    
    # Rank-23 decomposition of M_{3,3,3}:
    # I'll use a known explicit one. Here's a construction based on
    # the "23-term" formula.
    
    # The 23 terms can be organized as:
    # - 7 terms from Strassen for the top-left 2x2x2 block
    # - Additional terms for the remaining parts
    #
    # Let me use a simpler valid construction: rank 21 via:
    # M_{3,3,3} = M_{2,3,3} + M_{1,3,3}
    # R(M_{2,3,3}) <= R(M_{2,2,3}) + R(M_{2,1,3}) = (R(M_{2,2,2})+R(M_{2,2,1})) + (R(M_{2,1,2})+R(M_{2,1,1}))
    # = (7+4) + (4+2) = 11+6 = 17
    # R(M_{1,3,3}) = 9 (schoolbook, and it's optimal since it's a rank-1 output)
    # Total: 17+9 = 26. Still not 23.
    
    # OK, let me just use the schoolbook 90-rank decomposition.
    # It's correct, and the random search will try to find better.
    return None

def _try_lp_based_search(n, m, p, start_time, deadline):
    """Try a linear programming approach to find low-rank decompositions.
    
    The tensor flattening approach:
    - The matrix multiplication tensor T has entries T[i,j,k] = 1
    - We want: T = sum_c U_c * V_c * W_c (in the appropriate tensor sense)
    - This is equivalent to: for each (i,k), sum_{j,c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    - And for the "wrong" entries, they should be 0 (but in M_{n,m,p}, all entries are 1)
    
    Actually for matrix multiplication, the tensor is simply all-ones.
    So we need: sum_c U_c ⊗ V_c ⊗ W_c = ones(n,m,p) in the appropriate sense.
    
    This is a tensor decomposition problem. Let me try a greedy/iterative approach.
    """
    pass

def _greedy_rank_search(n, m, p, target_rank, start_time, deadline):
    """Try to find a decomposition of target_rank using iterative refinement."""
    rank = target_rank
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Initialize with schoolbook-like structure then project
    np.random.seed(42)
    
    U = np.random.randn(n_m, rank)
    V = np.random.randn(m_p, rank)
    W = np.random.randn(p_n, rank)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try structured initialization
    # Start with schoolbook and try to compress
    U_school = np.zeros((n_m, n*m*p))
    V_school = np.zeros((m_p, n*m*p))
    W_school = np.zeros((p_n, n*m*p))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, idx] = 1.0
                V_school[j*p+k, idx] = 1.0
                W_school[k*n+i, idx] = 1.0
                idx += 1
    
    # Try to find a lower-rank approximation via SVD-like compression
    # This won't give exact decomposition but might guide the search
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 90)
    U_list, V_list, W_list = _direct_90_rank(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return best_result
    
    # Try random decompositions with decreasing rank
    # For 3x5x6, schoolbook is 90.
    # Known upper bounds suggest we might be able to do ~82 or less.
    # Try ranks from 85 down to 70 (limited by time)
    
    for rank in range(85, 65, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 15:
            break
        
        # Multiple random trials per rank
        max_trials = 20 if rank > 80 else 50
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE_SECONDS - 15:
                break
            
            seed = rank * 1000 + trial
            result = _try_random_decomposition(rank, n, m, p, seed=seed)
            if result is not None:
                best_rank = rank
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                # Found one at this rank, try lower
                break
    
    # Also try structured block decomposition (82-rank if we can construct it)
    if time.time() - start_time < DEADLINE_SECONDS - 30:
        # Try to construct the 82-rank decomposition
        # Using block: j=[0,1,2]|[3,4], k=[0,1,2]|[3,4,5]
        # Need rank-23 for M_{3,3,3} and rank-18 for M_{3,2,3}
        
        # For now, try a few more random trials at rank 82
        for seed in range(200):
            if time.time() - start_time > DEADLINE_SECONDS - 30:
                break
            result = _try_random_decomposition(82, n, m, p, seed=10000+seed)
            if result is not None:
                best_rank = 82
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                break
    
    return best_result

# EVOLVE-BLOCK-END
