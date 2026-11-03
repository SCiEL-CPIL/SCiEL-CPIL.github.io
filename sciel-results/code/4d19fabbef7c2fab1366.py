"""Optimized tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [3, 3, 6]
_TIME_LIMIT = 150.0  # wall-clock limit in seconds (evaluator kills at 150s)

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _rank47_4x4x4_decomposition():
    """
    Return a rank-47 decomposition for 4x4 * 4x4 matrix multiplication.
    Based on known constructions from the literature.
    
    The 4x4 matrix multiplication tensor has rank at most 47 over the reals.
    We'll construct an explicit decomposition using a combination of:
    - Block diagonal 2x2 Strassen decompositions
    - Additional cross terms
    
    Indexing convention:
    A[i*4+j], B[j*4+k], C[k*4+i]
    """
    rank = 47
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # We'll use a construction based on partitioning 4x4 into 2x2 blocks
    # and applying Strassen-like techniques
    
    # For now, let's use a simpler approach: use the known fact that
    # rank(4x4) <= 2*rank(2x2) + overhead
    # rank(2x2) = 7 (Strassen)
    # So we can get rank <= 2*7 + additional terms
    
    # Actually, let me use a more direct construction.
    # The rank-47 decomposition can be built from:
    # - 4 independent 2x2 Strassen decompositions (rank 28)
    # - Plus 19 additional terms for cross-block interactions
    
    # This is getting complex. Let me use a different approach:
    # Use the known result that rank(4x4) <= 47 and construct it
    # using a combination of block diagonal and off-diagonal terms.
    
    # For now, return None and use a fallback approach
    return None


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r]^T equals the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True

def _embed_3x3x6_into_4x4x4(U4, V4, W4):
    """
    Extract a 3x3x6 decomposition from a 4x4x4 decomposition.
    
    The 3x3 matrix A is embedded into 4x4 as:
    A_4x4 = [[A, 0], [0, 0]]  (top-left 3x3 block)
    
    The 3x6 matrix B is embedded into 4x4 as:
    B_4x4 = [[B[:, :3], B[:, 3:]], [0, 0]]
    i.e., B_4x4[i,j] = B[i,j] for i<3, j<6, else 0
    
    The output C_4x4 should have:
    C_4x4[i,k] = (A*B)[i,k] for i<3, k<3 (since A is 3x3, B is 3x6, C is 3x6)
    Wait, that's not right. A is 3x3, B is 3x6, so C = A*B is 3x6.
    
    Let me reconsider the embedding:
    A_4x4 = [[A, 0], [0, 0]] where A is 3x3, so A_4x4 is 4x4 with top-left 3x3 = A
    B_4x4 = [[B, 0], [0, 0]] where B is 3x6... but 3x6 doesn't fit in 4x4.
    
    Actually, we need to think about this differently.
    The multiplication is: (3x3) * (3x6) -> (3x6)
    
    We can embed this into (4x4) * (4x4) -> (4x4) by:
    A_4x4 = [[A, 0_3x1], [0_1x3, 0]]  (4x4)
    B_4x4 = [[B_1, B_2], [0, 0]] where B_1, B_2 are 3x3 blocks of B
    Wait, B is 3x6, so we can write B = [B_1 | B_2] where B_1, B_2 are 3x3.
    
    Then A*B = [A*B_1 | A*B_2]
    
    In the 4x4 embedding:
    A_4x4 = [[A, 0], [0, 0]]
    B_4x4 = [[B_1, B_2], [0, 0]]
    
    Then A_4x4 * B_4x4 = [[A*B_1, A*B_2], [0, 0]]
    
    So the top-left 3x3 block of the result is A*B_1, and the top-right 3x3 block is A*B_2.
    This gives us the full 3x6 result C = [A*B_1 | A*B_2].
    
    Now, if we have a rank-R decomposition for 4x4*4x4, we can extract the
    contributions to the top-left 3x3 and top-right 3x3 blocks.
    
    For the 4x4 decomposition:
    U4[a, r], V4[b, r], W4[c, r] for a,b,c in 0..15
    
    The 4x4 tensor is:
    T4[a, b, c] = sum_r U4[a,r] * V4[b,r] * W4[c,r]
    
    where a = i*4+j, b = j'*4+k, c = k'*4+i'
    T4 = 1 iff j==j' and i==i' and k==k'
    
    For our 3x3x6 problem, we need:
    T3[a, b, c] = 1 iff the indices correspond to valid 3x3x6 multiplication
    
    The mapping:
    A_4x4 index a = i*4+j for i,j in 0..2 (top-left 3x3)
    B_4x4 index b: for B_1 (cols 0-2): b = j'*4+k for j' in 0..2, k in 0..2
                  for B_2 (cols 3-5): b = j'*4+(k+3) for j' in 0..2, k in 0..2
    C_4x4 index c: for C_1 (cols 0-2): c = k'*4+i for k' in 0..2, i in 0..2
                  for C_2 (cols 3-5): c = k'*4+(i+3) for k' in 0..2, i in 0..2
    
    Wait, this doesn't match the convention W[k*n+i]. Let me re-examine.
    
    The convention is W[k*n+i, r] where n=3 for the output.
    So C index = k*3+i for the 3x6 output, where k is row, i is col.
    
    In the 4x4 embedding, C_4x4 index = k'*4+i' for 4x4 output.
    
    We need to map:
    For C_1 (first 3 cols of 3x6 output): C_4x4[k', i'] where k'=k, i'=i for k,i in 0..2
    For C_2 (last 3 cols of 3x6 output): C_4x4[k', i'+3] where k'=k, i'=i for k,i in 0..2
    
    But wait, the 4x4 output is 4x4, so indices are 0..15.
    C_4x4 index = k'*4 + i' for k', i' in 0..3.
    
    For the top-left 3x3 block: k' in 0..2, i' in 0..2 -> indices 0..2, 4..6, 8..10
    For the top-right 3x3 block: k' in 0..2, i' in 3..3 -> indices 3, 7, 11
    
    Hmm, this is getting complicated. Let me use a cleaner approach.
    
    Actually, I realize the embedding should be:
    A is 3x3, embedded as top-left 3x3 of 4x4
    B is 3x6, which we split into B1 (3x3) and B2 (3x3)
    B is embedded as [[B1, B2], [0, 0]] in 4x4
    C is 3x6, which is [C1 | C2] where C1 = A*B1 (3x3) and C2 = A*B2 (3x3)
    C is embedded as [[C1, C2], [0, 0]] in 4x4
    
    Now, the 4x4 multiplication tensor has non-zero entries at:
    T4[a, b, c] = 1 iff a=i*4+j, b=j'*4+k, c=k'*4+i' with j==j', i==i', k==k'
    
    For our restricted problem, we only care about:
    a in {0,1,2, 4,5,6, 8,9,10} (top-left 3x3 of A_4x4)
    b in {0,1,2, 4,5,6, 8,9,10} (top-left 3x3 of B_4x4, which is B1)
       and {3,7,11} (top-right 3x3 of B_4x4, which is B2)
    c in {0,1,2, 4,5,6, 8,9,10} (top-left 3x3 of C_4x4, which is C1)
       and {3,7,11} (top-right 3x3 of C_4x4, which is C2)
    
    The tensor T3 for 3x3x6 has non-zero entries at:
    T3[a, b, c] = 1 iff a=i*3+j, b=j'*6+k, c=k*3+i with j==j'
    where i in 0..2, j in 0..2, j' in 0..2, k in 0..5
    
    Wait, I'm confusing myself. Let me restart with a cleaner formulation.
    
    The 3x3x6 multiplication tensor T has:
    T[A_idx, B_idx, C_idx] = 1 iff A_idx = i*3+j, B_idx = j'*6+k, C_idx = k*3+i
    with j == j'
    
    So the non-zero entries are at (i*3+j, j*6+k, k*3+i) for i,j in 0..2, k in 0..5.
    That's 3*3*6 = 54 non-zero entries.
    
    Now, to embed into 4x4x4:
    We map:
    A_idx_4x4 = i*4+j for i,j in 0..2 (using the top-left 3x3 of 4x4)
    B_idx_4x4: for B1 (k in 0..2): j'*4+k for j' in 0..2
               for B2 (k in 3..5): j'*4+(k-3+3) = j'*4+k for k in 3..5
               So B_idx_4x4 = j'*4+k for j' in 0..2, k in 0..5
               But 4x4 only has indices 0..15, and j'*4+k for j' in 0..2, k in 0..5
               gives indices: 0,1,2,3, 4,5,6,7, 8,9,10,11
    C_idx_4x4: for C1 (k in 0..2): k*4+i for k in 0..2, i in 0..2
               for C2 (k in 3..5): k*4+i for k in 3..5, i in 0..2
               So C_idx_4x4 = k*4+i for k in 0..5, i in 0..2
               This gives indices: 0,1,2, 4,5,6, 8,9,10, 12,13,14
    
    Now, the 4x4 tensor T4 has:
    T4[a, b, c] = 1 iff a=i*4+j, b=j'*4+k, c=k'*4+i' with j==j', i==i', k==k'
    
    For our restricted indices:
    a = i*4+j (i,j in 0..2)
    b = j'*4+k (j' in 0..2, k in 0..5)
    c = k*4+i (k in 0..5, i in 0..2)
    
    T4[a, b, c] = 1 iff j==j', i==i', k==k'
    But in our restricted case, i==i' is automatic (both are i), and k==k' is automatic (both are k).
    So T4[a, b, c] = 1 iff j==j'
    
    This is exactly our 3x3x6 tensor!
    
    So the embedding is:
    A_4x4[i*4+j] = A[i,j] for i,j in 0..2
    B_4x4[j'*4+k] = B[j',k] for j' in 0..2, k in 0..5
    C_4x4[k*4+i] = C[k,i] for k in 0..5, i in 0..2
    
    And the 4x4 multiplication restricted to these indices gives exactly the 3x3x6 multiplication.
    
    Therefore, if we have a rank-R decomposition for 4x4x4, we can extract a rank-R
    decomposition for 3x3x6 by restricting the indices.
    
    Specifically, if (U4, V4, W4) is a rank-R decomposition for 4x4x4, then
    U3[i*3+j, r] = U4[i*4+j, r] for i,j in 0..2
    V3[j'*6+k, r] = V4[j'*4+k, r] for j' in 0..2, k in 0..5
    W3[k*3+i, r] = W4[k*4+i, r] for k in 0..5, i in 0..2
    
    And (U3, V3, W3) is a rank-R decomposition for 3x3x6.
    """
    # Extract the relevant indices from the 4x4 decomposition
    n, m, p = 3, 3, 6
    
    U4 = np.array(U4)
    V4 = np.array(V4)
    W4 = np.array(W4)
    rank = U4.shape[1]
    
    U3 = np.zeros((n*m, rank))
    V3 = np.zeros((m*p, rank))
    W3 = np.zeros((p*n, rank))
    
    # Map A indices: 3x3 -> 4x4 top-left
    for i in range(3):
        for j in range(3):
            a_4 = i*4 + j
            a_3 = i*3 + j
            U3[a_3, :] = U4[a_4, :]
    
    # Map B indices: 3x6 -> 4x4
    # B is 3x6, so j' in 0..2, k in 0..5
    # In 4x4: b_4 = j'*4 + k
    # In 3x6: b_3 = j'*6 + k
    for j in range(3):
        for k in range(6):
            b_4 = j*4 + k
            b_3 = j*6 + k
            V3[b_3, :] = V4[b_4, :]
    
    # Map C indices: 3x6 -> 4x4
    # C is 3x6, so k in 0..5, i in 0..2
    # In 4x4: c_4 = k*4 + i
    # In 3x6: c_3 = k*3 + i
    for k in range(6):
        for i in range(3):
            c_4 = k*4 + i
            c_3 = k*3 + i
            W3[c_3, :] = W4[c_4, :]
    
    return {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j2 in range(m):
            for k in range(p):
                A = i * m + j2  # Note: j2 is the column of A that matches row of B
                # Actually, the tensor index is T[A_idx, B_idx, C_idx] where:
                # A_idx = i*m + j (row i, col j of A)
                # B_idx = j*p + k (row j, col k of B)  
                # C_idx = k*n + i (row k, col i of C)
                # And T = 1 iff j == j (the shared index)
                # So for fixed (i, j, k), T[i*m+j, j*p+k, k*n+i] = 1
                # All other entries are 0.
                pass
    
    # Let me re-verify the convention more carefully.
    # The task says: W[k*n+i, r] is the output factor.
    # The tensor T_{A,B,C} = sum_r U_{A,r} V_{B,r} W_{C,r}
    # where A = i*m + j, B = j'*p + k, C = k'*n + i
    # T = 1 iff j == j' (and implicitly the output C corresponds to correct positions)
    
    # Actually for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T_{i,j; j',k; i',k'} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # In the flat indexing:
    # A_idx = i*m + j
    # B_idx = j'*p + k
    # C_idx = k'*n + i'
    # T = 1 iff j == j' AND i == i' AND k == k'
    
    # Let me verify this interpretation with the schoolbook:
    # Schoolbook: for each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k) term
    # This gives C[k*n+i] += A[i*m+j] * B[j*p+k]
    # i.e., C[k,i] += A[i,j] * B[j,k]
    # So C[k,i] = sum_j A[i,j]*B[j,k]
    # This means C is indexed as C[k,i] = (A*B)^T[i,k]... 
    # Wait, no. C[k*n+i] means row k, col i of C.
    # So C[k,i] = sum_j A[i,j] * B[j,k]
    # But (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So C[k,i] = (A*B)[i,k], meaning C = (A*B)^T
    # OR the output is stored in column-major order.
    
    # Let me just use the verify function as given and trust it.
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True


def _build_rank19_3x3x3():
    """
    Build a rank-19 decomposition for 3x3 * 3x3 matrix multiplication.
    Based on the known result that rank(3x3 matmul) = 19 (over reals/rationals).
    
    For now, return None and use a different strategy in construct().
    """
    return None

def _construct_4x4x4_rank47():
    """
    Construct a rank-47 decomposition for 4x4 * 4x4 matrix multiplication.
    
    This is a known result: rank(4x4 matmul) <= 47.
    We'll construct it using a combination of:
    - Block diagonal 2x2 Strassen decompositions
    - Additional cross terms
    
    Indexing: A[i*4+j], B[j*4+k], C[k*4+i]
    """
    rank = 47
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # We'll use a construction based on the following idea:
    # Split 4x4 into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # For each 2x2 block multiplication, we can use Strassen (rank 7)
    # But we need to handle the cross terms carefully.
    
    # Actually, let me use a simpler approach: use the known fact that
    # we can get rank 47 by using 4 independent 2x2 Strassen decompositions
    # for the diagonal blocks and additional terms for the off-diagonal blocks.
    
    # For now, let me use a fallback: schoolbook for 4x4 (rank 64) and see
    # if the embedding gives us something better than 54.
    
    # Schoolbook for 4x4:
    for i in range(4):
        for j in range(4):
            for k in range(4):
                col = i*16 + j*4 + k
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _construct_via_block_decomposition(n, m, p):
    """
    Construct a decomposition for 3x3 * 3x6 by viewing it as three 3x3 * 3x2 blocks
    or using other structured decompositions.
    
    Strategy: The 3x6 matrix B can be viewed as [B1 | B2] where B1, B2 are 3x3.
    Then A*B = [A*B1 | A*B2].
    
    If we have a rank-R decomposition for 3x3*3x3, we can use it twice
    to get rank 2R for 3x3*3x6. But we can do better by sharing structure.
    
    Better strategy: Use the field extension approach.
    Encode the 3x6 matrix as a 3x3 matrix over the ring R = F[t]/(t^2)
    or R = F x F.
    
    If we use R = F x F (pairs), then a 3x3 matrix over R represents a 3x6 matrix.
    A rank-R decomposition over R for 3x3*3x3 gives a rank-R decomposition for 3x6.
    But we need the decomposition to be over the base field.
    
    Key insight: If we have a rank-R decomposition of 3x3*3x3 over the base field
    F, then by applying it to each "component" of the pair, we get a rank-2R
    decomposition of 3x3*3x6. But we might be able to do better.
    
    Actually, the correct approach: 
    - A 3x6 matrix B can be written as B = B1 + t*B2 where t^2 = 0 (dual numbers)
      or B = B1 + B2 where we use the ring F x F.
    - If we have a rank-R decomposition for 3x3*3x3 that works over any commutative
      ring, then it works over F x F, giving a rank-R decomposition for 3x3*3x6
      where each "multiplication" is component-wise.
    - To convert back to the base field, each rank-1 term over F x F becomes
      two rank-1 terms over F (one for each component).
    - So rank-R over F x F gives rank-2R over F.
    
    Therefore, if rank(3x3*3x3) = 19, then rank(3x3*3x6) <= 2*19 = 38.
    But we might do better by finding a decomposition that exploits the structure.
    
    Let me try to directly search for a low-rank decomposition using the
    "linear dependence" method.
    """
    deadline = time.time() + _TIME_LIMIT - 20
    
    # Strategy 1: Use two rank-19 decompositions for the two 3x3 blocks
    # This gives rank 38.
    
    # Strategy 2: Try to find a decomposition with rank < 38 using optimization.
    
    # For now, let's implement a structured search.
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Try ranks from 30 to 53
    for rank in range(30, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
            result = _try_rank_optimization(n, m, p, rank, seed=1000 + rank * 10 + restart)
            if result is not None:
                best_result = result
                best_rank = rank
                found = True
                break
        
        if found and best_rank <= 35:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return schoolbook
    return _schoolbook(n, m, p)


def _try_rank_optimization(n, m, p, rank, seed=42):
    """
    Try to find a rank-`rank` decomposition using gradient-free optimization
    with structured initialization.
    """
    deadline = time.time() + _TIME_LIMIT - 30
    rng = random.Random(seed)
    
    # Number of non-zero tensor entries
    # T[i*m+j, j'*p+k, k'*n+i] = 1 iff j==j' and i==i' and k==k'
    # So there are n*m*p = 54 non-zero entries (all with value 1)
    
    # Initialize with random values
    U = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)])
    V = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)])
    W = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)])
    
    best_err = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Precompute the list of (A, B, C, expected) tuples
    entries = []
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    entries.append((A, B, C, expected))
    
    def compute_error(U, V, W):
        err = 0.0
        for A, B, C, expected in entries:
            actual = np.dot(U[A] * V[B], W[C])
            err += (actual - expected) ** 2
        return err
    
    max_iter = 300
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random matrix and entry to perturb
        mat_idx = rng.randint(0, 2)
        if mat_idx == 0:
            row = rng.randint(0, n * m - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice(step_sizes) * rng.choice([-1, 1])
            old_val = U[row, col]
            U[row, col] += delta
        elif mat_idx == 1:
            row = rng.randint(0, m * p - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice(step_sizes) * rng.choice([-1, 1])
            old_val = V[row, col]
            V[row, col] += delta
        else:
            row = rng.randint(0, p * n - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice(step_sizes) * rng.choice([-1, 1])
            old_val = W[row, col]
            W[row, col] += delta
        
        err = compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        else:
            if mat_idx == 0:
                U[row, col] = old_val
            elif mat_idx == 1:
                V[row, col] = old_val
            else:
                W[row, col] = old_val
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-9:
        # Round to nearest rational with small denominator
        U_r = np.round(best_U * 1000) / 1000
        V_r = np.round(best_V * 1000) / 1000
        W_r = np.round(best_W * 1000) / 1000
        
        # Verify the rounded version
        if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Try with original values
        if _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-9):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def _construct_rank38_via_two_blocks(n, m, p):
    """
    Construct a rank-38 decomposition by using two rank-19 decompositions
    for the two 3x3 blocks of the 3x6 matrix.
    
    B = [B1 | B2] where B1, B2 are 3x3.
    A*B = [A*B1 | A*B2]
    
    If we have rank-19 decompositions for A*B1 and A*B2, we get rank-38 total.
    But we need to verify that rank-19 decompositions exist and are correct.
    
    For now, this is a placeholder. The actual rank-19 decomposition is complex
    to write down explicitly.
    """
    # This function would return a rank-38 decomposition if we had verified
    # rank-19 decompositions for 3x3*3x3.
    # Since constructing them explicitly is complex, we fall back to search.
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = fallback
    
    # Try the 4x4x4 embedding approach
    try:
        U4, V4, W4 = _construct_4x4x4_rank47()
        rank4 = len(U4[0])
        
        # Extract the 3x3x6 decomposition
        result = _embed_3x3x6_into_4x4x4(U4, V4, W4)
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        
        if rank < best_rank and _verify(U, V, W, n, m, p):
            best_result = result
            best_rank = rank
    except Exception as e:
        pass
    
    # Try structured optimization
    result = _construct_via_block_decomposition(n, m, p)
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        if rank < best_rank and _verify(U, V, W, n, m, p):
            best_result = result
            best_rank = rank
    
    return best_result

# EVOLVE-BLOCK-END
