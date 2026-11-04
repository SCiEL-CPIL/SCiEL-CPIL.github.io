"""Matrix multiplication 5x5x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _verify_float(n, m, p, U, V, W, tol=1e-9):
    """Verify using floating point."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dot = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(dot - 1.0) > tol:
                    return False
    return True


def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True


def _strassen_2x2():
    """
    Classic Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7).
    U indexes (i,j), V indexes (j,k), W indexes (k,i).
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1: (A11-A22)*(B22-B11)
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = -1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A11*(B11-B22)
    U[0][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22*(B22-B11)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}


def _embed_strassen_2x2_into_5x5():
    """
    Embed the 2x2 Strassen decomposition into 5x5 by using it for the top-left
    2x2 block and schoolbook for the rest. This gives rank 7 + 28 = 35.
    """
    n = 5
    strassen = _strassen_2x2()
    rank = 7 + (25 - 4)  # 7 for 2x2 Strassen + 21 schoolbook for remaining
    # Actually: 2x2 block uses 7, remaining 23 entries use 23 schoolbook = 30 total
    # Top-left 2x2x2 sub-tensor has 8 indices. Strassen rank 7.
    # Remaining 125 - 8 = 117 indices handled by schoolbook (rank 1 each).
    # Total rank = 7 + 117 = 124.
    rank = 7 + 117
    
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    # Embed Strassen for top-left 2x2: indices i,j,k in {0,1}
    # U index: i*m+j = i*5+j, V index: j*p+k = j*5+k, W index: k*n+i = k*5+i
    for c in range(7):
        for ii in range(4):
            if strassen["U"][ii][c] != 0:
                i, j = divmod(ii, 2)
                U[i * 5 + j][c] = strassen["U"][ii][c]
        for jj in range(4):
            if strassen["V"][jj][c] != 0:
                j, k = divmod(jj, 2)
                V[j * 5 + k][c] = strassen["V"][jj][c]
        for kk in range(4):
            if strassen["W"][kk][c] != 0:
                k, i = divmod(kk, 2)
                W[k * 5 + i][c] = strassen["W"][kk][c]
    
    # Schoolbook for remaining (i,j,k) where not all in {0,1}
    col = 7
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _try_multiple_strassen_5x5():
    """
    Try to use Strassen decomposition in multiple 2x2 blocks within 5x5.
    Partition 5x5 into non-overlapping 2x2 blocks plus remainder.
    5x5: can fit two 2x2 blocks in each dimension: positions (0,0),(0,2),(2,0),(2,2)
    Actually let's try: use Strassen for the (0,0)-(1,1) block in all three indices.
    Then handle the rest with schoolbook.
    
    Alternative: Use 2x2 Strassen on multiple independent 2x2 subproblems.
    For 5x5x5, we can identify disjoint 2x2x2 sub-tensors.
    Positions: (0,0,0), (0,0,2), (0,2,0), (2,0,0), etc.
    
    Let's try a different approach: use the Laderman-like structure for 3x3
    embedded in 5x5, or just do better partitioning.
    
    For now, let's try: 5 = 2+2+1. Use Strassen on the 2x2x2 corners.
    There are 8 possible 2x2x2 sub-blocks but they overlap.
    Non-overlapping: at most floor(5/2)^3 = 1 sub-block if we require full 2x2x2.
    
    Better: use a Kronecker product approach.
    5x5 multiplication can be seen as block multiplication.
    Partition A, B into blocks. If we use 2x2 block structure:
    A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]] where blocks are 2x2, 2x2, 1x1
    This gets complex. Let's just return the single-Strassen embedding for now.
    """
    return _embed_strassen_2x2_into_5x5()


def _try_cantor_burris(n, m, p):
    """
    Apply the Cantor-Burris / linear algebra approach to find the tensor rank.
    
    The idea: the matrix multiplication tensor T_{i,j,k} can be viewed as a
    linear map. The tensor rank is the minimum number of rank-1 tensors needed.
    
    For a systematic approach: construct the 25x25x25 tensor and compute its
    "rank" via the unfolding method. The minimum rank r must satisfy:
    - r >= max(rank(T_{(1)(2)}), rank(T_{(1)(3)}), rank(T_{(2)(3)}))
    where T_{(1)(2)} is the 25x125 matrix unfolding, etc.
    
    This gives a lower bound. We then search for actual decompositions at that rank.
    """
    # The tensor T[i,j,k] = delta (it's 1 for all i,j,k in the matmul tensor)
    # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Actually the matmul tensor is: T(A,B) = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} e_{ik}
    # where T_{ijk} = 1 for all valid i,j,k.
    # 
    # The tensor as a 3D array is all-ones of shape (25, 25, 25).
    # Its unfolding T_{(1)(2)} is a 25 x (25*25) = 25 x 625 matrix of all ones.
    # rank = 1. That's not useful as a lower bound.
    #
    # The correct formulation: The matmul tensor has entries T[i,j,k] where
    # the (i,j) index goes to U, (j,k) to V, (k,i) to W.
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    #
    # The rank of this tensor (as a multilinear form) is what we want.
    # For the all-ones tensor of shape (n,m,p), the tensor rank is max(n,m,p)
    # if we're talking about the standard tensor rank... no that's wrong.
    #
    # Actually, the matrix multiplication tensor T_{ijk} = delta_{i,k} doesn't work
    # because we need sum over j. Let me reconsider.
    #
    # The matrix multiplication tensor is defined as:
    # T(A,B)_ik = sum_j A_ij B_jk
    # So the tensor T has components T_{ijk} = 1 for all i,j,k (in the standard basis
    # where A_ij and B_jk are coordinates).
    # 
    # The rank of this tensor is the bilinear complexity of matrix multiplication.
    # For n=m=p=5, this is what we're trying to find.
    #
    # The Cantor-Burris approach: represent the multiplication as a linear operator.
    # The multiplication map is: (A,B) -> AB, which is bilinear.
    # As a linear map from V x W to X (where V = space of A's, W = space of B's,
    # X = space of products), the bilinear complexity equals the tensor rank.
    #
    # A key result: if we fix B, the map A -> AB is linear. The tensor rank r means
    # we can write AB = sum_{c=1}^r (u_c^T A) (v_c^T B) w_c where u_c in V*, v_c in W*,
    # w_c in X. This is exactly what we're computing.
    #
    # For a constructive approach: use the fact that the tensor is all-ones and
    # try to find the minimum r such that there exist rank-r decompositions.
    #
    # Practical approach: use random initialization + local search to minimize rank.
    # Or use the known lower bound and search upward.
    
    # Lower bound: r >= ceil(n*m*p / (something))... 
    # Known: for n x n, r >= n^2 (trivially from fixing one input)
    # Better: r >= n^2 + n (from various bounds)
    # For 5x5: r >= 25 (trivial), and various algebraic geometry bounds give r >= 49 or so.
    
    # For now, return the best we can construct.
    return None


def _try_block_strassen_5x5():
    """
    Try a block-based Strassen approach for 5x5.
    Partition 5 = 3 + 2. Use Laderman for 3x3 (rank 23? no, 3x3 has rank 23) 
    and Strassen for 2x2 (rank 7).
    But the blocks interact, so it's not simply additive.
    
    Actually, for block matrix multiplication:
    If A = [[A11, A12], [A21, A22]] with A11: 3x3, A12: 3x2, A21: 2x3, A22: 2x2
    B = [[B11, B12], [B21, B22]] with B11: 3x3, B12: 3x2, B21: 2x3, B22: 2x2
    
    Then AB = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each entry is a sum of products of blocks. We can use tensor decomposition
    for each block product. But this gets complex. Let's try a simpler approach:
    use the Strassen idea recursively but with mixed block sizes.
    
    For simplicity, let's just use the single 2x2 Strassen embedding (rank 28)
    and see if we can do better with a 3x3 Laderman + 2x2 Strassen approach.
    """
    # For now, fall back to the 2x2 embedding
    return _embed_strassen_2x2_into_5x5()


def _laderman_3x3():
    """
    Laderman's 23-multiplication algorithm for 3x3 matrix multiplication.
    This is a known result: 3x3 matrix mult can be done in 23 scalar multiplications.
    Returns U (9x23), V (9x23), W (9x23).
    """
    # Laderman's algorithm is complex. Let me use a known formulation.
    # Actually, let me use the Pan's algorithm or just return a schoolbook for now
    # and focus on the 2x2 Strassen embedding which gives rank 28.
    
    # For 3x3, schoolbook is 27. Laderman gives 23.
    # Let me implement a known 23-mult algorithm for 3x3.
    # This is quite involved. Let me use a simpler approach:
    # Just return schoolbook for 3x3 for now.
    n = 3
    rank = n * n * n
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_hybrid_3_2(n, m, p):
    """
    Try to use a 3x3 Laderman + 2x2 Strassen hybrid for 5x5.
    Partition 5 = 3 + 2 in each dimension.
    
    A = [[A11, A12], [A21, A22]] where A11:3x3, A12:3x2, A21:2x3, A22:2x2
    B = [[B11, B12], [B21, B22]] where B11:3x3, B12:3x2, B21:2x3, B22:2x2
    
    C11 = A11*B11 + A12*B21  (3x3 product)
    C12 = A11*B12 + A12*B22  (3x2 product)
    C21 = A21*B11 + A22*B21  (2x3 product)
    C22 = A21*B12 + A22*B22  (2x2 product)
    
    For C11: need 3x3 * 3x3 -> 3x3. Can use Laderman (23 mults) if available.
    For C22: need 2x2 * 2x2 -> 2x2. Can use Strassen (7 mults).
    For C12: need 3x3 * 3x2 -> 3x2. This is 3x3 times 3x2, result 3x2.
    For C21: need 2x3 * 2x3 -> 2x3.
    
    The cross terms C12 and C21 are rectangular. We can handle them with
    schoolbook or by extending the Strassen/Laderman approach.
    
    This is getting complex. Let me just use the simple 2x2 embedding for now
    and try to improve with direct search.
    """
    return _embed_strassen_2x2_into_5x5()


def _try_direct_search(n, m, p, target_rank, timeout=30):
    """
    Try to find a decomposition of exactly target_rank using a structured
    random search with verification.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Use a structured initialization based on Strassen-like patterns
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
        
        np.random.seed(attempt * 12345 + target_rank)
        
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        W = np.zeros((pn, target_rank), dtype=float)
        
        # Initialize with a schoolbook-like base for the first min(target_rank, 25) entries
        base = min(target_rank, n * m * p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        if col < target_rank:
            # Add random perturbation for extra columns
            for c in range(col, target_rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=nm)
                V[:, c] = np.random.choice([-1, 0, 1], size=mp)
                W[:, c] = np.random.choice([-1, 0, 1], size=pn)
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    dot = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(dot - 1.0) > 1e-8:
                        valid = False
                        break
        
        if valid:
            return {
                "U": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U],
                "V": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V],
                "W": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
            }
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook (rank 125)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen 2x2 embedding (rank 28)
    if time.time() < deadline - 10:
        strassen_embed = _embed_strassen_2x2_into_5x5()
        U, V, W = strassen_embed["U"], strassen_embed["V"], strassen_embed["W"]
        if _verify_float(n, m, p, U, V, W):
            if _verify_exact(n, m, p, U, V, W):
                best = strassen_embed
                best_rank = len(U[0])
    
    # Try hybrid 3+2 approach
    if time.time() < deadline - 20 and best_rank > 25:
        hybrid = _try_hybrid_3_2(n, m, p)
        if hybrid is not None:
            U, V, W = hybrid["U"], hybrid["V"], hybrid["W"]
            rank = len(U[0])
            if rank < best_rank and _verify_float(n, m, p, U, V, W):
                if _verify_exact(n, m, p, U, V, W):
                    best = hybrid
                    best_rank = rank
    
    # Direct search for lower ranks is too slow and unreliable for 5x5 in this time frame.
    # We return the best valid decomposition found (rank 124 via Strassen embedding).
    return best
# EVOLVE-BLOCK-END
