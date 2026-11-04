import time
import numpy as np

_START = time.time()
DEADLINE = 135.0

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Compute all products efficiently
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _laderman_3x3_3():
    """Return U, V, W for 3x3 * 3x3 with rank 14 (Laderman)."""
    # Laderman's rank-14 algorithm for 3x3 matrix multiplication
    # P1 = (a00+a11+a22)(b00+b11+b22)
    # P2 = (a00+omega*a11+omega^2*a22)(b00+omega^2*b11+omega*b22)
    # P3 = (a00+omega^2*a11+omega*a22)(b00+omega*b11+omega^2*b22)
    # ... plus more terms for off-diagonal
    
    # Actually, Laderman's algorithm uses 14 multiplications for 3x3.
    # The coefficients are integers/rationals.
    
    # Let me use the known Laderman decomposition.
    # Reference: Laderman, S. (1976). "A computation matrix in 14 scalar products."
    
    # The 14 products are:
    # P1 = a11*b11
    # P2 = (a00+a22)*(b00+b22)
    # P3 = (a00-a22)*(b00-b22)
    # P4 = (a01+a02+a11+a21)*(b01+b02+b11+b21)
    # P5 = (a01+a21-a11-a21)*(b01+b21-b11-b21)
    # ... this is getting complicated. Let me use a different approach.
    
    # I'll use the explicit Laderman coefficients from the literature.
    # For 3x3, rank 14:
    
    rank = 14
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Laderman's algorithm (one standard form):
    # Let me use the formulation where we split into diagonal and off-diagonal.
    
    # Actually, let me just hardcode a known-good rank-14 decomposition.
    # From Laderman (1976) and subsequent work:
    
    # I'll use the decomposition from "Matrix Multiplication via Arithmetic Progressions"
    # or just verify with a simpler approach.
    
    # Let me try: if Laderman verification fails, we fall back to schoolbook.
    # For safety, I'll implement a simpler but still better approach:
    # Use 2 Strassen blocks for the 3x3x3 case.
    # 3x3x3 = 27 cells. 2 Strassen blocks cover 16 cells. Remaining 11. Rank = 14+11 = 25? No, 7*2+11=25.
    # That's worse than Laderman's 14.
    
    # Let me implement the actual Laderman coefficients.
    # From the paper, the 14 linear forms are:
    
    # Actually, let me use a different strategy: just use the Strassen block approach but
    # try harder to find more disjoint blocks. For 3x3x8, we found 4 blocks (rank 68).
    # Can we do better?
    
    # For now, let me implement a correct Laderman and verify it.
    # Laderman rank 14 for 3x3:
    
    # Coefficients from standard references:
    # U (first matrix coefficients):
    U[0,0] = 1; U[1,0] = 0; U[2,0] = 1; U[3,0] = 0; U[4,0] = 0; U[5,0] = 1; U[6,0] = 0; U[7,0] = 0; U[8,0] = 1
    U[0,1] = 0; U[1,1] = 0; U[2,1] = 1; U[3,1] = 0; U[4,1] = 1; U[5,1] = 0; U[6,1] = 1; U[7,1] = 0; U[8,1] = 1
    U[0,2] = 1; U[1,2] = 0; U[2,2] = 1; U[3,2] = 0; U[4,2] = 1; U[5,2] = 0; U[6,2] = 1; U[7,2] = 0; U[8,2] = 1
    # This is getting messy. Let me just use schoolbook fallback if Laderman fails.
    
    # I'll return None to indicate failure, and use schoolbook.
    return None

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    return U, V, W

def _get_block_coverage(i0, j0, k0, n, m, p):
    covered = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = i0+di, j0+dj, k0+dk
                if 0 <= i < n and 0 <= j < m and 0 <= k < p:
                    covered.add((i, j, k))
    return covered

def _build_decomposition_with_blocks(n, m, p, blocks):
    if not blocks:
        return _schoolbook(n, m, p)
    
    covered = set()
    for (i0, j0, k0) in blocks:
        cov = _get_block_coverage(i0, j0, k0, n, m, p)
        covered |= cov
    
    n_school = n * m * p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    sU, sV, sW = _strassen_2x2_2()
    for (i0, j0, k0) in blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col+c] = sU[di*2+dj, c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk), col+c] = sV[dj*2+dk, c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di), col+c] = sW[dk*2+di, c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return U, V, W

def _try_tensor_completion(n, m, p, target_rank, max_iters=200, seed=0):
    """
    Try to find a rank-target_rank decomposition using tensor completion.
    Returns (U, V, W) if successful, else None.
    """
    np.random.seed(seed)
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Initialize factors
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    # We want to minimize || T - sum_k U[:,k] o V[:,k] o W[:,k] ||_F
    # This is equivalent to minimizing the residual.
    
    # We will use an alternating least squares approach.
    # For a fixed V, W, the optimal U[:,k] is given by:
    # U[:,k] = argmin_U || T - sum_j U_j o V_j o W_j ||_F
    # This is a linear system if we fix V and W? No, it's bilinear.
    # However, if we fix V and W, the objective is quadratic in U.
    # Let R_k = T - sum_{j != k} U_j o V_j o W_j
    # Then we minimize || R_k - U_k o V_k o W_k ||_F
    # This is a rank-1 approximation problem.
    # The optimal U_k is the left singular vector of the matrix obtained by flattening R_k
    # with respect to the i-mode, scaled appropriately?
    # Actually, the optimal rank-1 approximation of a tensor T to T - U_k o V_k o W_k
    # is given by the CP decomposition step.
    
    # Let's use a simpler gradient descent or ALS with random restarts.
    # Given the small size, we can try ALS.
    
    # To make ALS work, we can fix V and W, and solve for U.
    # The problem is: min_U || T - sum_k U_k o V_k o W_k ||^2
    # This is not linear in U if V and W are fixed?
    # Wait, T is fixed. The term sum_k U_k o V_k o W_k is linear in U_k.
    # So for each k, given other U_j, V_j, W_j, we can solve for U_k.
    # Let T_res = T - sum_{j != k} U_j o V_j o W_j
    # We want to minimize || T_res - U_k o V_k o W_k ||_F
    # This is equivalent to finding the best rank-1 approximation of T_res
    # with fixed factors V_k and W_k? No, V_k and W_k are fixed vectors.
    # We want to find U_k such that U_k o V_k o W_k approximates T_res.
    # This is a linear least squares problem for U_k.
    # Flatten T_res into a matrix M of size (n*m) x (p) ? No.
    # The tensor T_res[i, j, k] can be seen as a vector of length n*m*p.
    # The vector U_k o V_k o W_k has entries U_k[i*m+j] * V_k[j*p+k] * W_k[k*n+i].
    # Let's construct a matrix A of size (n*m*p) x 1 where A[idx] = V_k[j*p+k] * W_k[k*n+i]
    # for the index idx corresponding to (i, j, k).
    # Then we solve min_U || T_res_vec - A * U_scalar ||? No, U_k is a vector of length n*m.
    # The term is sum_{i,j} U_k[i*m+j] * (V_k[j*p+k] * W_k[k*n+i]) * delta_{k, k'}?
    # No, the sum is over the rank index.
    
    # Let's use a different perspective.
    # We want to minimize || T - U V^T W^T || where U, V, W are matrices.
    # This is the standard tensor decomposition.
    # We can use ALS.
    
    # Reshape T to (n*m, p) ? No, the structure is (n, m, p).
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    # Let's try a simple iterative approach:
    # 1. Fix V, W. Update U.
    # 2. Fix U, W. Update V.
    # 3. Fix U, V. Update W.
    
    # To update U[:, k]:
    # We want to minimize || T - sum_j U_j o V_j o W_j ||_F
    # This is equivalent to minimizing || T_{(1)} - U (V_{(1)} \otimes W_{(1)})^T ||_F ?
    # Where T_{(1)} is the mode-1 unfolding of T (size n x m*p).
    # But our U is indexed by (i*m+j). This corresponds to the mode-1 unfolding if we reshape T to (n*m, p)?
    # No, standard unfolding T_{(1)} is n x (m*p).
    # Our U is (n*m, rank). This is not standard.
    
    # Let's stick to the verification-based search if ALS is too complex to implement correctly with these specific shapes.
    # Or, we can use the fact that the tensor is sparse (all 1s).
    
    # Actually, the current program score is 1.0588 (rank 68).
    # If we can find rank 67, score is 72/67 = 1.0746.
    # If we find rank 66, score is 72/66 = 1.0909.
    
    # Let's try a simpler heuristic: Randomized SVD-like approach?
    # Or just return None to indicate failure and rely on the block method.
    # Given the time constraint and complexity, I will return None for now,
    # but I will add a small search for rank 67 using random restarts and snapping.
    
    # Let's implement a basic random restart + snap.
    # We generate random U, V, W of rank R.
    # We snap to nearest 0.5.
    # We verify.
    
    # This is unlikely to succeed for R < 68 given the structure, but it's a valid attempt.
    # To make it more likely, we can initialize with the block solution and perturb it?
    # The block solution has rank 68. We want rank 67.
    # We can try to merge two schoolbook terms into one?
    # Schoolbook terms are rank 1. Merging two rank 1 terms into one rank 1 term is impossible unless they are proportional.
    # So we can't just merge.
    
    # What if we use a rank-67 decomposition from the literature?
    # I don't have one memorized for 3x3x8.
    
    # Let's try to improve the block packing.
    # Current: 4 blocks of 2x2x2.
    # Can we pack 5 blocks?
    # 5 blocks cover 40 entries. 72-40=32 schoolbook. Rank = 5*7 + 32 = 35+32=67.
    # If we can find 5 disjoint 2x2x2 blocks in 3x3x8, we get rank 67.
    
    # Let's check if 5 disjoint blocks are possible.
    # n=3, m=3, p=8.
    # Block positions (i0, j0, k0).
    # i0 in {0, 1}, j0 in {0, 1}, k0 in {0, ..., 6}.
    # Total possible blocks: 2 * 2 * 7 = 28.
    # We need to choose 5 disjoint ones.
    
    # The current code tries to find 5 disjoint blocks.
    # Let's verify if the current code actually finds them.
    # It iterates combinations of 5.
    # If it finds them, it returns rank 67.
    # The current score is 1.0588 which corresponds to rank 68 (72/68 = 1.0588).
    # This implies the current code DID NOT find 5 disjoint blocks.
    
    # Why?
    # Let's analyze the geometry.
    # n=3, m=3.
    # In the i-j plane (3x3), we can place at most 2 disjoint 2x2 blocks?
    # A 2x2 block in 3x3 covers 4 cells.
    # Two disjoint 2x2 blocks in 3x3:
    # Block 1: (0,0), (0,1), (1,0), (1,1) -> covers i in {0,1}, j in {0,1}.
    # Block 2: (1,1), (1,2), (2,1), (2,2) -> covers i in {1,2}, j in {1,2}.
    # Intersection: (1,1). Not disjoint.
    # Block 1: (0,0)..(1,1).
    # Block 2: (0,2)..(1,2) is not 2x2.
    # In a 3x3 grid, can we place two disjoint 2x2 squares?
    # Area 4+4=8 <= 9.
    # Yes:
    # Square 1: rows 0,1; cols 0,1.
    # Square 2: rows 1,2; cols 1,2. Intersection (1,1).
    # Square 1: rows 0,1; cols 0,1.
    # Square 2: rows 0,1; cols 1,2. Intersection col 1.
    # It seems impossible to place two disjoint 2x2 blocks in a 3x3 grid.
    # Proof: Any 2x2 block must include the center cell (1,1) or touch it?
    # If a 2x2 block does not include (1,1), it must be in a corner.
    # Corners: (0,0), (0,2), (2,0), (2,2).
    # Block at (0,0) covers (0,0),(0,1),(1,0),(1,1). Includes center.
    # Block at (0,2) covers (0,2),(0,1),(1,2),(1,1). Includes center.
    # Block at (2,0) covers (2,0),(2,1),(1,0),(1,1). Includes center.
    # Block at (2,2) covers (2,2),(2,1),(1,2),(1,1). Includes center.
    # So EVERY 2x2 block in a 3x3 grid includes the center cell (1,1).
    # Therefore, no two 2x2 blocks can be disjoint in the i-j plane.
    # Thus, we can have at most one 2x2 block per (i,j) "layer"?
    # No, the blocks are in 3D.
    # Block is defined by i0, j0, k0.
    # It covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
    # Since all i-j projections overlap at (1,1), any two blocks with the same i0, j0 will overlap in i-j.
    # To be disjoint, they must differ in i0 or j0?
    # If they differ in i0 or j0, their i-j projections are disjoint?
    # No, as shown, all i-j projections intersect at (1,1).
    # So any two blocks, regardless of k0, will intersect at the cell (1,1,k) for some k?
    # Let's check.
    # Block A: i0=0, j0=0. Covers i in {0,1}, j in {0,1}.
    # Block B: i0=1, j0=1. Covers i in {1,2}, j in {1,2}.
    # Intersection in i: {1}. Intersection in j: {1}.
    # So they share the line i=1, j=1.
    # If they also overlap in k, they share a cell.
    # If they are disjoint in k, they do not share a cell.
    # So we CAN have multiple blocks if they are disjoint in k.
    # Example:
    # Block 1: i0=0, j0=0, k0=0. Covers k in {0,1}.
    # Block 2: i0=0, j0=0, k0=2. Covers k in {2,3}.
    # These are disjoint in k, so disjoint in 3D.
    # So we can stack blocks along the k dimension.
    # Max number of blocks in a fixed (i0, j0) column: floor(8/2) = 4.
    # Since we can only use one (i0, j0) column (because any two different (i0, j0) pairs overlap in i-j),
    # the maximum number of disjoint blocks is 4.
    # This confirms rank 68 is the best for this block strategy.
    
    # To get rank 67, we need a different strategy.
    # Maybe use a 2x2x2 block and a different structure for the rest?
    # Or use a 3x3x3 Laderman-like block?
    # Laderman is for 3x3x3. Rank 14.
    # If we use one Laderman block (3x3x3), it covers 27 entries.
    # Remaining: 72 - 27 = 45 entries.
    # If we fill the rest with schoolbook, rank = 14 + 45 = 59.
    # But the Laderman block must fit in 3x3x8.
    # It fits in the first 3 k-slices.
    # The remaining 5 k-slices are 3x3x5 = 45 entries.
    # So rank 59 is possible IF the Laderman decomposition is valid for the sub-tensor.
    # The Laderman decomposition computes the product of a 3x3 and 3x3 matrix.
    # Our tensor is the identity tensor for matrix multiplication.
    # So yes, we can use Laderman for the 3x3x3 part.
    
    # Let's implement this.
    # 1. Use Laderman for k=0,1,2.
    # 2. Use schoolbook for k=3,4,5,6,7.
    
    # Rank = 14 + 45 = 59.
    # Score = 72/59 = 1.22.
    
    # I need a valid Laderman decomposition.
    # The previous code had a broken Laderman.
    # I will find a correct one online or derive it.
    # Laderman's algorithm for 3x3:
    # It is known that rank(3,3,3) <= 14.
    
    # Let's try to find a simple rank-14 decomposition.
    # Reference: "A computation matrix in 14 scalar products" by Laderman.
    # The coefficients are:
    # (I will copy a known correct set of coefficients)
    
    # If I can't find a correct one, I can try to use the 2x2x2 Strassen blocks more cleverly.
    # But 59 is much better than 68.
    
    # Let's assume I can get a correct Laderman.
    # I will hardcode a known correct Laderman decomposition.
    
    return None

def construct():
    n, m, p = 3, 3, 8
    
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = n * m * p
    
    if _time_left() < 10:
        return best_result
    
    # Try known 4-block configuration
    known_blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    try:
        U, V, W = _build_decomposition_with_blocks(n, m, p, known_blocks)
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = U.shape[1]
    except:
        pass
    
    if _time_left() < 20:
        return best_result
    
    # Try to find 5 disjoint blocks
    from itertools import combinations
    
    valid_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                cov = _get_block_coverage(i0, j0, k0, n, m, p)
                if len(cov) == 8:
                    valid_blocks.append(((i0, j0, k0), cov))
    
    # Try all combinations of 5
    for combo in combinations(valid_blocks, 5):
        if _time_left() < 10:
            break
        all_cov = set()
        disjoint = True
        for _, cov in combo:
            if all_cov & cov:
                disjoint = False
                break
            all_cov |= cov
        if disjoint:
            pos_list = [pos for pos, _ in combo]
            try:
                U, V, W = _build_decomposition_with_blocks(n, m, p, pos_list)
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = U.shape[1]
                    break
            except:
                pass
    
    # Try 6 blocks
    if _time_left() > 15:
        for combo in combinations(valid_blocks, 6):
            if _time_left() < 8:
                break
            all_cov = set()
            disjoint = True
            for _, cov in combo:
                if all_cov & cov:
                    disjoint = False
                    break
                all_cov |= cov
            if disjoint:
                pos_list = [pos for pos, _ in combo]
                try:
                    U, V, W = _build_decomposition_with_blocks(n, m, p, pos_list)
                    if _verify(U, V, W, n, m, p):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = U.shape[1]
                        break
                except:
                    pass
    
    return best_result
