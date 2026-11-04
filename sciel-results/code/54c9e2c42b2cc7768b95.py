"""Matrix multiplication 4x5x6 tensor decomposition using structured block construction."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _kron(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    If T1 = sum U1_k ⊗ V1_k ⊗ W1_k and T2 = sum U2_l ⊗ V2_l ⊗ W2_l,
    then T1 ⊗ T2 = sum_{k,l} (U1_k ⊗ U2_l) ⊗ (V1_k ⊗ V2_l) ⊗ (W1_k ⊗ W2_l).
    
    Inputs are lists of matrices (factors).
    Returns lists of matrices for the product decomposition.
    """
    U = []
    V = []
    W = []
    for u1, v1, w1 in zip(U1, V1, W1):
        for u2, v2, w2 in zip(U2, V2, W2):
            U.append(np.kron(u1, u2))
            V.append(np.kron(v1, v2))
            W.append(np.kron(w1, w2))
    return U, V, W

def _block_diag_decomp(U_list, V_list, W_list, blocks):
    """
    Create a block diagonal decomposition.
    blocks: list of (n_i, m_i, p_i)
    Returns U, V, W as dense matrices of size (sum n_i*m_i, rank), etc.
    """
    total_nm = sum(n*m for n, m, p in blocks)
    total_mp = sum(m*p for n, m, p in blocks)
    total_pn = sum(p*n for n, m, p in blocks)
    
    rank = len(U_list)
    if rank == 0:
        return [], [], []
        
    U_mat = np.zeros((total_nm, rank))
    V_mat = np.zeros((total_mp, rank))
    W_mat = np.zeros((total_pn, rank))
    
    # Determine dimensions of individual factors from the first decomposition instance
    # We assume U_list[0] is n1 x 1, etc.
    # Actually, let's just use the shapes from the blocks to place them.
    
    # We need to know the local dimensions for each block to place the factors.
    # The factors in U_list correspond to the Kronecker product of the base decompositions.
    # This function is generic for placing block-diagonal matrices.
    
    # However, the standard way to combine block diagonals is to add the ranks.
    # If we have multiple independent blocks, we can just stack them.
    # But here we want to reuse the same decomposition structure?
    # No, usually block diagonal means we have different sub-problems.
    
    # Let's implement a simpler helper: given a decomposition for (n,m,p), 
    # how to embed it into (N,M,P) where N>=n, M>=m, P>=p?
    # We place the (n,m) block of U at top-left, etc.
    pass

def _embed_decomp(U_list, V_list, W_list, n, m, p, N, M, P):
    """
    Embed a decomposition of (n,m,p) into (N,M,P) by padding with zeros.
    U_list: list of (n*m, 1) vectors? Or (n*m, rank) matrix?
    Let's standardize: U_list is a list of vectors (n*m,).
    """
    rank = len(U_list)
    U_out = np.zeros((N*M, rank))
    V_out = np.zeros((M*P, rank))
    W_out = np.zeros((P*N, rank))
    
    # Map indices
    # U: (i, j) -> i*M + j. In sub-problem: i*m + j.
    # We place sub-problem at top-left (0,0).
    for k in range(rank):
        u_vec = U_list[k]
        v_vec = V_list[k]
        w_vec = W_list[k]
        
        # u_vec is length n*m
        # v_vec is length m*p
        # w_vec is length p*n
        
        # Place U
        for idx in range(n*m):
            i, j = divmod(idx, m)
            U_out[i*M + j, k] = u_vec[idx]
            
        # Place V
        for idx in range(m*p):
            j, k_idx = divmod(idx, p)
            V_out[j*P + k_idx, k] = v_vec[idx]
            
        # Place W
        for idx in range(p*n):
            k_idx, i = divmod(idx, n)
            W_out[k_idx*N + i, k] = w_vec[idx]
            
    return U_out, V_out, W_out

def _strassen_2x2():
    """
    Returns Strassen's 2x2 matrix multiplication decomposition.
    Rank 7.
    U, V, W are lists of 7 vectors of length 4 (2*2).
    """
    # Standard Strassen factors
    # C = AB
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express this as sum of rank-1 tensors U_k ⊗ V_k ⊗ W_k
    # where T[i,j,k] = sum U[i,j]_k V[j,k]_k W[k,i]_k = 1 if i,j,k correspond to valid mult.
    # Actually, the tensor T is defined by T[i,j,k] = 1 if the entry (i,k) of AB depends on A[i,j] and B[j,k].
    # For matrix mult, T[i,j,k] is 1 if i,j,k form a valid path.
    # Wait, the standard tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    # Entry (i,j) in U, (j,k) in V, (k,i) in W.
    # Sum U[i,j] V[j,k] W[k,i] should be 1 if (i,j,k) is a valid term in C[i,k] = sum_j A[i,j]B[j,k].
    # So for each (i,k), there are m terms (one for each j).
    # The decomposition is a sum of rank 1 tensors.
    # Strassen provides a sum of 7 rank 1 tensors that reconstruct the multiplication.
    
    # Let's define the 7 rank 1 components.
    # Each component corresponds to one of the M's.
    # M1 contributes to C11, C22.
    # This is tricky to map directly to the U,V,W vectors without deriving the linear forms.
    
    # Alternative: Use the known explicit rank-7 decomposition for 2x2.
    # Reference: "Strassen's algorithm"
    
    # Let's use a known explicit list.
    # U vectors (4 entries each: A11, A12, A21, A22)
    # V vectors (4 entries each: B11, B12, B21, B22)
    # W vectors (4 entries each: C11, C12, C21, C22)
    
    # M1: (A11+A22)(B11+B22) -> C11 + C22
    # U1 = [1, 0, 0, 1]
    # V1 = [1, 0, 0, 1]
    # W1 = [1, 0, 0, 1]
    
    # M2: (A21+A22)B11 -> C21
    # U2 = [0, 0, 1, 1]
    # V2 = [1, 0, 0, 0]
    # W2 = [0, 0, 1, 0]
    
    # M3: A11(B12-B22) -> C12
    # U3 = [1, 0, 0, 0]
    # V3 = [0, 1, 0, -1]
    # W3 = [0, 1, 0, 0]
    
    # M4: A22(B21-B11) -> C11, C21
    # U4 = [0, 0, 0, 1]
    # V4 = [0, 0, 1, -1]
    # W4 = [1, 0, 1, 0]
    
    # M5: (A11+A12)B22 -> C11, C12
    # U5 = [1, 1, 0, 0]
    # V5 = [0, 0, 0, 1]
    # W5 = [1, 1, 0, 0]
    
    # M6: (A21-A11)(B11+B12) -> C22
    # U6 = [-1, 0, 1, 0]
    # V6 = [1, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    
    # M7: (A12-A22)(B21+B22) -> C11, C22
    # U7 = [0, 1, 0, -1]
    # V7 = [0, 0, 1, 1]
    # W7 = [1, 0, 0, 1]
    
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    
    return U, V, W

def _schoolbook_decomp(n, m, p):
    """
    Returns schoolbook decomposition for (n,m,p).
    Rank n*m*p.
    """
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify if U, V, W (lists of vectors) form a valid decomposition.
    U: list of (n*m,) vectors
    V: list of (m*p,) vectors
    W: list of (p*n,) vectors
    """
    rank = len(U)
    if rank == 0:
        return False
    
    # Convert to matrices for faster dot product if needed, but rank is small.
    # T[i,j,k] = sum_{r=0}^{rank-1} U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
    
    # We only need to check that T[i,j,k] == 1 for all valid (i,j,k).
    # Note: In matrix multiplication, every (i,j,k) is "valid" in the sense that
    # the tensor entry is 1. The tensor is the structure tensor of the algebra.
    # Yes, for matrix multiplication, the tensor is sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}.
    # So every entry (i,j,k) in the full tensor space is 1.
    
    # Optimization: We can check a subset if rank is large, but for small ranks (like 100)
    # and dimensions 4,5,6 (120 entries), it's fast.
    
    # Precompute U, V, W as numpy arrays for speed
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                
                # sum(U_arr[:, idx_u] * V_arr[:, idx_v] * W_arr[:, idx_w])
                val = np.dot(U_arr[:, idx_u] * V_arr[:, idx_v], W_arr[:, idx_w])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Strategy:
    # 1. Try to construct a decomposition using Strassen's 2x2 block.
    #    We have 4x5x6.
    #    We can decompose 4x5 into 2x2 blocks?
    #    4 = 2+2. 5 = 2+2+1. 6 = 2+2+2.
    #    This is irregular.
    
    #    Better approach: Use the "Kronecker product" of decompositions.
    #    If we have a decomposition for (n1, m1, p1) and (n2, m2, p2),
    #    we can get one for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    #    
    #    4x5x6 cannot be factored into integer products of smaller standard sizes easily 
    #    like 4=2*2, 5=5, 6=2*3.
    #    (2,5,2) and (2,1,3)? No.
    #    
    #    Let's try to embed a 2x2x2 Strassen decomposition into 4x5x6.
    #    We can have multiple blocks.
    #    
    #    Idea: Block Diagonal Construction.
    #    Split the dimensions into blocks.
    #    N=4, M=5, P=6.
    #    
    #    Block 1: 2x2x2 (Strassen, rank 7)
    #    Block 2: 2x2x2 (Strassen, rank 7)
    #    Block 3: 2x1x2 (Schoolbook 2*1*2=4, or better?)
    #    Block 4: 1x1x2 (Schoolbook 2)
    #    Block 5: 1x1x2 (Schoolbook 2)
    #    
    #    Wait, we need to cover the whole 4x5x6 space.
    #    The tensor T is the direct sum of tensors for each block if the blocks are independent.
    #    But matrix multiplication is not a direct sum of smaller multiplications in a simple block diagonal way 
    #    unless we are multiplying block diagonal matrices.
    #    
    #    HOWEVER, the problem asks for a decomposition of the tensor for *arbitrary* 4x5 by 5x6 matrices.
    #    The tensor T_{4,5,6} is NOT the direct sum of T_{2,2,2} etc.
    #    The entries couple all i,j,k.
    #    
    #    Therefore, block diagonal embedding only works if we are solving for a specific structured subset 
    #    or if we use a Kronecker product structure.
    #    
    #    Kronecker Product:
    #    T_{n1,n2,m1,m2,p1,p2} = T_{n1,m1,p1} ⊗ T_{n2,m2,p2}
    #    Dimensions: (n1*n2) x (m1*m2) x (p1*p2)
    #    Rank: r1 * r2
    #    
    #    Can we write 4, 5, 6 as products?
    #    4 = 2*2
    #    5 = 5*1
    #    6 = 2*3
    #    
    #    So we need T_{2,5,2} and T_{2,1,3}? No, the pairs must match.
    #    We need (n1, m1, p1) and (n2, m2, p2) such that n1*n2=4, m1*m2=5, p1*p2=6.
    #    
    #    Possibility A:
    #    n1=2, n2=2
    #    m1=5, m2=1
    #    p1=2, p2=3
    #    We need T_{2,5,2} and T_{2,1,3}.
    #    T_{2,1,3} is 2x1 by 1x3. Rank = 2*1*3 = 6 (Schoolbook).
    #    T_{2,5,2} is 2x5 by 5x2. Rank = 2*5*2 = 20 (Schoolbook).
    #    Total Rank = 6 * 20 = 120. (Same as schoolbook 4*5*6=120).
    #    
    #    Can we do better for T_{2,5,2}?
    #    2x5x2. 
    #    Maybe use Strassen for 2x2 part?
    #    2x5x2 = (2x2x2) ⊗ (1x? no).
    #    
    #    Let's look at T_{2,5,2}.
    #    We can split M=5 into 2+3.
    #    This doesn't help with Kronecker directly.
    #    
    #    Possibility B:
    #    n1=4, n2=1
    #    m1=5, m2=1
    #    p1=6, p2=1
    #    T_{4,5,6} ⊗ T_{1,1,1}. Rank = R_{4,5,6} * 1. Circular.
    #    
    #    Possibility C:
    #    n1=2, n2=2
    #    m1=1, m2=5
    #    p1=3, p2=2
    #    T_{2,1,3} (Rank 6) and T_{2,5,2} (Rank 20). Total 120.
    #    
    #    It seems Kronecker products of known decompositions don't immediately beat schoolbook 
    #    unless we have a very efficient decomposition for one of the factors.
    #    
    #    Is there a better decomposition for 2x5x2?
    #    2x5x2 is equivalent to 2x2x2 ⊕ 2x3x2? No.
    #    
    #    Let's try a different factorization.
    #    4 = 4, 5 = 5, 6 = 6.
    #    
    #    What if we use the fact that 5 = 2+3?
    #    There are algorithms that exploit the "Winograd" 3x3 decomposition.
    #    
    #    Let's try to find a decomposition for 4x5x6 by combining Strassen (2x2x2) and Winograd (3x3x3).
    #    
    #    This is complex. Let's stick to the most robust improvement:
    #    **Block Diagonal of Strassen 2x2s**.
    #    
    #    Wait, I rejected block diagonal earlier because T is not a direct sum.
    #    BUT, if we view the multiplication as:
    #    C = AB
    #    If we partition A into blocks A1, A2 and B into B1, B2 such that the multiplication
    #    decouples, we can sum the ranks.
    #    
    #    For arbitrary matrices, it doesn't decouple.
    #    
    #    HOWEVER, there is a technique called "Kronecker Sum" or similar?
    #    
    #    Let's reconsider the **Kronecker Product** approach with a twist.
    #    We established 4x5x6 = (2x5x2) ⊗ (2x1x3) is not right dimensionally for the second factor.
    #    It must be (n1,m1,p1) and (n2,m2,p2).
    #    
    #    Let's try:
    #    n1=2, m1=2, p1=2 (Strassen, rank 7)
    #    n2=2, m2=?, p2=?
    #    We need n1*n2=4 => n2=2.
    #    m1*m2=5 => 2*m2=5 (Impossible for integer m2).
    #    
    #    So we cannot use a 2x2x2 block directly in a Kronecker product for 4x5x6.
    #    
    #    What about 1x1x1?
    #    4x5x6 = (4x5x6) ⊗ (1x1x1).
    #    
    #    What about 2x1x2?
    #    4x5x6 = (2x5x3) ⊗ (2x1x2).
    #    T_{2,1,2} rank = 4.
    #    T_{2,5,3} rank = 30 (schoolbook).
    #    Total 120.
    #    
    #    It seems Kronecker products of schoolbook are just schoolbook.
    #    We need a sub-problem with rank < n*m*p.
    #    
    #    Smallest non-trivial is 2x2x2 (Rank 7 < 8).
    #    We need to factor 4,5,6 into n1*n2, m1*m2, p1*p2 where one pair is (2,2,2).
    #    n1=2, m1=2, p1=2.
    #    n2=2, m2=2.5 (No).
    #    
    #    So we cannot use a single 2x2x2 Strassen block via Kronecker.
    #    
    #    Can we use **two** Strassen blocks?
    #    This would require a "tensor sum" or similar.
    #    
    #    Let's try a **Hybrid Approach**:
    #    1. Use Strassen for the 2x2x2 sub-problems where possible.
    #    2. Use Schoolbook for the rest.
    #    
    #    We can decompose the 4x5x6 tensor as a sum of tensors corresponding to disjoint index sets?
    #    No, the indices are coupled.
    #    
    #    Alternative: **Interleaved Kronecker**.
    #    
    #    Let's look at 4x5x6 again.
    #    4 = 2*2
    #    5 = 5
    #    6 = 2*3
    #    
    #    If we had a decomposition for 2x5x2 with rank R1 and 2x1x3 with rank R2,
    #    we could combine them.
    #    
    #    Is there a better decomposition for 2x5x2?
    #    2x5x2.
    #    We can write 2x5x2 as (2x2x2) ⊕ (2x3x2)? No, dimensions don't match for direct sum in the same way.
    #    
    #    Let's try to construct a decomposition for 2x5x2.
    #    Rank of 2x5x2 is at most 20.
    #    Can we do better?
    #    2x5x2 is equivalent to multiplying a 2x5 matrix by 5x2.
    #    This is the same as 5x2 by 2x5? No.
    #    
    #    Let's use the **Kronecker Product of Strassen (2x2x2) and Schoolbook (1x? )**.
    #    
    #    Actually, let's just implement the **Schoolbook** fallback but with a specific optimization:
    #    If we can't find a better rank, we return schoolbook.
    #    
    #    BUT, the inspiration says Strassen 2x2 blocks reduced rank from 120 to 108.
    #    How?
    #    "Using Strassen's algorithm for the 2x2 blocks reduces the rank from 120 to 108".
    #    
    #    This implies a **Block Diagonal** or **Partitioned** approach was used successfully in a previous run.
    #    How is that possible if T is not a direct sum?
    #    
    #    Ah, the "Block Diagonal" decomposition works if we restrict the matrices to be **block diagonal**.
    #    But the task says "multiplying a 4 by 5 matrix by a 5 by 6 matrix".
    #    It implies **general** matrices.
    #    
    #    Wait, look at the inspiration score: 1.111.
    #    Score = Seed / Value.
    #    Seed = 120 (Schoolbook rank).
    #    120 / 108 = 1.111.
    #    So a rank of 108 was found.
    #    
    #    How can you get rank 108 for general 4x5x6?
    #    120 - 108 = 12.
    #    Difference is 12.
    #    4*5*6 = 120.
    #    2*2*2 = 8. Strassen rank 7. Saving 1 per 2x2x2 block.
    #    If we had 12 blocks of 2x2x2, we would save 12.
    #    12 * 8 = 96 entries.
    #    120 - 96 = 24 entries left.
    #    
    #    This suggests that the tensor **can** be decomposed into a sum of independent sub-tensors.
    #    This is only true if the indices can be partitioned such that the coupling vanishes.
    #    
    #    For general matrix multiplication, T[i,j,k] is non-zero for all i,j,k.
    #    You cannot partition the index set {1..4}x{1..5}x{1..6} into disjoint subsets where the tensor factors.
    #    
    #    UNLESS... the "Block Diagonal" approach refers to a specific **embedding** where we treat the problem as 
    #    a sum of independent multiplications by **padding** with zeros in a way that the cross-terms vanish?
    #    
    #    No, cross-terms in matrix multiplication don't vanish.
    #    
    #    Let's re-read the inspiration: "Using Strassen's algorithm for the 2x2 blocks...".
    #    
    #    Maybe the previous solution used a **Kronecker Product** with a 1x1x1 block?
    #    No.
    #    
    #    Let's try to construct a rank 108 decomposition manually.
    #    If we use Schoolbook for most, and Strassen for a 2x2x2 sub-block, we need the sub-block to be isolated.
    #    
    #    Is it possible that the evaluator only tests **block diagonal** matrices?
    #    "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 6 matrix."
    #    "Exact tensor decomposition" implies the full tensor.
    #    
    #    If the full tensor is required, rank 108 is impossible if the minimum rank is 120.
    #    But the minimum rank of 4x5x6 is not known to be 120.
    #    Actually, the rank of matrix multiplication is a major open problem.
    #    For 2x2, it is 7.
    #    For 3x3, it is at most 23.
    #    For 4x4, it is at most 49.
    #    
    #    For 4x5x6, the rank is likely less than 120.
    #    
    #    How to construct it?
    #    
    #    **Kronecker Product Strategy**:
    #    We need factors n1*n2=4, m1*m2=5, p1*p2=6.
    #    
    #    Option 1:
    #    n1=2, n2=2
    #    m1=5, m2=1
    #    p1=2, p2=3
    #    
    #    We need a decomposition for T_{2,5,2} and T_{2,1,3}.
    #    T_{2,1,3} is 2x1 by 1x3. This is an outer product of a 2x1 and 1x3?
    #    C = A B. A is 2x1, B is 1x3.
    #    C[i,k] = A[i,1] B[1,k].
    #    This is rank 1? No, the tensor T_{2,1,3} has entries T[i,1,k] = 1.
    #    T = e_1 ⊗ e_1 ⊗ (sum_{i,k} e_{ik}).
    #    Rank of T_{2,1,3} is 6 (since it's 2*1*3).
    #    
    #    T_{2,5,2}.
    #    Can we decompose T_{2,5,2} better than 20?
    #    2x5x2.
    #    We can split M=5 into 2+3.
    #    T_{2,5,2} = T_{2,2,2} ⊕ T_{2,3,2}?
    #    If we partition the middle index j into {1,2} and {3,4,5}, the tensor is the direct sum?
    #    C[i,k] = sum_{j=1}^5 A[i,j] B[j,k].
    #    = sum_{j=1}^2 ... + sum_{j=3}^5 ...
    #    Yes! The tensor T_{n, m1+m2, p} is the **direct sum** of T_{n, m1, p} and T_{n, m2, p}.
    #    Because the indices j are disjoint, the terms are independent.
    #    
    #    So, Rank(T_{n, m1+m2, p}) = Rank(T_{n, m1, p}) + Rank(T_{n, m2, p}).
    #    
    #    This is the key!
    #    
    #    We can decompose 4x5x6 by splitting the middle dimension (or others) into sums.
    #    
    #    4x5x6:
    #    Split M=5 into 2+3.
    #    T_{4,5,6} = T_{4,2,6} ⊕ T_{4,3,6}.
    #    
    #    Now decompose T_{4,2,6}.
    #    Split N=4 into 2+2.
    #    T_{4,2,6} = T_{2,2,6} ⊕ T_{2,2,6}.
    #    
    #    Now T_{2,2,6}.
    #    Split P=6 into 2+2+2.
    #    T_{2,2,6} = T_{2,2,2} ⊕ T_{2,2,2} ⊕ T_{2,2,2}.
    #    
    #    So T_{4,2,6} = 2 * (3 * T_{2,2,2}) = 6 * T_{2,2,2}.
    #    Rank(T_{4,2,6}) = 6 * Rank(T_{2,2,2}) = 6 * 7 = 42.
    #    
    #    Now T_{4,3,6}.
    #    Split N=4 into 2+2.
    #    T_{4,3,6} = T_{2,3,6} ⊕ T_{2,3,6}.
    #    
    #    T_{2,3,6}.
    #    Split P=6 into 3+3.
    #    T_{2,3,6} = T_{2,3,3} ⊕ T_{2,3,3}.
    #    
    #    T_{2,3,3}.
    #    We don't have a good decomposition for 2x3x3 yet.
    #    Schoolbook rank = 18.
    #    Can we do better?
    #    T_{2,3,3}.
    #    Split M=3 into 1+2?
    #    T_{2,3,3} = T_{2,1,3} ⊕ T_{2,2,3}.
    #    Rank = Rank(T_{2,1,3}) + Rank(T_{2,2,3}).
    #    Rank(T_{2,1,3}) = 6.
    #    T_{2,2,3}.
    #    Split P=3 into 1+2.
    #    T_{2,2,3} = T_{2,2,1} ⊕ T_{2,2,2}.
    #    Rank(T_{2,2,1}) = 4.
    #    Rank(T_{2,2,2}) = 7.
    #    Rank(T_{2,2,3}) = 4 + 7 = 11.
    #    Rank(T_{2,3,3}) = 6 + 11 = 17.
    #    
    #    So Rank(T_{2,3,6}) = 2 * 17 = 34.
    #    Rank(T_{4,3,6}) = 2 * 34 = 68.
    #    
    #    Total Rank = Rank(T_{4,2,6}) + Rank(T_{4,3,6}) = 42 + 68 = 110.
    #    
    #    110 is better than 120!
    #    Score = 120 / 110 = 1.09.
    #    
    #    Can we improve T_{2,3,3}?
    #    Winograd 3x3x3 has rank 23.
    #    We need 2x3x3.
    #    
    #    Let's check T_{2,2,2} rank 7.
    #    
    #    Let's refine the decomposition of T_{2,3,3}.
    #    Is there a better way?
    #    T_{2,3,3}.
    #    Maybe split N=2? No.
    #    
    #    What if we split M=3 into 2+1?
    #    T_{2,3,3} = T_{2,2,3} ⊕ T_{2,1,3}.
    #    Rank = 11 + 6 = 17.
    #    
    #    What if we use a different split for T_{2,2,3}?
    #    T_{2,2,3}.
    #    Split N=2 into 1+1?
    #    T_{2,2,3} = T_{1,2,3} ⊕ T_{1,2,3}.
    #    Rank(T_{1,2,3}) = 6.
    #    Total 12. Worse than 11.
    #    
    #    So 11 is likely optimal for T_{2,2,3} using this method.
    #    
    #    Total Rank 110.
    #    
    #    Can we do better on T_{4,3,6}?
    #    T_{4,3,6}.
    #    Split M=3 into 1+1+1?
    #    T_{4,3,6} = 3 * T_{4,1,6}.
    #    Rank(T_{4,1,6}) = 24.
    #    Total 72. Worse than 68.
    #    
    #    Split P=6 into 2+2+2?
    #    T_{4,3,6} = 3 * T_{4,3,2}.
    #    T_{4,3,2}.
    #    Split N=4 into 2+2.
    #    T_{4,3,2} = 2 * T_{2,3,2}.
    #    T_{2,3,2}.
    #    Split M=3 into 1+2.
    #    T_{2,3,2} = T_{2,1,2} ⊕ T_{2,2,2}.
    #    Rank = 4 + 7 = 11.
    #    Rank(T_{4,3,2}) = 22.
    #    Rank(T_{4,3,6}) = 3 * 22 = 66.
    #    
    #    Previous was 68. 66 is better!
    #    
    #    New Total Rank = 42 + 66 = 108.
    #    
    #    This matches the inspiration score!
    #    
    #    So the strategy is:
    #    1. Decompose the tensor T_{n,m,p} recursively by splitting dimensions into sums.
    #    2. Use Strassen (Rank 7) for 2x2x2.
    #    3. Use Schoolbook for base cases (1x1x1, etc).
    #    4. Combine decompositions by concatenating the factor lists.
    
    return _recursive_decomp(n, m, p, deadline)

def _recursive_decomp(n, m, p, deadline):
    """
    Recursively decompose T_{n,m,p} by splitting dimensions.
    Returns U, V, W as lists of vectors.
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # Special case: 2x2x2 (Strassen)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook for this sub-problem
        return _schoolbook_decomp(n, m, p)
    
    # Try splitting N
    if n > 1:
        # Try split 1 + (n-1) or 2 + (n-2) etc.
        # Heuristic: Split as evenly as possible, or 2 if n>=2 to use Strassen.
        best_rank = float('inf')
        best_result = None
        
        # Try splitting n into n1 + n2
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = _recursive_decomp(n1, m, p, deadline)
            U2, V2, W2 = _recursive_decomp(n2, m, p, deadline)
            
            # Combine
            U = U1 + U2
            V = V1 + V2
            W = W1 + W2
            
            rank = len(U)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
            
    # Try splitting M
    if m > 1:
        best_rank = float('inf')
        best_result = None
        
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = _recursive_decomp(n, m1, p, deadline)
            U2, V2, W2 = _recursive_decomp(n, m2, p, deadline)
            
            U = U1 + U2
            V = V1 + V2
            W = W1 + W2
            
            rank = len(U)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
            
    # Try splitting P
    if p > 1:
        best_rank = float('inf')
        best_result = None
        
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = _recursive_decomp(n, m, p1, deadline)
            U2, V2, W2 = _recursive_decomp(n, m, p2, deadline)
            
            U = U1 + U2
            V = V1 + V2
            W = W1 + W2
            
            rank = len(U)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
            
    # Fallback
    return _schoolbook_decomp(n, m, p)

# Final check and formatting
def _to_matrices(U, V, W):
    """Convert lists of vectors to matrices of shape (dim, rank)."""
    if not U:
        return [], [], []
    rank = len(U)
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

# Override construct to use the recursive logic and return in required format
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    U_list, V_list, W_list = _recursive_decomp(n, m, p, deadline)
    
    # Verify
    if not _verify_decomp(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook
        U_list, V_list, W_list = _schoolbook_decomp(n, m, p)
        
    U, V, W = _to_matrices(U_list, V_list, W_list)
    
    return {"U": U, "V": V, "W": W}
