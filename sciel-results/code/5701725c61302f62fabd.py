"""Tensor rank minimization for 2x4x4 matrix multiplication via ILP-free combinatorial search."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 4]

# Deadline guard: evaluator kills after 150s, we want a safety margin
_DEADLINE = 145.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _remaining():
    return _DEADLINE - _elapsed()

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    Convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] summed over r
    Actually the tensor equality is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 if the standard matrix multiplication has a 1 there.
    
    Wait, let me re-read the convention. The matrix multiplication tensor T is defined by:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor form: T[i,j,k] such that C[i,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] * ... 
    
    Actually for the decomposition: C = A * B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 (it's the standard tensor).
    The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me think about this more carefully.
    
    In tensor decomposition of matrix multiplication, we have:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the appropriate sense).
    
    Actually no. The matrix multiplication tensor T is defined as:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (with appropriate indices)
    
    More precisely, if we index A as A[i,j] (i in 0..n-1, j in 0..m-1),
    B as B[j,k] (j in 0..m-1, k in 0..p-1),
    C as C[i,k] (i in 0..n-1, k in 0..p-1),
    
    then C[i,k] = sum_j A[i,j] B[j,k].
    
    The bilinear form: C = f(A, B) where f is the matrix multiplication map.
    The tensor T associated with this bilinear map satisfies:
    C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{i',j',k'} A[i',j'] B[j',k'] T[i',j',k',i,k]
    
    Hmm, let me use the standard convention from the problem statement.
    
    The problem says: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Output factor follows the official trace convention W[k*n+i, r].
    
    So the decomposition is:
    For the matrix multiplication tensor, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    where T[i,j,k] = 1 for all valid (i,j,k) - wait, that's not right.
    
    Let me look at the schoolbook decomposition in the seed:
    For each (i,j,k): U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    
    So the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    In the schoolbook, for each (i,j,k) triple, there's exactly one r where all three are 1,
    so T[i,j,k] = 1 for all (i,j,k).
    
    But wait, that would mean the tensor is all-ones, which doesn't correspond to matrix multiplication.
    
    Let me reconsider. The matrix multiplication tensor should satisfy:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In terms of a trilinear form or tensor: the "tensor" T[i,j,k] encodes the coefficient
    such that the bilinear map (A,B) -> C can be written as:
    C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * (something for C index)
    
    Actually, I think the correct interpretation is:
    The matrix multiplication can be viewed as a bilinear map from (A, B) to C.
    We can write: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k, r] * B[j,k']) * W[k*n+i, r]
    
    No wait, that's not right either. Let me think about it differently.
    
    The standard way: we want to compute C = A*B.
    We introduce intermediate variables: for each r, compute s_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    Then t_r = sum_{j,k} V[j*p+k, r] * B[j,k]  
    Then C[i,k] = sum_r W[k*n+i, r] * s_r * t_r
    
    For this to equal A*B, we need:
    C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''])
    
    = sum_{i',j',j'',k''} (sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k'', r]) * A[i',j'] * B[j'',k'']
    
    This should equal sum_j A[i,j] * B[j,k].
    
    So we need: sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k'', r] = delta_{i',i} * delta_{j',j''} * delta_{k'',k}
    
    In other words, for fixed (i, i', j', j'', k, k''):
    sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k'', r] = 1 if i'=i, j'=j'', k''=k, else 0.
    
    So the tensor T is defined by: T[i, j, k] where the three "slices" are:
    - First mode indexed by (i, j) [flattened as i*m+j] - this is the A-index
    - Second mode indexed by (j, k) [flattened as j*p+k] - this is the B-index  
    - Third mode indexed by (k, i) [flattened as k*n+i] - this is the C-index
    
    And T[a, b, c] = 1 if the indices are consistent (i.e., the j in A-index matches the j in B-index, and the i in A-index matches the i in C-index, and the k in B-index matches the k in C-index), else 0.
    
    More precisely, if a = i1*m + j1, b = j2*p + k2, c = k3*n + i3, then:
    T[a, b, c] = 1 if i1 == i3, j1 == j2, k2 == k3, else 0.
    
    And the decomposition is: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    Let me verify with the schoolbook: for a given (i,j,k), a = i*m+j, b = j*p+k, c = k*n+i.
    T[a,b,c] = 1 since i1=i3=i, j1=j2=j, k2=k3=k.
    And the schoolbook sets U[a,r]=1, V[b,r]=1, W[c,r]=1 for a unique r, so sum_r = 1. ✓
    
    For inconsistent indices, say a = i1*m+j1, b = j2*p+k2, c = k3*n+i3 where j1 != j2:
    T[a,b,c] = 0. In the schoolbook, there's no r where all three are 1 simultaneously (since each r corresponds to exactly one (i,j,k) triple), so sum_r = 0. ✓
    
    Great, so now I understand the problem correctly.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check shapes
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # Compute T_decomposed = sum_r U[:,r] * V[:,r] * W[:,r] (outer product sum)
    # T_decomposed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need this to equal T_true[a, b, c]
    
    # T_true[a, b, c] = 1 if indices are consistent, else 0
    # a = i1*m + j1, b = j2*p + k2, c = k3*n + i3
    # consistent iff i1 == i3, j1 == j2, k2 == k3
    
    # Number of nonzero entries in T_true: n * m * p (one for each consistent (i,j,k))
    # Total entries: n*m * m*p * p*n = (n*m*p)^2 = 32^2 = 1024 for our case
    
    # For efficiency, only check the nonzero and a sample of zero entries
    # Actually, let's check all entries since it's only 1024
    
    # Build T_true as a sparse check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check some zero entries (sample)
    # For speed, check all zero entries if rank is small enough
    # Total entries = n*m * m*p * p*n
    total_entries = (n*m) * (m*p) * (p*n)
    if total_entries <= 100000:
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                for c in range(p*n):
                    k3 = c // n
                    i3 = c % n
                    # Skip if consistent
                    if i1 == i3 and j1 == j2 and k2 == k3:
                        continue
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val) > 1e-9:
                        return False
    else:
        # Sample zero entries
        np.random.seed(42)
        for _ in range(10000):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k2 = b % p
            k3 = c // n
            i3 = c % n
            if i1 == i3 and j1 == j2 and k2 == k3:
                continue
            val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
            if abs(val) > 1e-9:
                return False
    
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W as lists of lists, each of shape (4, 7) for n=m=p=2.
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i].
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our indexing: i in {0,1}, j in {0,1}, k in {0,1}
    # A[i,j], B[j,k], C[i,k]
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # r=0: M1 = (a00+a11)*(b00+b11)
    #   U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c11=1
    # r=1: M2 = (a10+a11)*b00
    #   U: a10=1, a11=1; V: b00=1; W: c10=1
    # r=2: M3 = a00*(b01-b11)
    #   U: a00=1; V: b01=1, b11=-1; W: c01=1
    # r=3: M4 = a11*(b10-b00)
    #   U: a11=1; V: b10=1, b00=-1; W: c10=1, c11=-1
    # r=4: M5 = (a00+a01)*b11
    #   U: a00=1, a01=1; V: b11=1; W: c01=-1, c11=1
    # r=5: M6 = (a10-a00)*(b00+b01)
    #   U: a10=1, a00=-1; V: b00=1, b01=1; W: c11=1
    # r=6: M7 = (a01-a11)*(b10+b11)
    #   U: a01=1, a11=-1; V: b10=1, b11=1; W: c00=1, c01=-1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11
    
    # r=1: M2
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # c10
    
    # r=2: M3
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11
    W[1][2] = 1  # c01
    
    # r=3: M4
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00
    W[2][3] = 1  # c10
    W[3][3] = -1 # c11
    
    # r=4: M5
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[1][4] = -1 # c01
    W[3][4] = 1  # c11
    
    # r=5: M6
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # c11
    
    # r=6: M7
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    W[0][6] = 1  # c00
    W[1][6] = -1 # c01
    
    return U, V, W


def _combine_decompositions(decs):
    """Combine multiple block decompositions into a full decomposition.
    
    Each dec is (U_block, V_block, W_block, i_offset, j_offset, k_offset)
    where U_block has shape (n_block*m_block, rank_block), etc.
    
    Actually, let me think about this differently. For 2x4x4, we can view it as:
    - Two 2x4x4 blocks (splitting n into two 2s)
    - Or a 2x2x2 Strassen block combined with something
    
    Let me use a block structure approach.
    """
    pass


def _try_rank(target_rank):
    """Try to find a decomposition of the given rank using combinatorial search.
    
    Strategy: Use the Strassen decomposition as a building block and try to combine
    blocks to cover the 2x4x4 case.
    """
    if _remaining() < 5:
        return None
    
    n, m, p = DIMENSIONS
    
    # Approach 1: Use Strassen 2x2x2 blocks
    # 2x4x4 can be seen as:
    # - Split m into two 2s: 2x2x4 and 2x2x4 blocks
    # - Split p into two 2s: 2x4x2 and 2x4x2 blocks
    # - Split n into two 2s: 2x4x4 = two 1x4x4... no that doesn't work
    # - Split into 2x2x2 blocks: we have 1 * 2 * 2 = 4 blocks of 2x2x2
    #   Total rank with Strassen: 4 * 7 = 28
    #   Schoolbook: 32
    #   So 28 is already an improvement!
    
    # Let's construct the 4-block Strassen decomposition
    # Block (i0, j0, k0) where i0 in {0,1}, j0 in {0,1}, k0 in {0,1}
    # covers A[i0*2..i0*2+1, j0*2..j0*2+1] * B[j0*2..j0*2+1, k0*2..k0*2+1] -> C[i0*2..i0*2+1, k0*2..k0*2+1]
    
    # For each block, we use the Strassen 2x2x2 decomposition
    # U_block[i_local*2+j_local, r] maps to U[(i0*2+i_local)*4 + (j0*2+j_local), r_global]
    # V_block[j_local*2+k_local, r] maps to V[(j0*2+j_local)*4 + (k0*2+k_local), r_global]  
    # W_block[k_local*2+i_local, r] maps to W[(k0*2+k_local)*2 + (i0*2+i_local), r_global]
    
    U = [[0] * 28 for _ in range(n*m)]  # 8 x 28
    V = [[0] * 28 for _ in range(m*p)]  # 16 x 28
    W = [[0] * 28 for _ in range(p*n)]  # 8 x 28
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    col = 0
    for i0 in range(n // 2):
        for j0 in range(m // 2):
            for k0 in range(p // 2):
                for r in range(7):
                    for i_l in range(2):
                        for j_l in range(2):
                            i_global = i0 * 2 + i_l
                            j_global = j0 * 2 + j_l
                            a_idx = i_global * m + j_global
                            U[a_idx][col] = U_str[i_l * 2 + j_l][r]
                    
                    for j_l in range(2):
                        for k_l in range(2):
                            j_global = j0 * 2 + j_l
                            k_global = k0 * 2 + k_l
                            b_idx = j_global * p + k_global
                            V[b_idx][col] = V_str[j_l * 2 + k_l][r]
                    
                    for k_l in range(2):
                        for i_l in range(2):
                            k_global = k0 * 2 + k_l
                            i_global = i0 * 2 + i_l
                            c_idx = k_global * n + i_global
                            W[c_idx][col] = W_str[k_l * 2 + i_l][r]
                    
                    col += 1
    
    if col != 28:
        return None
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank_with_mixed_blocks():
    """Try mixed block structures to potentially get rank < 28.
    
    For 2x4x4, we can try:
    - One 2x2x2 Strassen block (rank 7) + three 2x2x2 Strassen blocks (rank 7 each) = 28
    - Or try to use the asymmetry: n=2, m=4, p=4
    
    Alternative: split m=4 into 2+2, giving two 2x2x4 subproblems.
    Each 2x2x4 can be seen as two 2x2x2 blocks: rank 7+7=14 per subproblem, total 28.
    
    Or split p=4 into 2+2: two 2x4x2 subproblems, each = two 2x2x2 blocks = 14, total 28.
    
    The known lower bound for 2x4x4 matrix multiplication rank... 
    The rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = max(8, 16, 8) = 16.
    And at most n*m*p = 32.
    
    With Strassen blocks, we get 28. Can we do better?
    
    Let me try a different approach: use the fact that n=2 is small.
    C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0,1,2,3}, j in {0,1,2,3}.
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    This is like multiplying a 2x4 matrix by a 4x4 matrix.
    
    We can think of it as: for each k, C[:,k] = A * B[:,k] where B[:,k] is a 4x1 vector.
    So we're doing 4 matrix-vector multiplications of size 2x4 * 4x1.
    
    A 2x4 * 4x1 multiplication: C[i] = sum_j A[i,j] B[j] for i in {0,1}.
    This is a bilinear map from (A, B) to C where A is 2x4 and B is 4x1.
    The tensor rank of this... it's essentially two dot products of length 4.
    The rank of a single dot product of length 4 is 4 (you need 4 products).
    Two independent dot products: rank 8.
    
    But wait, the A matrix is shared across all k values. So there might be savings.
    
    Let me try a direct combinatorial search for rank < 28.
    """
    if _remaining() < 10:
        return None
    
    n, m, p = DIMENSIONS
    
    # Try rank 27, 26, ... down to some limit
    # For each rank, we need to find U (8 x r), V (16 x r), W (8 x r)
    # such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a very hard problem in general. Let me try a heuristic approach.
    
    # Heuristic: Start from the rank-28 Strassen block decomposition and try to
    # merge or eliminate one rank term.
    
    # Alternative: Use a different block structure.
    # What if we use one 2x4x4 "full" decomposition approach?
    
    # Let me try: view 2x4x4 as involving the 2x2x2 Strassen structure differently.
    # Split into: top-left 2x2x2, top-right 2x2x2, bottom-left 2x2x2, bottom-right 2x2x2
    # That's what we did: 4 blocks * 7 = 28.
    
    # Can we share some rank terms between blocks?
    # The blocks are:
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block (0,1): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block (1,0): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block (1,1): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Wait, I think I made an error in my block decomposition. Let me re-examine.
    # n=2, m=4, p=4.
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # 
    # If I split j into {0,1} and {2,3}, and k into {0,1} and {2,3}:
    # Block (j0, k0) for j0 in {0,1}, k0 in {0,1}:
    #   A[:, j0*2:(j0+1)*2] * B[j0*2:(j0+1)*2, k0*2:(k0+1)*2] -> C[:, k0*2:(k0+1)*2]
    #   This is a 2x2 x 2x2 -> 2x2 multiplication, i.e., 2x2x2.
    #   4 such blocks, each rank 7, total 28.
    
    # My earlier code was correct. Let me verify it works.
    
    # Let me try to see if we can get rank 27 by modifying the block structure.
    # One idea: use a different decomposition for one of the 2x2x2 blocks that
    # shares structure with another block.
    
    # For now, let's just return the rank-28 solution if it verifies.
    result = _try_rank(28)
    if result is not None:
        return result
    
    return None


def construct():
    global _start_time
    _start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # First, try the 4-block Strassen decomposition (rank 28)
    result = _try_rank(28)
    if result is not None:
        return result
    
    # If that fails, fall back to schoolbook (rank 32)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
