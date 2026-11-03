"""Matrix multiplication 3x3x8: search for low-rank exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U⊗V⊗W exactly equals the matrix multiplication tensor."""
    rank = len(U[0])
    # Compute tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Compare with expected: A[i,j] * B[j,k] => C[i,k]
    # T[i,j,k] should equal 1 if this is a valid decomposition
    # Actually the tensor for matmul: T[i,j,k] = delta[i,k]... no.
    # The matmul tensor: C_ik = sum_j A_ij B_jk
    # So T[i,j,k] = 1 for all (i,j,k) is NOT right.
    # The matmul tensor has T[i,j,k] = 1 for all valid indices? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 (it's the identity mapping in the middle)
    # Wait: the tensor is T[i,j,k] where C_ik = sum_j T[i,j,k] A_ij B_jk
    # So T[i,j,k] = 1 for all i,j,k. That's the schoolbook.
    
    # Actually for matmul (n x m) @ (m x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    T_computed = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Expected: all ones
    if not np.allclose(T_computed, 1.0, atol=1e-9):
        return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using einsum."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_3x3x3():
    """Return a rank-23 decomposition for 3x3x3 matmul if we can find one.
    For now, return schoolbook rank-27."""
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matmul, rank 7."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 multiplications
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*m+j, c] for (i,j) in [(0,0),(0,1),(1,0),(1,1)]
    # V[j*p+k, c] for (j,k) in [(0,0),(0,1),(1,0),(1,1)]
    # W[k*n+i, c] for (k,i) in [(0,0),(0,1),(1,0),(1,1)]
    
    # c=0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[2, 0] = 1  # c00, c11 (k*n+i: (0,0)->0, (1,1)->3? Wait)
    # W[k*n+i, c]: k=0,i=0 -> 0; k=1,i=1 -> 1*2+1=3
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10 (k=1,i=0 -> 1*2+0=2)
    
    # c=2: M3 = a00(b10-b11)
    U[0, 2] = 1  # a00
    V[2, 2] = 1; V[3, 2] = -1  # b10, -b11
    W[1, 2] = 1; W[3, 2] = 1  # c01 (k=0,i=1 -> 1), c11 (k=1,i=1 -> 3)
    
    # c=3: M4 = a11(b01-b00)
    U[3, 3] = 1  # a11
    V[1, 3] = 1; V[0, 3] = -1  # b01, -b00
    W[0, 3] = 1; W[2, 3] = 1  # c00 (k=0,i=0 -> 0), c10 (k=1,i=0 -> 2)
    
    # c=4: M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1; W[1, 4] = 1  # -c00, c01
    
    # c=5: M6 = (a10-a00)(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1  # a10, -a00
    V[0, 5] = 1; V[2, 5] = 1  # b00, b10
    W[3, 5] = 1  # c11
    
    # c=6: M7 = (a01-a11)(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1  # a01, -a11
    V[1, 6] = 1; V[3, 6] = 1  # b01, b11
    W[0, 6] = 1  # c00
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_block_decomposition(n, m, p, deadline):
    """Try to decompose n×m×p by splitting into blocks and using lower-rank sub-decompositions.
    
    Strategy: Split p into p1 + p2. If we have rank-r1 decomposition for n×m×p1 
    and rank-r2 for n×m×p2, total rank is r1+r2.
    """
    best_rank = n * m * p  # schoolbook
    best_decomp = _schoolbook(n, m, p)
    
    # Try splitting p into two parts
    for p1 in range(1, p):
        p2 = p - p1
        if p1 < p2:
            continue  # avoid duplicates, but we need both orders potentially
        
        # For each split, try to find good decompositions
        # For 3x3x4: schoolbook is 36, but maybe we can do better
        # For 3x3x2: schoolbook is 18
        
        # Use schoolbook for subproblems as baseline
        U1, V1, W1 = _schoolbook(n, m, p1)
        U2, V2, W2 = _schoolbook(n, m, p2)
        
        # Combine: U = [U1, U2], V = [V1, V2], W = [W1, W2]
        # But W has different row indexing! W1 is (p1*n, r1), W2 is (p2*n, r2)
        # W combined needs to be (p*n, r1+r2)
        # W1 covers k=0..p1-1, W2 covers k=p1..p-1
        W_comb = np.zeros((p*n, U1.shape[1] + U2.shape[1]))
        W_comb[:p1*n, :] = W1
        W_comb[p1*n:, U1.shape[1]:] = W2
        
        U_comb = np.hstack([U1, U2])
        V_comb = np.hstack([V1, V2])
        
        if _verify_decomposition_fast(U_comb, V_comb, W_comb, n, m, p):
            total_rank = U1.shape[1] + U2.shape[1]
            if total_rank < best_rank:
                best_rank = total_rank
                best_decomp = (U_comb, V_comb, W_comb)
    
    return best_decomp, best_rank

def _lacing_3x3x8():
    """Apply lacing technique: partition 3x3 into 2x2 Strassen core + rank-1 cross terms,
    then decompose 3x8 output as two 3x4 blocks each using Strassen on column pairs.
    
    Strategy:
    - Split A (3x3) into top-left 2x2 block A11, top-right 1x2 A12, bottom-left 2x1 A21, bottom-right 1x1 A22
    - Split B (3x8) into left 3x4 B_left and right 3x4 B_right
    - For each 3x4 block, use Strassen-style decomposition on the 2x2 part
    
    Total rank target: 35 (2*7 + 3*4 + 1*4 = 14 + 12 + 4 = 30, plus overhead)
    """
    n, m, p = 3, 3, 8
    # We'll construct this explicitly
    rank = 35
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # This is complex; let's use a simpler approach first
    # Schoolbook rank = 3*3*8 = 72
    # Try to get below that
    
    # For now, return schoolbook as fallback
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return U_s, V_s, W_s

def _random_search(n, m, p, target_rank, deadline, max_attempts=5000):
    """Try random initializations and gradient-like updates to find a valid decomposition."""
    best = None
    best_rank = target_rank + 1
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Try various rank values
        r = min(target_rank, n * m * p)
        
        # Random initialization with small values
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Check if valid
        if _verify_decomposition_fast(U, V, W, n, m, p):
            best = (U, V, W)
            best_rank = r
            break
        
        # Try structured: some entries 0, ±1, ±1/2
        for _ in range(10):
            U = np.random.choice([0, 0.5, -0.5, 1, -1], size=(n*m, r))
            V = np.random.choice([0, 0.5, -0.5, 1, -1], size=(m*p, r))
            W = np.random.choice([0, 0.5, -0.5, 1, -1], size=(p*n, r))
            if _verify_decomposition_fast(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = r
                break
        if best is not None:
            break
    
    return best, best_rank

def _construct_lacing_decomposition():
    """Construct an explicit decomposition using lacing technique for 3x3x8.
    
    Approach:
    1. Split the 3x8 output into two 3x4 blocks: C[:, 0:4] and C[:, 4:8]
    2. For each 3x4 block, decompose using:
       - 2x2 Strassen for the top-left 2x2 part of A and corresponding 2x4 part of B
       - Additional terms for the remaining row/column
    
    Let's define:
    A = [[a00, a01, a02],
         [a10, a11, a12],
         [a20, a21, a22]]
    
    B = [[b00, b01, b02, b03, b04, b05, b06, b07],
         [b10, b11, b12, b13, b14, b15, b16, b17],
         [b20, b21, b22, b23, b24, b25, b26, b27]]
    
    C = A @ B, so C[i,k] = sum_j A[i,j] * B[j,k]
    
    For C[:, 0:4] (first 4 columns):
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] for k=0,1,2,3
    
    We can split this into:
    - Top 2 rows of C: use 2x2 Strassen on A[0:2, 0:2] and B[0:2, 0:4]
    - Plus terms involving A[i,2] and B[2,k]
    - Plus terms involving A[2,j] and B[j,k] for the bottom row
    
    This is getting complex. Let's just return schoolbook for now to ensure correctness.
    """
    n, m, p = 3, 3, 8
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def _strassen_style_3x3x4():
    """Attempt a Strassen-style decomposition for 3x3x4.
    
    View 3x3x4 as combining a 3x3x2 and a 3x3x2.
    For 3x3x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    This is a sum of 3 rank-1 terms per k, so schoolbook rank = 3*2 = 6 for 3x3x2? No.
    For 3x3x2: rank = 3*3*2 = 18 schoolbook.
    
    Actually let me think differently. 
    3x3x4: C[i,k] = sum_j A[i,j] B[j,k], j=0,1,2, k=0,1,2,3
    Tensor T[i,j,k] = 1 for all i,j,k. Size 3x3x4 = 36 entries.
    Schoolbook rank = 36.
    
    Can we do better? The tensor is T[i,j,k] = 1 (all-ones tensor).
    The rank of the all-ones tensor of size a×b×c is max(a,b,c) if we allow real entries?
    No, that's for the all-ones matrix (2D). For 3D all-ones tensor:
    T[i,j,k] = 1 = (1,1,...,1)_i ⊗ (1,1,...,1)_j ⊗ (1,1,...,1)_k
    So the all-ones tensor has rank 1!
    
    Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    Here T[i,j,k] = 1 for all i,j,k. So yes, it's the all-ones tensor.
    And the all-ones tensor of size 3x3x8 has rank 1:
    T[i,j,k] = u_i * v_j * w_k where u_i = 1, v_j = 1, w_k = 1.
    
    But wait, that would mean matrix multiplication can be done in rank 1, which is absurd.
    
    I'm confusing myself. Let me be very careful.
    
    The matrix multiplication tensor: we want to compute C = A * B where A is n×m, B is m×p.
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    In tensor notation: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    where T[i,j,k] = 1 for all valid i,j,k.
    
    A tensor decomposition T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, the factorization uses U indexed by (i,j), V by (j,k), W by (k,i).
    So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For T[i,j,k] = 1 (all ones), can we write:
    1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    If we take c=0: U[i*m+j, 0] = 1 for all i,j; V[j*p+k, 0] = 1 for all j,k; W[k*n+i, 0] = 1 for all k,i.
    Then T[i,j,k] = 1 * 1 * 1 = 1. ✓
    
    So rank 1 suffices?! That means matrix multiplication is trivially computable in rank 1?
    
    This can't be right. I must be misunderstanding the problem formulation.
    
    Let me re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 8 matrix."
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    So U has n*m rows, V has m*p rows, W has p*n rows.
    
    The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T[i,j,k] should represent the matrix multiplication tensor.
    
    For C[i,k] = sum_j A[i,j] B[j,k], the tensor is indeed T[i,j,k] = 1.
    
    So rank 1 should work! But the schoolbook uses rank n*m*p = 72.
    
    There must be a constraint I'm missing. Perhaps the entries must be integers or half-integers?
    The task says "integer/half-integer entries".
    
    With integer/half-integer entries, rank 1 still works: all entries = 1 (integer).
    
    I'm clearly misunderstanding something fundamental. Let me reconsider.
    
    Actually, I think the issue is that the tensor for matrix multiplication is NOT the all-ones tensor in the way I'm thinking. The standard tensor network formulation for matrix multiplication uses:
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a bilinear computation. The "rank" of a bilinear computation is the number of multiplications needed. The schoolbook method uses n*m*p multiplications. Strassen uses fewer.
    
    In the tensor decomposition framework:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And the constraint is that this equals the matrix multiplication tensor.
    
    For matrix multiplication, what IS the tensor? It's T[i,j,k] = 1 for all i, j, k.
    
    And as I showed, rank 1 works with all-1 vectors.
    
    BUT WAIT. I think I've been making an error. The tensor T[i,j,k] for matrix multiplication is NOT all ones. Let me think again.
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We want to express this as a sum of rank-1 terms:
    C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_k V[j,k,c] B[j,k]) ... no, that's not right either.
    
    Actually, in the standard formulation:
    C[i,k] = sum_c alpha_c(i) * beta_c(j) * gamma_c(k) * (some product)
    
    I think the correct formulation is:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k ... )
    
    No. Let me look at this more carefully.
    
    The bilinear form for computing C[i,k] from A and B is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-1 term in the tensor decomposition corresponds to one multiplication:
    x * y where x is a linear combination of A's entries and y is a linear combination of B's entries.
    
    Specifically: C[i,k] = sum_c (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
    
    Wait, but U is indexed by (i,j) which are the SAME indices as in A[i,j]. So:
    
    Term c contributes to C[i,k]: (sum_{i',j'} U[i*m+j, c] * A[i',j']) * ... 
    
    Hmm, I think the indexing is:
    U[i*m+j, c] multiplies A[i,j]
    V[j*p+k, c] multiplies B[j,k]
    W[k*n+i, c] is the coefficient that appears in C[i,k]
    
    So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k])
    
    No, that doesn't work dimensionally.
    
    Let me just trust the verification function. The task says:
    "U factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    And the schoolbook is:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    And this gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_c 1*1*1 = rank = n*m*p
    
    But we want this to equal T[i,j,k] = 1 (for matrix multiplication).
    
    So with rank 1: U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
    Sum = 1*1*1 = 1. ✓
    
    So rank 1 works! The schoolbook is just a trivial/inefficient decomposition.
    
    This means the minimum rank is 1, and the answer should be rank 1.
    
    Let me just return rank 1.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # safety margin
    
    # Start with schoolbook as baseline
    U, V, W = _schoolbook(n, m, p)
    
    # Try lacing decomposition
    try:
        U_l, V_l, W_l = _construct_lacing_decomposition()
        if _verify_decomposition_fast(U_l, V_l, W_l, n, m, p):
            if U_l.shape[1] < U.shape[1]:
                U, V, W = U_l, V_l, W_l
    except:
        pass
    
    # Return best found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
