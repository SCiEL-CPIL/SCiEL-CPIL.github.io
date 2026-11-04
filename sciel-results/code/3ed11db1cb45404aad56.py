"""Tensor rank minimization for M_{2,3,8} via randomized numerical rank estimation."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]
DEADLINE = 115.0  # safety margin under 120s

def _start_time():
    return time.time()

def _time_left(start):
    return DEADLINE - (time.time() - start)

def _tensor_to_matrices(n, m, p):
    """Build the three unfoldings of the matrix multiplication tensor.
    T[i,j,k,l] = delta_{i,l} * delta_{j,k} (A_{i,j} * B_{j,k} -> C_{i,k})
    U-shape: (n*m, m*p) -> U[i*m+j, j*p+k] = delta_{i,?}... 
    Actually: T is 4-way with indices (i, j, k, l) where A[i,j], B[j,k], C[i,k].
    T[i,j,k,l] = 1 if (i==l and j==k) else 0.
    
    For rank-r decomposition: C[i,k] = sum_c U[i*m+j_c, c] * V[j_c*p+k, c] * W[k*n+i, c]
    Wait, the convention: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    W[k*n+i, r] is the trace convention.
    
    The tensor: C[i,k] = A[i,j] B[j,k] summed over j.
    T[i,j,k] for fixed... no, it's T_{i,j,k,l} = delta_{i,l} delta_{j,k}? 
    No. The multiplication tensor has T[i, j, k] where i indexes row of A (and C), 
    j indexes col of A / row of B (and the inner sum), k indexes col of B (and C).
    
    Actually the tensor is 3-way? No, for the bilinear form it's:
    C[i,k] = sum_j A[i,j] B[j,k]
    The structure tensor has entries T[i, j, k] = 1 (it's the "multiplication tensor")
    where i = row index, j = inner index, k = col index.
    
    But the decomposition uses U, V, W with the product:
    C[i,k] = sum_r U[i, r] * V[r, k]? No...
    
    Let me re-read: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W[k*n+i, r] trace convention.
    
    The standard: C[i,k] = sum_{j, r} A[i,j] U[j, r] V[r, k] ... no.
    
    The correct formulation: 
    C[i,k] = sum_r (sum_j A[i,j] U[j*n+j, r]) ... 
    
    Let me use the standard tensor decomposition for matmul:
    The tensor T of shape (n, m, p) where T[i,j,k] = 1 for all (i,j,k) 
    represents C[i,k] = sum_j A[i,j] B[j,k].
    
    Wait, that's not right either. Let me think carefully.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a bilinear map (A,B) -> C. The structure tensor is:
    T in R^{(n*m) x (m*p) x (p*n)}? No.
    
    Actually, the tensor rank of matmul M_{n,m,p}:
    We write C[i,k] = sum_{j} A[i,j] B[j,k]
    
    As a 3-tensor: T[i, j, k] where the "first mode" is (i,k) for C, 
    "second mode" is (i,j) for A, "third mode" is (j,k) for B.
    
    T[(i,k), (i,j), (j,k)] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1
    when the indices are consistent.
    
    More precisely: T is a tensor of shape (n*p, n*m, m*p) where
    T[i*p+k, i*m+j, j*p+k] = 1
    
    For a rank-r decomposition:
    T = sum_{c=1}^r u_c v_c^T w_c  (outer product of three vectors)
    
    Where u_c in R^{n*p}, v_c in R^{n*m}, w_c in R^{m*p}
    
    Then: C[i,k] = sum_c U[i*p+k, c] * (sum_j A[i,j] V[i*m+j, c]) * (sum_k B[j,k] W[j*p+k, c])
    
    Hmm, that's not quite right. Let me use the standard formulation.
    
    The matmul tensor M_{n,m,p} has rank r if we can find matrices:
    U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    
    such that for all (i,j,k):
    1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that would mean the tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k) pairs.
    
    Let me verify with schoolbook: for each (i,j,k) triple, we have one term with
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1, and the product is 1.
    Sum over c gives 1 for each valid triple. ✓
    
    So the tensor is: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T[a, b, c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    (and the indices are consistent: same j in a and b, same k in b and c, same i in c and a)
    
    For n=2, m=3, p=8:
    a in [0, 5], b in [0, 23], c in [0, 15]
    
    Valid triples: i in {0,1}, j in {0,1,2}, k in {0,...,7}
    a = i*3 + j, b = j*8 + k, c = k*2 + i
    
    So T[a, b, c] = 1 iff:
    a = i*3+j for some i,j
    b = j*8+k for some j,k  
    c = k*2+i for some k,i
    AND the j from a equals j from b, k from b equals k from c, i from c equals i from a.
    """
    n, m, p = 2, 3, 8
    # Build the 3-way tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _numerical_rank_via_randomized(T, n_iter=30, tol=1e-10):
    """Estimate the numerical rank of a 3-tensor via randomized SVD on unfoldings."""
    # Flatten and use the mode-1 unfolding (matrix of shape (n*m, m*p*p*n))
    # The rank of the tensor >= rank of any unfolding
    # But we want the tensor rank, which is >= max unfolding rank
    
    # Mode-1 unfolding: reshape T to (n*m, m*p * p*n)
    mode1 = T.reshape(T.shape[0], -1)
    # Randomized SVD
    n_cols = mode1.shape[1]
    # Use full SVD for this small matrix
    try:
        U, s, Vt = np.linalg.svd(mode1, full_matrices=False)
        # Numerical rank
        if len(s) > 0:
            rank_est = np.sum(s > tol * s[0])
        else:
            rank_est = 0
    except:
        rank_est = T.shape[0]
    
    return rank_est, s

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for M_{2,2,2}.
    
    The 7 rank-1 terms (u_c, v_c, w_c) where:
    C[i,k] = sum_c u_c[i,k] * v_c[i,j] * w_c[j,k]
    
    Actually for M_{2,2,2}: n=m=p=2
    a = i*2+j in [0,3], b = j*2+k in [0,3], c = k*2+i in [0,3]
    
    Strassen's 7 multiplications:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-A00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4
    C10 = M2 + M5
    C11 = M1 - M2 + M3 + M6
    
    In tensor form, each multiplication M_c corresponds to a rank-1 term.
    For M_c = (linear combo of A) * (linear combo of B), the tensor term is:
    u_c[i,k] * v_c[i,j] * w_c[j,k]
    
    where v_c[i,j] = coefficient of A[i,j] in the A-part
    w_c[j,k] = coefficient of B[j,k] in the B-part
    u_c[i,k] = coefficient of C[i,k] in the output
    
    Let me enumerate:
    
    M1 = (A00+A11)(B00+B11)
    v: A00=1, A11=1 -> v[0,0]=1, v[1,1]=1
    w: B00=1, B11=1 -> w[0,0]=1, w[1,1]=1
    C = M1 - M4 + M5 - M7, so M1 contributes: C00+=1, C01+=0, C10+=0, C11+=1
    u: u[0,0]=1, u[1,1]=1
    
    M2 = (A10+A11)B00
    v: A10=1, A11=1 -> v[1,0]=1, v[1,1]=1
    w: B00=1 -> w[0,0]=1
    C: C10+=1, C11+=(-1) [from -M2 in C11]
    u: u[1,0]=1, u[1,1]=-1
    
    M3 = A00(B01-B11)
    v: A00=1 -> v[0,0]=1
    w: B01=1, B11=-1 -> w[0,1]=1, w[1,1]=-1
    C: C01+=1, C11+=1
    u: u[0,1]=1, u[1,1]=1
    
    M4 = A11(B10-A00)
    v: A11=1 -> v[1,1]=1
    w: B10=1, B00=-1 -> w[1,0]=1, w[0,0]=-1
    C: C00+=(-1), C01+=1
    u: u[0,0]=-1, u[0,1]=1
    
    M5 = (A00+A01)B11
    v: A00=1, A01=1 -> v[0,0]=1, v[0,1]=1
    w: B11=1 -> w[1,1]=1
    C: C00+=1, C10+=1
    u: u[0,0]=1, u[1,0]=1
    
    M6 = (A10-A00)(B00+B01)
    v: A10=1, A00=-1 -> v[1,0]=1, v[0,0]=-1
    w: B00=1, B01=1 -> w[0,0]=1, w[0,1]=1
    C: C11+=1
    u: u[1,1]=1
    
    M7 = (A01-A11)(B10+B11)
    v: A01=1, A11=-1 -> v[0,1]=1, v[1,1]=-1
    w: B10=1, B11=1 -> w[1,0]=1, w[1,1]=1
    C: C00+=(-1), C11+=(-1)
    u: u[0,0]=-1, u[1,1]=-1
    """
    n = m = p = 2
    terms = []
    # Each term: (u, v, w) where u is 2x2, v is 2x2, w is 2x2
    # Flatten to vectors of length 4 (row-major: index = i*m+j)
    
    def mat2vec_2x2(M):
        return [M[0][0], M[0][1], M[1][0], M[1][1]]
    
    def zero():
        return [[0.0, 0.0], [0.0, 0.0]]
    
    # M1
    u1 = zero(); u1[0][0] = 1; u1[1][1] = 1
    v1 = zero(); v1[0][0] = 1; v1[1][1] = 1
    w1 = zero(); w1[0][0] = 1; w1[1][1] = 1
    terms.append((u1, v1, w1))
    
    # M2
    u2 = zero(); u2[1][0] = 1; u2[1][1] = -1
    v2 = zero(); v2[1][0] = 1; v2[1][1] = 1
    w2 = zero(); w2[0][0] = 1
    terms.append((u2, v2, w2))
    
    # M3
    u3 = zero(); u3[0][1] = 1; u3[1][1] = 1
    v3 = zero(); v3[0][0] = 1
    w3 = zero(); w3[0][1] = 1; w3[1][1] = -1
    terms.append((u3, v3, w3))
    
    # M4
    u4 = zero(); u4[0][0] = -1; u4[0][1] = 1
    v4 = zero(); v4[1][1] = 1
    w4 = zero(); w4[1][0] = 1; w4[0][0] = -1
    terms.append((u4, v4, w4))
    
    # M5
    u5 = zero(); u5[0][0] = 1; u5[1][0] = 1
    v5 = zero(); v5[0][0] = 1; v5[0][1] = 1
    w5 = zero(); w5[1][1] = 1
    terms.append((u5, v5, w5))
    
    # M6
    u6 = zero(); u6[1][1] = 1
    v6 = zero(); v6[1][0] = 1; v6[0][0] = -1
    w6 = zero(); w6[0][0] = 1; w6[0][1] = 1
    terms.append((u6, v6, w6))
    
    # M7
    u7 = zero(); u7[0][0] = -1; u7[1][1] = -1
    v7 = zero(); v7[0][1] = 1; v7[1][1] = -1
    w7 = zero(); w7[1][0] = 1; w7[1][1] = 1
    terms.append((u7, v7, w7))
    
    return terms

def _verify_decomposition(U, V, W, n=2, m=3, p=8, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check: for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_strassen_based(n=2, m=3, p=8):
    """Build a rank-44 decomposition using Strassen for the 2x2x2 core.
    
    Strategy: The 2x3x8 matmul can be seen as:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can group the j=0,1,2 terms. For the 2x2x2 sub-problems, use Strassen.
    
    Actually, a cleaner approach: 
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    This is 3 separate 2x1x8 multiplications, each of rank 2*1*8=16 schoolbook.
    Total: 3*16 = 48 (same as schoolbook).
    
    To get rank 44, we need to exploit structure. One known approach:
    Use Strassen on a 2x2x2 block. Since p=8, we can view B as having rows of length 8.
    
    Alternative: The known result is that R(M_{2,3,8}) = 44.
    One construction: partition the 3 rows of A into... 
    
    Let me try a different approach. Use the fact that:
    M_{2,3,8} can be decomposed by first doing a 2x2x2 Strassen on a subset.
    
    Actually, let me just use the schoolbook (rank 48) as a fallback and try to find 
    rank 44 via the Strassen embedding.
    
    A rank-44 construction:
    - Use 4 rank-1 terms for a 2x2x2 Strassen sub-block (saves 4 terms: 8->4? No, 8 schoolbook -> 7 Strassen, saves 1)
    
    Hmm, let me think differently. 
    
    For 2x3x8:
    Schoolbook: 2*3*8 = 48
    
    If I apply Strassen to a 2x2x2 sub-matrix, I save 1 (from 8 to 7) per 2x2x2 block.
    How many 2x2x2 blocks can I fit? 
    - i: 2 rows, so 1 block in i-direction
    - j: 3 cols, so 1 full 2x2 block (j=0,1) and 1 remaining (j=2)
    - k: 8 cols, so 4 full 2x2 blocks
    
    If I apply Strassen to the (i=0,1, j=0,1, k=0..7) part:
    The 2x2x8 part: C[i,k] for i in {0,1}, from j in {0,1}: 
    C_partial[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is a 2x2x8 matmul. Can I apply Strassen here?
    2x2x8: schoolbook rank = 2*2*8 = 32.
    Using Strassen on the 2x2 part: treat it as 4 separate 1x1x2 multiplications? No.
    
    Actually, 2x2x8 matmul: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    This has rank at most 2*2*8/... 
    
    The tensor rank of M_{2,2,p}: it's known that R(M_{2,2,p}) = 2p + 1 for p >= 2? 
    No, R(M_{2,2,2}) = 7, R(M_{2,2,p}) <= 2p+1 by one Strassen recursion? 
    
    Actually R(M_{2,2,p}) = 2p + 1 is not right. Let me compute:
    M_{2,2,p}: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    Schoolbook: 2*2*p = 4p terms.
    
    With Strassen on the 2x2: we can do it in 7*p? No, that's not how it works.
    
    I think the cleanest approach is to just implement rank 48 (schoolbook) which is correct,
    and note that the measured best is 44. Since I can't easily construct 44 in this time,
    let me at least return a valid rank-48 decomposition.
    
    But wait - the current best island score is 44 (from the knowledge: "reduces rank from 48 to 44, score 1.091").
    So I need to match or beat 44.
    
    Let me try to construct rank 44 explicitly.
    
    Known construction for R(M_{2,3,8}) = 44:
    
    One approach from the literature: 
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Write j = 0, 1, 2.
    
    Use the identity for 2x2x2 Strassen on the (i, j=0,1) part with a clever combination.
    
    Actually, let me just try: 48 - 4 = 44. So I need to save 4 terms.
    
    If I can find 4 pairs of terms that can be merged... 
    
    Alternatively, use the known result directly. The rank-44 decomposition exists.
    Let me construct it by combining Strassen on a 2x2x2 block with schoolbook on the rest.
    
    2x3x8 = 2x2x8 (for j=0,1) + 2x1x8 (for j=2)
    
    R(M_{2,2,8}) + R(M_{2,1,8})
    R(M_{2,1,8}) = 2*1*8 = 16 (schoolbook, and this is tight since it's rank 16)
    
    For R(M_{2,2,8}): C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    This is a 2x2x8 tensor. 
    
    Using Strassen recursively: M_{2,2,8} can be computed by treating it as 4 copies of M_{1,1,2} 
    with Strassen overhead? 
    
    Actually, R(M_{2,2,p}) for even p: 
    M_{2,2,p} = M_{2,2,p/2} x 2 (two independent halves in k)
    So R(M_{2,2,p}) <= 2 * R(M_{2,2,p/2})
    R(M_{2,2,2}) = 7
    R(M_{2,2,4}) <= 14
    R(M_{2,2,8}) <= 28
    
    Then R(M_{2,3,8}) <= R(M_{2,2,8}) + R(M_{2,1,8}) <= 28 + 16 = 44. ✓
    
    So the construction is:
    - Compute the j=0,1 part (2x2x8) using rank-28 Strassen recursion
    - Compute the j=2 part (2x1x8) using rank-16 schoolbook
    - Total: 44
    
    Now I need to implement the rank-28 decomposition of M_{2,2,8}.
    
    M_{2,2,8}: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k], i in {0,1}, k in {0,...,7}
    
    Split k into two halves: k = 0..3 and k = 4..7
    Each half is a 2x2x4 problem.
    
    M_{2,2,4}: split k into 0..1 and 2..3
    Each half is 2x2x2 (Strassen, rank 7).
    
    So: R(M_{2,2,8}) = 2 * R(M_{2,2,4}) = 2 * (2 * R(M_{2,2,2})) = 4 * 7 = 28. ✓
    
    Implementation:
    For M_{2,2,2}: use the 7 Strassen terms.
    For M_{2,2,4}: apply Strassen to k=0,1 and k=2,3 separately (two independent 2x2x2).
    For M_{2,2,8}: apply to k=0..3 and k=4..7 separately.
    
    Let me implement this recursively.
    """
    n, m, p = 2, 3, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: j=0,1 (2x2x8 using Strassen recursion)
    # Part 2: j=2 (2x1x8 schoolbook)
    
    # For Part 1: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Use Strassen recursion on the 2x2x8
    
    def add_strassen_2x2x2(i_offset, j0, k0):
        """Add Strassen terms for 2x2x2 block at i=i_offset,i_offset+1, j=j0,j0+1, k=k0,k0+1"""
        # For 2x2x2: indices i in {i_offset, i_offset+1}, j in {j0, j0+1}, k in {k0, k0+1}
        # Local indices: il = i - i_offset in {0,1}, jl = j - j0 in {0,1}, kl = k - k0 in {0,1}
        # a = i*m + j, b = j*p + k, c = k*n + i
        # But in the 2x2x2 sub-block, we're computing C[i,k] = A[i,j0]*B[j0,k] + A[i,j0+1]*B[j0+1,k]
        
        terms = _strassen_2x2x2()
        for (u, v, w) in terms:
            # u is 2x2: u[il, kl] -> global U[i_offset+il, ...] 
            # But U has shape (n*m, rank) where the row index is i*m+j
            # Wait, U[i*m+j, c] and we need U indexed by (i,j) pairs
            # 
            # In the decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # 
            # For a rank-1 term in the 2x2x2 sub-block:
            # Contribution to C[i,k] = u[il,kl] * v[il,jl] * w[jl,kl]
            # where il = i - i_offset, jl = j - j0, kl = k - k0
            #
            # This means: U[i_offset*2 + il][c] *= u[il, kl]? No...
            # 
            # Wait, I'm confusing myself. Let me re-derive.
            # 
            # The full decomposition: C[i,k] = sum_{j,c} A[i,j] * (U part) * B[j,k] * (V part) * (W part)
            # 
            # No, the standard form is:
            # C[i,k] = sum_c (sum_j A[i,j] * U_c[i,j]) * (sum_k B[j,k] * V_c[j,k]) ... 
            # 
            # I think I have the convention wrong. Let me re-read the task.
            #
            # "U', 'V', 'W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # "W[k*n+i,r] is the trace convention"
            #
            # The bilinear form: C[i,k] = sum_{j, c} A[i,j] B[j,k] * (something)
            #
            # Actually, the standard tensor decomposition for matmul:
            # C[i,k] = sum_c U_c[i,k] * sum_j A[i,j] V_c[i,j] * sum_k' B[j,k'] W_c[j,k']
            # No, that doesn't work dimensionally.
            #
            # Let me use the schoolbook as reference:
            # For each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
            # Then: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for each (i,j,k)
            #
            # And C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c A[i,j] B[j,k] delta_{i*m+j, ?} ...
            #
            # I think the correct interpretation is:
            # C[i,k] = sum_j A[i,j] B[j,k] 
            #          = sum_j sum_c A[i,j] B[j,k] * [U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]]
            # where the bracket is 1 for the schoolbook term corresponding to (i,j,k).
            #
            # So the general form is:
            # C[i,k] = sum_{j,c} A[i,j] B[j,k] U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            #
            # Wait, that has both j in the sum and j in the indices. That seems off.
            #
            # Let me try: C[i,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c]) * W[k*n+i, c]
            # No, that mixes j.
            #
            # I think the correct form must be:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[j, c]) * (sum_j B[j,k] V[j, c])
            # where U has shape (m, rank) and V has shape (m, rank)?
            # But the task says U has shape (n*m, rank).
            #
            # OK let me just go with what the schoolbook does:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Schoolbook: for each (i,j,k), one term with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
            # Verification: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should give the coefficient of A[i,j]B[j,k] in C[i,k]
            # For schoolbook: when c corresponds to (i',j',k'), the product is 1 iff i'=i, j'=j, k'=k
            # So sum_c gives 1 for each (i,j,k), and C[i,k] = sum_j 1 * A[i,j] B[j,k] = sum_j A[i,j]B[j,k] ✓
            #
            # So the formula is:
            # C[i,k] = sum_{j} sum_c A[i,j] B[j,k] U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            #
            # Wait, but j appears in U's row index AND in the sum. So for a fixed (i,k), we sum over j and c:
            # C[i,k] = sum_j sum_c A[i,j] B[j,k] U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            #
            # For schoolbook, U[i*m+j, c] is nonzero only when c corresponds to (i,j,k') for some k'
            # This is getting complicated. Let me just verify numerically.
            #
            # For a rank-1 term with U_vec, V_vec, W_vec:
            # Contribution to C[i,k] = sum_j A[i,j] B[j,k] U_vec[i*m+j] V_vec[j*p+k] W_vec[k*n+i]
            #
            # This is NOT a rank-1 bilinear form in (A,B) because of the sum over j.
            #
            # I think the correct interpretation is that the tensor is:
            # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
            # And the rank-r decomposition means:
            # T = sum_c u_c v_c w_c (3-way tensor decomposition)
            # where u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n
            #
            # Then: T[a,b,c] = sum_r u_c[a] v_c[b] w_c[c]
            # And the bilinear form is:
            # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{a,b,c} T[a,b,c] A[?] B[?]
            #
            # I'm going in circles. Let me just implement the schoolbook (which is known correct)
            # and verify it passes, then try to optimize.
            pass
    
    # Just do schoolbook for now - rank 48
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

def construct():
    start = _start_time()
    
    # First, try to build a rank-44 decomposition
    # Strategy: 2x2x8 via Strassen recursion (rank 28) + 2x1x8 schoolbook (rank 16) = 44
    
    n, m, p = 2, 3, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        """u_vec: length n*m, v_vec: length m*p, w_vec: length p*n"""
        col = len(U_list)
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    def zero_u():
        return [0] * (n * m)
    def zero_v():
        return [0] * (m * p)
    def zero_w():
        return [0] * (p * n)
    
    # Strassen 2x2x2 terms (local indices il, jl, kl in {0,1})
    # Each term has u[il,kl], v[il,jl], w[jl,kl]
    # Global: i = i_offset + il, j = j_offset + jl, k = k_offset + kl
    # U index: i*m + j = (i_offset+il)*m + (j_offset+jl)
    # V index: j*p + k = (j_offset+jl)*p + (k_offset+kl)
    # W index: k*n + i = (k_offset+kl)*n + (i_offset+il)
    
    strassen_terms = [
        # (u 2x2, v 2x2, w 2x2)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),  # M1
        ([[0,0],[1,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),  # M2
        ([[0,1],[0,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),  # M3
        ([[-1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]), # M4
        ([[1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),   # M5
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),  # M6
        ([[-1,0],[0,-1]], [[0,1],[-1,0]], [[0,1],[1,1]]),# M7
    ]
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add 7 Strassen terms for 2x2x2 block at offsets (i_off, j_off, k_off).
        i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        """
        for (u, v, w) in strassen_terms:
            u_vec = zero_u()
            v_vec = zero_v()
            w_vec = zero_w()
            for il in range(2):
                for jl in range(2):
                    for kl in range(2):
                        i = i_off + il
                        j = j_off + jl
                        k = k_off + kl
                        u_vec[i * m + j] = u[il][kl]
                        v_vec[j * p + k] = v[il][jl]
                        w_vec[k * n + i] = w[jl][kl]
            add_term(u_vec, v_vec, w_vec)
    
    # Part 1: j=0,1 (2x2x8 via Strassen recursion)
    # Split k=0..7 into k=0..3 and k=4..7
    # Each 2x2x4 split into two 2x2x2 blocks
    # k=0,1: add_strassen_block(0, 0, 0)
    # k=2,3: add_strassen_block(0, 0, 2)
    # k=4,5: add_strassen_block(0, 0, 4)
    # k=6,7: add_strassen_block(0, 0, 6)
    
    add_strassen_block(0, 0, 0)  # i=0,1; j=0,1; k=0,1
    add_strassen_block(0, 0, 2)  # i=0,1; j=0,1; k=2,3
    add_strassen_block(0, 0, 4)  # i=0,1; j=0,1; k=4,5
    add_strassen_block(0, 0, 6)  # i=0,1; j=0,1; k=6,7
    
    # Part 2: j=2 (2x1x8 schoolbook)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            u_vec = zero_u()
            v_vec = zero_v()
            w_vec = zero_w()
            u_vec[i * m + 2] = 1
            v_vec[2 * p + k] = 1
            w_vec[k * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_list)
    
    # Verify
    if rank != 44:
        # Fallback to schoolbook
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
    
    # Convert to the required format (list of lists)
    U_out = U_list
    V_out = V_list
    W_out = W_list
    
    # Verify correctness
    if not _verify_decomposition(U_out, V_out, W_out, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U_out, "V": V_out, "W": W_out}
