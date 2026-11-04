import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication via Strassen, rank 7.
    A is 2x2, B is 2x2, C = AB is 2x2.
    U is 4x7, V is 4x7, W is 4x7.
    Indexing: A[i,j] -> U[i*2+j, t], B[i,j] -> V[i*2+j, t], C[i,j] -> W[i*2+j, t]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen:
    # C11 = M1 + M4 - M5 + M6
    # C12 = M2 + M5
    # C21 = M3 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # U rows: a11=0, a12=1, a21=2, a22=3
    # V rows: b11=0, b12=1, b21=2, b22=3
    # W rows: c11=0, c12=1, c21=2, c22=3
    
    # M1: (a11+a22)*(b11+b22)
    # U: a11=1, a22=1 -> rows 0,3
    # V: b11=1, b22=1 -> rows 0,3
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[2,0]=1  # Wait, c11 gets +M1, c22 gets +M1
    
    # Let me redo carefully
    U[:,0]=0; V[:,0]=0; W[:,0]=0
    U[0,0]=1; U[3,0]=1  # a11, a22
    V[0,0]=1; V[3,0]=1  # b11, b22
    W[0,0]=1; W[3,0]=1  # c11, c22 both get +1
    
    # M2: (a21+a22)*b11
    U[:,1]=0; V[:,1]=0; W[:,1]=0
    U[2,1]=1; U[3,1]=1  # a21, a22
    V[0,1]=1  # b11
    W[1,1]=1; W[2,1]=-1  # c12 gets +M2? No: c12 = M2+M5, c21 = M3+M4
    # Wait: c12 = M2+M5, c21 = M3+M4, c11 = M1+M4-M5+M6, c22 = M1-M2+M3+M7
    W[1,1]=1  # c12 += M2
    # M2 doesn't appear in c21
    
    # M3: a11*(b12-b22)
    U[:,2]=0; V[:,2]=0; W[:,2]=0
    U[0,2]=1  # a11
    V[1,2]=1; V[3,2]=-1  # b12, -b22
    W[1,2]=-1; W[2,2]=1  # c21 += M3, c22 += M3? No c12=M2+M5, c21=M3+M4
    # c21 = M3+M4, so W[2,2]=1. c22=M1-M2+M3+M7, so W[3,2]=1.
    W[2,2]=1; W[3,2]=1
    
    # M4: a22*(b21-b11)
    U[:,3]=0; V[:,3]=0; W[:,3]=0
    U[3,3]=1  # a22
    V[2,3]=1; V[0,3]=-1  # b21, -b11
    W[0,3]=1; W[2,3]=1  # c11 += M4, c21 += M4
    
    # M5: (a11+a12)*b22
    U[:,4]=0; V[:,4]=0; W[:,4]=0
    U[0,4]=1; U[1,4]=1  # a11, a12
    V[3,4]=1  # b22
    W[0,4]=-1; W[1,4]=1  # c11 -= M5, c12 += M5
    
    # M6: (a21-a11)*(b11+b12)
    U[:,5]=0; V[:,5]=0; W[:,5]=0
    U[2,5]=1; U[0,5]=-1  # a21, -a11
    V[0,5]=1; V[1,5]=1  # b11, b12
    W[0,5]=1  # c11 += M6
    
    # M7: (a12-a22)*(b21+b22)
    U[:,6]=0; V[:,6]=0; W[:,6]=0
    U[1,6]=1; U[3,6]=-1  # a12, -a22
    V[2,6]=1; V[3,6]=1  # b21, b22
    W[3,6]=1  # c22 += M7
    
    return U, V, W

def _decompose_1xm_by_mxp(m, p):
    """Decompose 1xm x mxp -> 1xp. Rank = m (for m<=p) or p (for p<m).
    Actually for 1xm * mxp, rank is min(m,p) if we can share, but generally m.
    For 1x4 * 4x7, rank = 4 (since m=4, p=7, and we can use 4 terms).
    A is 1x4, B is 4x7, C is 1x7.
    C_k = sum_j A_j * B_jk
    This is a sum of 4 products: A_j * B_{j,:} but B_{j,:} is a row vector...
    Actually, C is a 1x7 vector. Each term t produces a scalar, and we need to scale C.
    Term j: U has a1..a4, V has b1..b4 (but we only use one row of B).
    Hmm, this needs careful handling.
    
    For 1xm * mxp -> 1xp:
    The output is 1xp, so W is 1xp (flattened to p entries).
    Each rank-1 term: u (m-dim, but A is 1xm so m entries) * v (p-dim for the column of B? No...)
    
    Actually, let's think of it differently.
    A is 1x4, so it has 4 entries: a1,a2,a3,a4.
    B is 4x7, so it has 28 entries: b_{i,k} for i in 0..3, k in 0..6.
    C is 1x7, so 7 entries: c1..c7.
    
    C_k = sum_{i=0}^{3} a_i * b_{i,k}
    
    Each term t: U[*,t] * V[*,t] * W[*,t]
    U has 4 rows (for a1..a4), V has 28 rows (for B flattened), W has 7 rows (for C1..C7).
    
    For term t corresponding to index i (i in 0..3):
    U[i,t] = 1
    V[i*7+k, t] = 1 for all k in 0..6
    W[k, t] = 1 for all k in 0..6
    Wait, that would give C_k += a_i * (sum of b_{i,k'} over k') * ... no that's wrong.
    
    Actually, each rank-1 term produces: for each output position r, C[r] += U[a_idx] * V[b_idx] * W[r]
    So if we set U[i,t]=1, V[i*7+k,t]=1 (for the specific k we want), and W[k,t]=1... 
    That's not right either because V is one vector per term.
    
    Let me reconsider. A rank-1 tensor term is:
    For all (a,b,c) indices: T[a,b,c] += u_a * v_b * w_c
    
    So if I want T[i, i*7+k, k] = 1 (the multiplication tensor entry), I need:
    u_i * v_{i*7+k} * w_k = 1
    
    One way: set u_i=1, v_{i*7+k}=1, w_k=1 for a single (i,k). But that's schoolbook with 28 terms.
    
    Better: use the fact that for fixed i, we can have v_{i*7+k}=1 for all k, and w_k=1 for all k.
    Then for term i: u_i=1, v_{i*7+k}=1 for all k, w_k=1 for all k.
    This gives T[i, i*7+k, k] += 1*1*1 = 1 for all k. 
    So 4 terms give us all 28 entries!
    
    U: 4 rows, 4 cols. U[i,i]=1.
    V: 28 rows, 4 cols. V[i*7+k, i] = 1 for all i,k.
    W: 7 rows, 4 cols. W[k, i] = 1 for all i,k.
    """
    rank = m
    U = np.zeros((m, rank))  # A is 1xm, so m entries
    V = np.zeros((m*p, rank))  # B is mxp, so mp entries
    W = np.zeros((p, rank))  # C is 1xp, so p entries
    
    for i in range(m):
        U[i, i] = 1
        for k in range(p):
            V[i*p + k, i] = 1
            W[k, i] = 1
    
    return U, V, W

def _pad_1xm_to_nxm(U1, n, m):
    """Take U1 for 1xm (shape m x rank) and extend to nxm (shape nm x rank).
    For row i of A, use U1. Other rows are 0.
    But wait, each row of A is independent, so we can't just reuse.
    Actually for 2x4x7, we have two rows of A. Each row's multiplication is independent.
    So we need separate decompositions for each row, or a combined one.
    """
    pass

def _build_2x4x7():
    """Build a rank decomposition for 2x4 x 4x7 -> 2x7.
    
    A is 2x4 (8 entries), B is 4x7 (28 entries), C is 2x7 (14 entries).
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..3, k in 0..6.
    
    Strategy: For each row i of A, we have a 1x4 * 4x7 problem.
    Rank 4 for each, total rank 8.
    But can we do better by sharing between the two rows?
    
    For now, let's use rank 8 (4 per row).
    """
    n, m, p = 2, 4, 7
    rank = 8
    U = np.zeros((n*m, rank))  # 8 rows
    V = np.zeros((m*p, rank))  # 28 rows
    W = np.zeros((p*n, rank))  # 14 rows
    
    # Row 0 of A: entries at indices 0..3 (i=0, j=0..3)
    # Row 1 of A: entries at indices 4..7 (i=1, j=0..3)
    # C row 0: indices 0..6 (k=0..6, i=0)
    # C row 1: indices 7..13 (k=0..6, i=1)
    
    for i in range(n):
        offset = i * rank  # offset in rank dimension
        for j in range(m):
            U[i*m + j, offset + j] = 1
        for j in range(m):
            for k in range(p):
                V[j*p + k, offset + j] = 1
        for k in range(p):
            W[k*n + i, offset + j] = 1  # Wait, j here is wrong
    
    # Let me redo this properly
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        base = i * rank
        for j in range(m):
            U[i*m + j, base + j] = 1
            for k in range(p):
                V[j*p + k, base + j] = 1
            for k in range(p):
                W[k*n + i, base + j] = 1
    
    # Verify: C[i,k] = sum_t W[k*n+i, t] * sum_{a} U[a,t] * sum_{b} V[b,t]
    # For term t = base + j:
    # U[i*m+j, t] = 1 (only if the A index matches)
    # V[j'*p+k', t] = 1 for all j', k'... wait that's wrong.
    
    # Let me reconsider. The tensor is T[a,b,c] where:
    # a = i*m + j (A index)
    # b = j'*p + k (B index) -- but j and j' must match for the product
    # c = k*n + i (C index)
    # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j,k), c corresponds to (i,k) and j matches.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    # For rank-1 term: sum over a,b,c: u_a * v_b * w_c
    # We want this to equal 1 at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    
    # My construction:
    # Term t = base + j where base = i*rank, j in 0..m-1
    # u_{i*m+j'} = 1 if j'==j, else 0. So only A[i,j] is selected.
    # v_{j''*p+k'} = 1 for all j'', k'. So all of B is selected.
    # w_{k''*n+i''} = 1 if i''==i, else 0. And for all k''.
    
    # So the contribution to C[i'',k''] from this term:
    # = w[k''*n+i'', t] * (sum_a u[a,t]*A[a]) * (sum_b v[b,t]*B[b])
    # But this isn't how it works. The bilinear form is:
    # C[c] += w_c * (sum_a u_a A[a]) * (sum_b v_b B[b])
    
    # Hmm, no. The standard tensor decomposition for matrix multiplication is:
    # C_{ik} = sum_t (sum_j U_{ij,t} A_{ij}) * (sum_j V_{jk,t} B_{jk}) * W_{ik,t}
    # where U, V, W are rank-r tensors, and the indices are grouped.
    
    # Actually the cleaner way: the multiplication tensor T has entries:
    # T_{(i,j),(j,k),(i,k)} = 1
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    # T_{a,b,c} = sum_t u_t[a] * v_t[b] * w_t[c]
    # with a in [nm], b in [mp], c in [np].
    
    # So for my term t (corresponding to row i, col j of A):
    # u_t[a] = 1 if a == i*m+j, else 0
    # v_t[b] = 1 for all b in [mp]
    # w_t[c] = 1 if c == k*n+i for some k, else 0
    # But w_t needs to be a single vector. If w_t[c] = 1 for all c in row i of C, then:
    # Contribution to C[i'', k''] = u_t[A[i'',j'']] * (sum_b v_t[b] B[b]) * w_t[C[i'',k'']]
    # = A[i,j] * (sum of all B entries) * 1 (if i''==i)
    # That's not right.
    
    # I think I'm overcomplicating. Let me use the standard form.
    
    # The matrix multiplication tensor for n x m times m x p is:
    # For each (i, j, k): the entry is A[i,j] * B[j,k] contributing to C[i,k].
    # In tensor form, T is a 3-way tensor with:
    # Mode 1: nm (A indices, indexed by (i,j))
    # Mode 2: mp (B indices, indexed by (j,k))  
    # Mode 3: np (C indices, indexed by (i,k))
    # T[i,j || j,k || i,k] = 1
    
    # A rank-1 tensor is u ⊗ v ⊗ w with:
    # (u ⊗ v ⊗ w)[a, b, c] = u[a] * v[b] * w[c]
    
    # So we need: sum_t u_t[a] * v_t[b] * w_t[c] = 1 if a=(i,j), b=(j,k), c=(i,k), else 0.
    
    # Schoolbook: for each (i,j,k): u_t[a]=1 if a=(i,j), v_t[b]=1 if b=(j,k), w_t[c]=1 if c=(i,k).
    
    # My 2-row decomposition attempt:
    # For row i of A, use m terms (one per j):
    # u_t[a] = 1 if a=(i,j), else 0
    # v_t[b] = 1 if b=(j,k) for ALL k... no, v_t[b] must be fixed.
    
    # The issue is that v_t[b] can only be 1 for specific b's. If I set v_t[b]=1 for all b in row j of B (i.e., b=(j,k) for all k), then:
    # The term contributes to all C[i,k] for fixed i, but weighted by sum of B[j,*].
    # That's not what we want.
    
    # So the simple sharing doesn't work. Each (i,j,k) needs its own term in general.
    # Unless we use Strassen or other algorithms.
    
    # Let's just do schoolbook: rank = n*m*p = 56.
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build schoolbook decomposition (rank 56)
    U, V, W = _build_2x4x7()
    
    # Try to reduce rank via linear dependence checking
    # If some columns of [U; V; W] are linearly dependent, we can remove them.
    # But since we built it with all 1s, there's no linear dependence.
    
    # Try a structured approach: use the 1xm decomposition for each row.
    # For 2x4x7: rank 4 per row = 8 total, if we can make it work.
    
    # Actually, let me try a direct construction with rank 8.
    rank_try = 8
    U2 = np.zeros((n*m, rank_try))
    V2 = np.zeros((m*p, rank_try))
    W2 = np.zeros((p*n, rank_try))
    
    for i in range(n):
        base = i * (m)  # base for this row
        for j in range(m):
            U2[i*m + j, base + j] = 1
            for k in range(p):
                V2[j*p + k, base + j] = 1
            for k in range(p):
                W2[k*n + i, base + j] = 1
    
    # Verify rank 8 decomposition
    # The term for (i, j): u selects A[i,j], v selects all of B[j,*], w selects all of C[i,*]
    # Contribution to C[i'', k'']:
    # Sum over t: W2[k''*n+i'', t] * (U2 dot A_vec) * (V2 dot B_vec)
    # This doesn't simplify to the correct form.
    
    # I think the rank 8 construction is invalid. Let me verify properly.
    # Use random test matrices.
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    # Flatten
    A_flat = A_test.flatten()  # n*m, row-major: A[0,0], A[0,1], ..., A[1,0], ...
    B_flat = B_test.flatten()  # m*p
    C_flat = C_expected.flatten()  # n*p: C[0,0], C[0,1], ..., C[1,0], ...
    # But the task says W[k*n+i, r], so C index is k*n + i (column-major in a sense)
    # C_flat should be C[k*n+i] = C_expected[i, k]
    C_flat_task = np.zeros(p * n)
    for k in range(p):
        for i in range(n):
            C_flat_task[k * n + i] = C_expected[i, k]
    
    # Compute C from decomposition
    # C_flat[r] = sum_t W[r, t] * (sum_a U[a,t] * A_flat[a]) * (sum_b V[b,t] * B_flat[b])
    def compute_C(U, V, W, A_flat, B_flat, rank):
        C = np.zeros(W.shape[0])
        for t in range(rank):
            u_dot = np.dot(U[:, t], A_flat)
            v_dot = np.dot(V[:, t], B_flat)
            C += W[:, t] * u_dot * v_dot
        return C
    
    # Check schoolbook
    C_schoolbook = compute_C(U, V, W, A_flat, B_flat, U.shape[1])
    if np.allclose(C_schoolbook, C_flat_task, atol=1e-10):
        schoolbook_ok = True
    else:
        schoolbook_ok = False
    
    # Check rank 8
    C_rank8 = compute_C(U2, V2, W2, A_flat, B_flat, U2.shape[1])
    rank8_ok = np.allclose(C_rank8, C_flat_task, atol=1e-10)
    
    if rank8_ok:
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
