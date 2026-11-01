"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank decomposition for 2x2 * 2x15 matrix multiplication.
    
    We decompose the 2x15 multiplication into blocks.
    The tensor is T_{i,j,k} where i in [0,1] (row of A), j in [0,1] (col of A / row of B),
    k in [0,14] (col of B).
    Output C[i, k] = sum_j A[i,j] * B[j,k].
    
    Indexing convention:
    U: shape (n*m, rank) = (4, rank), indexed by [i*m+j, c]
    V: shape (m*p, rank) = (30, rank), indexed by [j*p+k, c]
    W: shape (p*n, rank) = (30, rank), indexed by [k*n+i, c]
    
    The tensor entry T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    In the standard decomposition, for each (i,j,k) triple we have a rank-1 term:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    
    The total schoolbook rank is n*m*p = 2*2*15 = 60.
    
    To reduce rank, we can use Strassen-like ideas or exploit structure.
    
    For 2x2 * 2xp multiplication, we can think of it as 2 separate 1x2 * 2xp multiplications
    but that doesn't help directly.
    
    Actually, let's think about it differently. The multiplication is:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    For each column k, we're computing a 2x1 vector from a 2x2 matrix and a 2x1 vector.
    This is essentially 2 independent dot products of length 2.
    
    The key insight: for each column k, the computation is:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    We can factor out the A matrix. Let's group by A indices.
    
    Actually, the standard approach for matrix multiplication rank reduction uses the fact that
    we can share computation across columns.
    
    For 2x2 * 2xp, the tensor has rank at most 2*2*p = 4p if we do it naively per column.
    But we can do better by sharing.
    
    Let me think about this more carefully. The tensor T is in R^{2} (i) x R^{2} (j) x R^{15} (k).
    T[i,j,k] = delta_{i,j} ... no wait.
    
    Actually T[i, (j,k)] = 1 if the product A[i,j]*B[j,k] contributes to output C[i,k].
    So T[i, j, k] = 1 for all i, j, k where the term exists.
    
    Wait, the tensor for matrix multiplication M_{n,m,p} has entries:
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    and T[i,j,k] = 1 if A[i,j] and B[j,k] multiply to give C[i,k].
    
    So for our case, T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,14}.
    That's 2*2*15 = 60 entries all equal to 1.
    
    The rank of this tensor is what we want to minimize.
    
    For the standard schoolbook, rank = 60.
    
    Can we do better? The tensor is essentially the all-ones tensor of size 2x2x15.
    The rank of the all-ones tensor of size a x b x c is... well, it's not simply 1 because
    the rank-1 tensors are outer products of vectors.
    
    Actually, the all-ones tensor 1_{a,b,c} can be written as sum over a basis. The rank of the
    all-ones tensor is min(a*b, a*c, b*c) in some cases? No, that's not right either.
    
    Let me think again. The all-ones tensor J of size a x b x c has J[i,j,k] = 1 for all i,j,k.
    We want to write J = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r.
    
    One simple decomposition: J[i,j,k] = sum_{t=1}^{a} delta_{i,t} * sum_{s=1}^{b} delta_{j,s} * 1_k
    No, that gives rank a*b.
    
    Actually: J[i,j,k] = (1,1,...,1)_i ⊗ (1,1,...,1)_j ⊗ (1,1,...,1)_k is NOT rank 1 because
    the outer product of three vectors u ⊗ v ⊗ w has entries u[i]*v[j]*w[k], which is 1 only
    if all are 1, but then it's the all-ones tensor. Wait, u = (1,1), v = (1,1), w = (1,1,...,1)
    gives u[i]*v[j]*w[k] = 1*1*1 = 1. So the all-ones tensor IS rank 1!
    
    Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    Oh I see my confusion. In matrix multiplication, the tensor is NOT the all-ones tensor.
    The tensor T for matrix multiplication M_{n,m,p} is defined such that:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    So T[i,j,k] = 1 for all valid i,j,k. Yes, it IS the all-ones tensor!
    
    But wait, if T is the all-ones tensor, then its rank is 1, which would mean we can compute
    any matrix multiplication with rank 1, which is clearly false.
    
    I think I'm confusing the tensor with the bilinear form. Let me be more careful.
    
    The matrix multiplication tensor is T ∈ F^{n*m} ⊗ F^{m*p} ⊗ F^{p*n} where
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    where e_{i,j} is the basis vector in F^{n*m} corresponding to (i,j), etc.
    
    So T is NOT the all-ones tensor in the flattened sense. It's a specific tensor where
    T[i*m+j, j*p+k, k*n+i] = 1 and T[.,.,.] = 0 otherwise.
    
    So the tensor has 60 non-zero entries, each equal to 1, at specific positions.
    
    The rank of this tensor is what we're minimizing.
    
    For the schoolbook, we use 60 rank-1 terms, one for each (i,j,k).
    
    To reduce rank, we need to find a smaller set of rank-1 tensors that sum to T.
    
    For 2x2 * 2xp, let's see if we can do better than 4p.
    
    Actually, for 2x2 * 2x2, the rank is 7 (Strassen). For 2x2 * 2xp, we might be able to
    use a similar idea.
    
    Let me think about 2x2 * 2x2 first. The tensor has 8 non-zero entries:
    T[0*2+0, 0*2+0, 0*2+0] = 1  (i=0,j=0,k=0)
    T[0*2+0, 0*2+1, 1*2+0] = 1  (i=0,j=0,k=1)
    T[0*2+1, 0*2+0, 0*2+0] = 1  (i=0,j=1,k=0)
    T[0*2+1, 0*2+1, 1*2+0] = 1  (i=0,j=1,k=1)
    T[1*2+0, 0*2+0, 0*2+0] = 1  (i=1,j=0,k=0)
    T[1*2+0, 0*2+1, 1*2+0] = 1  (i=1,j=0,k=1)
    T[1*2+1, 0*2+0, 0*2+0] = 1  (i=1,j=1,k=0)
    T[1*2+1, 0*2+1, 1*2+0] = 1  (i=1,j=1,k=1)
    
    Strassen's 7-term algorithm for 2x2 * 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M4
    C10 = M2 + M6
    C11 = M1 - M3 + M6 - M7
    
    So the rank is 7, not 8.
    
    For 2x2 * 2xp, we can potentially use a similar approach. But the problem is that the
    B matrix has p columns, so we need to handle each column.
    
    One approach: group the 15 columns into groups and use Strassen-like decompositions.
    
    Actually, let's think about it as: for each pair of columns, we can use a 2x2 * 2x2 decomposition
    of rank 7. If we group 15 columns into 7 pairs (14 columns) plus 1 single column, we get
    7*7 + 4 = 49 + 4 = 53. That's better than 60.
    
    But wait, the single column (2x2 * 2x1) has rank 4 (since it's just 2 dot products of length 2,
    and each dot product of length 2 has rank 2, so total rank 4).
    
    So 7*7 + 4 = 53.
    
    Can we do better? What if we group into 3 columns at a time? 2x2 * 2x3.
    The rank of 2x2 * 2x3 is at most 2*2*3 = 12, but can we do better?
    
    Actually, I'm not sure about the exact rank of 2x2 * 2x3. Let me just use the pairing approach.
    
    Let me implement the 7-pair + 1-single approach.
    
    For a pair of columns (k, k+1), the 2x2 * 2x2 multiplication uses 7 terms.
    For the single column, it uses 4 terms.
    
    Total: 7*7 + 4 = 53.
    
    Let me code this up.
    
    Actually, I realize I should double-check the indexing.
    
    For the 2x2 * 2x2 block handling columns k and k+1:
    The Strassen algorithm gives 7 intermediate values M1..M7.
    Each M_r is a product of a linear combination of A entries and a linear combination of B entries.
    
    Let me write out the Strassen terms for 2x2 * 2x2:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M4
    C10 = M2 + M6
    C11 = M1 - M3 + M6 - M7
    
    For the decomposition, each M_r corresponds to a rank-1 term.
    M_r = (linear combo of A) * (linear combo of B)
    
    For M1: A_part = A00 + A11, B_part = B00 + B11
    This means: U has 1 at positions corresponding to A00 and A11, V has 1 at positions for B00 and B11.
    
    Let me define the indices:
    A indices: (i,j) -> i*2+j
    B indices: (j,k) -> j*2+k
    C indices: (k,i) -> k*2+i
    
    For M1: A00+A11 means U[0*2+0] += 1, U[1*2+1] += 1
    B00+B11 means V[0*2+0] += 1, V[1*2+1] += 1
    C00 and C11 get contributions: C00 = M1 - M4 + M5 + M7, C11 = M1 - M3 + M6 - M7
    
    So W[0*2+0] += 1 (for C00), W[1*2+1] += 1 (for C11)
    
    This is getting complex. Let me just implement it carefully.
    
    Actually, let me just use the schoolbook for now and see if I can improve it.
    The current score is 1.0 with rank 60. If I can get to 53, that's an improvement.
    
    Let me implement the Strassen-based decomposition for pairs of columns.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We'll use Strassen for pairs of columns, and schoolbook for the remaining column.
    # 15 columns = 7 pairs + 1 single
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are lists of (index, coeff)."""
        nonlocal rank
        U_cols.append([(idx, coeff) for idx, coeff in u_entries])
        V_cols.append([(idx, coeff) for idx, coeff in v_entries])
        W_cols.append([(idx, coeff) for idx, coeff in w_entries])
        rank += 1
    
    # Strassen 2x2 * 2x2 for a pair of columns (k, k+1)
    def strassen_pair(k1, k2):
        """Handle columns k1 and k2 using Strassen's 7-term algorithm."""
        # M1 = (A00+A11)(B00+B11)
        add_term(
            [(0*2+0, 1), (1*2+1, 1)],  # A00, A11
            [(0*2+k1, 1), (1*2+k2, 1)],  # B0,k1 and B1,k2
            [(k1*2+0, 1), (k2*2+1, 1)]   # C0,k1 and C1,k2
        )
        # M2 = (A10+A11)B00
        add_term(
            [(1*2+0, 1), (1*2+1, 1)],  # A10, A11
            [(0*2+k1, 1)],  # B0,k1
            [(k1*2+1, 1)]   # C1,k1
        )
        # M3 = A00(B10-B11)
        add_term(
            [(0*2+0, 1)],  # A00
            [(1*2+k1, 1), (1*2+k2, -1)],  # B1,k1 - B1,k2
            [(k2*2+0, 1)]   # C0,k2
        )
        # M4 = A11(B01-B00)
        add_term(
            [(1*2+1, 1)],  # A11
            [(0*2+k2, 1), (0*2+k1, -1)],  # B0,k2 - B0,k1
            [(k2*2+0, 1), (k1*2+0, -1)]   # C0,k2 - C0,k1
        )
        # M5 = (A00+A01)B11
        add_term(
            [(0*2+0, 1), (0*2+1, 1)],  # A00, A01
            [(1*2+k2, 1)],  # B1,k2
            [(k2*2+0, 1)]   # C0,k2
        )
        # M6 = (A10-A00)(B00+B10)
        add_term(
            [(1*2+0, 1), (0*2+0, -1)],  # A10 - A00
            [(0*2+k1, 1), (1*2+k1, 1)],  # B0,k1 + B1,k1
            [(k1*2+1, 1)]   # C1,k1
        )
        # M7 = (A01-A11)(B01+B11)
        add_term(
            [(0*2+1, 1), (1*2+1, -1)],  # A01 - A11
            [(0*2+k2, 1), (1*2+k2, 1)],  # B0,k2 + B1,k2
            [(k2*2+1, 1), (k2*2+0, -1)]   # C1,k2 - C0,k2
        )
    
    # Schoolbook for a single column k
    def schoolbook_single(k):
        for i in range(n):
            for j in range(m):
                add_term(
                    [(i*m+j, 1)],  # A[i,j]
                    [(j*p+k, 1)],  # B[j,k]
                    [(k*n+i, 1)]   # C[i,k]
                )
    
    # Process 7 pairs and 1 single
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        strassen_pair(k1, k2)
    
    # Last column
    schoolbook_single(14)
    
    # Now build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, coeff in U_cols[c]:
            U[idx][c] = coeff
        for idx, coeff in V_cols[c]:
            V[idx][c] = coeff
        for idx, coeff in W_cols[c]:
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
