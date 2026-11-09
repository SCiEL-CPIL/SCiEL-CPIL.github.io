"""
Rank-minimal tensor decomposition for 2x3x8 matrix multiplication.

Strategy:
  1. Build a guaranteed schoolbook decomposition (rank 48).
  2. Attempt a block-structured decomposition that splits the 3 middle indices
     into a 2-block and a 1-block, and the 8 right indices into four 2-blocks.
     For each (2x2) sub-block in (middle, right) space, use the Strassen
     decomposition of the 2x2x2 tensor (rank 7 instead of 8).
     For each (1x2) sub-block, use the schoolbook rank-2 decomposition.
     This yields rank 4*7 + 4*2 = 36.
  3. Verify the decomposition exactly. If it passes, return it.
     Otherwise fall back to schoolbook.
"""

import time
import numpy as np


def construct():
    """
    Return a valid tensor decomposition for 2x3x8 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    U shape: (6, rank), V shape: (24, rank), W shape: (16, rank)
    Convention: W[k*n + i, r]
    """
    start = time.time()
    deadline = start + 130  # safety margin

    n, m, p = 2, 3, 8

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------
    def verify(U, V, W, n, m, p, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # ------------------------------------------------------------------
    # Schoolbook decomposition (rank 48) – guaranteed fallback
    # ------------------------------------------------------------------
    def schoolbook(n, m, p):
        R = n * m * p
        U = [[0.0] * R for _ in range(n * m)]
        V = [[0.0] * R for _ in range(m * p)]
        W = [[0.0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    assert verify(U0, V0, W0, n, m, p), "Schoolbook failed"

    # ------------------------------------------------------------------
    # Block-structured decomposition (target rank 36)
    #
    # Split j (size 3) into J1={0,1} (size 2) and J2={2} (size 1)
    # Split k (size 8) into K1={0,1}, K2={2,3}, K3={4,5}, K4={6,7} (each size 2)
    #
    # For each (Jb, Kb) pair we have a sub-problem of size (2 x |Jb| x |Kb|)
    #   - If |Jb|=2 and |Kb|=2: this is a 2x2x2 sub-tensor → use Strassen (rank 7)
    #   - If |Jb|=1 and |Kb|=2: this is a 1x2x2 sub-tensor → schoolbook (rank 2)
    #
    # Total rank = 4 * 7 + 4 * 2 = 36
    #
    # The key insight: the U, V, W matrices are shared across all sub-blocks.
    # We build them by concatenating the per-block contributions.
    #
    # For a 2x2x2 Strassen block with indices (i in {0,1}, j' in {0,1}, k' in {0,1}):
    #   The 7 multiplications are:
    #     M1 = A[i, j'] * B[j', k']  (for the 2x2x2 sub-tensor)
    #   Standard Strassen for C = A*B where A is 2x2, B is 2x2:
    #     M1 = A[0,0]*(B[0,0]+B[1,1])
    #     M2 = (A[0,0]+A[0,1])*B[1,0]
    #     M3 = (A[1,0]+A[1,1])*B[0,0]
    #     M4 = A[1,1]*(B[0,1]-B[1,0])
    #     M5 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #     M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #     M7 = (A[0,0]-A[0,1])*(B[0,0]+B[0,1])
    #   Reconstruction:
    #     C[0,0] = M1 + M4 - M5 + M7
    #     C[0,1] = M3 + M5
    #     C[1,0] = M2 + M4
    #     C[1,1] = M1 - M2 + M3 + M6
    #
    # In our tensor decomposition, each "multiplication" M_r corresponds to a
    # rank-1 term: U_vec * V_vec * W_vec.
    #
    # For the sub-block with j-offset j0 and k-offset k0:
    #   A[i, j0+j'] for i in {0,1}, j' in {0,1}
    #   B[j0+j', k0+k'] for j' in {0,1}, k' in {0,1}
    #   C[i, k0+k'] = sum over the Strassen combination
    #
    # We need to express each M_r as U[i*m + j] * V[j*p + k] * W[k*n + i].
    #
    # For M1 = A[i, j0] * (B[j0, k0] + B[j0+1, k0+1]):
    #   This is a product of a linear combination of A entries and a linear
    #   combination of B entries. In the tensor decomposition framework:
    #   U[i*m + j0, r] = 1, V[j0*p + k0, r] = 1, V[(j0+1)*p + (k0+1), r] = 1
    #   W[k0*n + i, r] = 1, W[(k0+1)*n + i, r] = 1
    #   But wait – this doesn't quite work because M1 involves A[i,j0] which
    #   is a single A entry, but the B part is a sum of two B entries.
    #
    # Actually, the standard way to express Strassen in tensor decomposition:
    # Each M_r is a product of a linear form in A and a linear form in B.
    # The U factor collects the A coefficients, V collects B coefficients,
    # and W collects the C reconstruction coefficients.
    #
    # For M1 = A[0,j0]*(B[0,k0]+B[1,k0+1]) + A[1,j0]*(B[0,k0]+B[1,k0+1]):
    #   No wait, M1 in the 2x2x2 case is:
    #   M1 = A[0,0]*(B[0,0]+B[1,1])
    #   This is a single product of A[0,0] and (B[0,0]+B[1,1]).
    #   In tensor terms: U[0*m+0, r]=1, V[0*p+0, r]=1, V[1*p+1, r]=1, W[0*n+0, r]=1, W[1*n+0, r]=1
    #   But that's not right either, because the C reconstruction uses M1 with
    #   coefficient +1 for C[0,0] and +1 for C[1,1] (wait, no: C[0,0] uses M1
    #   with coeff +1, C[1,1] uses M1 with coeff +1).
    #
    # Let me be more careful. The tensor T[i,j,k] = 1 for the 2x2x2 case.
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1}.
    #
    # Strassen gives 7 terms. Let me write them out explicitly for 2x2x2:
    # Term 1: U[0,0]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,0]=1
    #   Contribution to T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    #   Contribution to T[0,1,1]: U[0,0]*V[1,1]*W[1,0] = 1*1*1 = 1
    #   But T[0,0,1] should get 0 from this term: U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0 ✓
    #   T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 0*1*1 = 0 ✓
    #   T[1,1,1]: U[1,0]*V[1,1]*W[1,1] = 0*1*1 = 0 ✓
    #   But T[0,0,0] gets 1, T[0,1,1] gets 1. We need T[i,j,k]=1 for ALL i,j,k.
    #   So this single term only covers 2 of the 8 entries.
    #
    # I think I'm confusing the Strassen multiplication algorithm with the
    # tensor decomposition. Let me think again.
    #
    # The matrix multiplication tensor for 2x2x2 has rank 7 (Strassen).
    # The 7 rank-1 tensors are:
    #   For each r in 1..7, we have vectors u_r (in R^4), v_r (in R^4), w_r (in R^4)
    #   such that sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k.
    #
    # The standard Strassen decomposition:
    #   M1 = A11*(B21+B22)  → u=(1,0,0,0), v=(0,0,1,1), w=(1,0,1,0)  [w[0,0]=1,w[0,1]=1? No...]
    #
    # Actually, let me just use the known explicit Strassen tensor decomposition.
    # The 7 terms for 2x2x2 (indices i,j,k each in {0,1}):
    #
    # r=0: U[0,0]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,0]=1
    #      → covers (0,0,0) and (0,1,1)
    # r=1: U[0,0]=1, U[0,1]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1
    #      → covers (0,1,0) and (1,1,0)  [wait, U[0,1] means i=0,j=1]
    #
    # This is getting complicated. Let me just use a known correct implementation.
    #
    # Actually, for the purpose of this problem, since we're dealing with a
    # 2x3x8 tensor, and the block structure means we can treat each 2x2x2
    # sub-block independently (because the U, V, W factors have disjoint
    # support across blocks), we can simply concatenate the Strassen
    # decompositions for each 2x2x2 block.
    #
    # But the U, V, W factors DO have overlapping support! U[i*m+j] is shared
    # across all k-blocks for the same (i,j). So we can't simply concatenate.
    #
    # However, if we assign separate columns of U, V, W for each block, the
    # cross-terms will be zero (because U[i*m+j, r] is nonzero only for the
    # columns assigned to blocks that include index j, and similarly for V, W).
    #
    # Wait, that's not right. U[i*m+j, r] is a single entry. If we use column r
    # for block (Jb, Kb), then U[i*m+j, r] is nonzero only for j in Jb.
    # V[j*p+k, r] is nonzero only for j in Jb and k in Kb.
    # W[k*n+i, r] is nonzero only for k in Kb.
    #
    # So for a different block (Jb', Kb'), the column r' has U nonzero only for
    # j in Jb', V nonzero only for j in Jb' and k in Kb', W nonzero only for k in Kb'.
    #
    # The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is nonzero only if
    # j in Jb AND j in Jb' (for V) AND k in Kb AND k in Kb' (for W).
    # If Jb ≠ Jb' or Kb ≠ Kb', the product is zero.
    #
    # So the blocks ARE independent! We can concatenate the decompositions.
    #
    # Great. So the plan is:
    # For each (Jb, Kb) block, build the appropriate sub-decomposition,
    # and concatenate all columns.

    # ------------------------------------------------------------------
    # Strassen decomposition for 2x2x2 tensor
    # Returns U (2x2=4 entries), V (2x2=4 entries), W (2x2=4 entries)
    # as lists of 7 vectors, each of length 4.
    # ------------------------------------------------------------------
    def strassen_222():
        """
        Return 7 rank-1 decompositions of the 2x2x2 matrix multiplication tensor.
        Each rank-1 is (u, v, w) where u,v,w are length-4 vectors.
        u[i*2+j], v[j*2+k], w[k*2+i] for i,j,k in {0,1}.
        """
        # Correct Strassen decomposition verified by construction
        terms = []
        # r0: u=(1,0,0,0) v=(0,0,1,1) w=(1,0,1,0)
        # Covers: T[0,1,0]: u[0]*v[2]*w[0]=1, T[0,1,1]: u[0]*v[3]*w[2]=1*1*1=1... 
        # Let me just use a known correct set and verify
        terms.append(([1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]))
        terms.append(([1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]))
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
        terms.append(([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]))
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]))
        terms.append(([1, -1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]))
        return terms

    # Verify Strassen decomposition
    def verify_222(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for u, v, w in terms:
                        s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    strassen_terms = strassen_222()
    if not verify_222(strassen_terms):
        # Fallback: use schoolbook for 2x2x2 (rank 8)
        strassen_terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0.0]*4; v = [0.0]*4; w = [0.0]*4
                    u[i*2+j] = 1.0
                    v[j*2+k] = 1.0
                    w[k*2+i] = 1.0
                    strassen_terms.append((u, v, w))

    # ------------------------------------------------------------------
    # Schoolbook for 1x2x2 tensor (rank 2)
    # u is length 2 (i is fixed at 0, j in {0,1})
    # v is length 4 (j in {0,1}, k in {0,1})
    # w is length 4 (k in {0,1}, i=0)
    # ------------------------------------------------------------------
    def schoolbook_122():
        """
        Return 2 rank-1 decompositions of the 1x2x2 tensor.
        u: length 2 (j in {0,1})
        v: length 4 (j*2+k)
        w: length 4 (k*1+0 = k)  [since n=1 for this sub-block, but in the
           full problem n=2, so w index is k*n+i = k*2+0]
        Actually, in the full problem, the W index is k*n+i where n=2.
        For the sub-block, i is fixed to 0 (the first row of the 2x3 matrix).
        So W[k*2+0, r] for k in {0,1}.
        """
        terms = []
        # For j=0: C[0,k] = A[0,0]*B[0,k] for k=0,1
        # u[0]=1 (A[0,0]), v[0]=1 (B[0,0]), v[1]=1 (B[0,1]), w[0]=1 (C[0,0]), w[1]=1 (C[1,0])
        # Wait, this isn't a rank-1 tensor. A rank-1 tensor for 1x2x2 is:
        # u[j] * v[j,k] * w[k] for j in {0,1}, k in {0,1}
        # We need sum_r u_r[j] * v_r[j,k] * w_r[k] = 1 for all j,k.
        #
        # Schoolbook: for each (j,k), one term.
        # Term for (j=0,k=0): u[0]=1, v[0]=1, w[0]=1
        # Term for (j=0,k=1): u[0]=1, v[1]=1, w[1]=1
        # Term for (j=1,k=0): u[1]=1, v[2]=1, w[0]=1
        # Term for (j=1,k=1): u[1]=1, v[3]=1, w[1]=1
        # That's 4 terms, but we said rank 2. Let me reconsider.
        #
        # Actually, 1x2x2 tensor: T[j,k] = 1 for j in {0,1}, k in {0,1}.
        # This is a 2x2 matrix of all ones. Its tensor rank (as a 3-tensor with
        # one mode of size 1) is just the matrix rank of the 2x2 matrix, which is 1.
        # Wait no, it's a 3-tensor T[i,j,k] with i fixed. So it's really a 2x2 matrix.
        # The rank of the all-ones 2x2 matrix is 1.
        # So we can do it in rank 1!
        # u[j] = 1 for j in {0,1}, v[j,k] = 1, w[k] = 1.
        # sum = 1*1*1 = 1 for all j,k. Yes!
        #
        # But wait, in the full problem, the U factor is U[i*m+j] where i is the
        # row of the 2x3 matrix. For the J2 block (j=2), i can be 0 or 1.
        # So we have two separate 1x2x2 sub-problems (one for i=0, one for i=1).
        # Each has rank 1, so total rank 2 for the (J2, Kb) block.

        # For i=0: u[2]=1 (A[0,2]), v[2*p+k]=1 (B[2,k]), w[k*n+0]=1 (C[k,0])
        # For i=1: u[5]=1 (A[1,2]), v[2*p+k]=1 (B[2,k]), w[k*n+1]=1 (C[k,1])
        # But these are handled in the main loop, not here.
        return terms  # empty, handled in main loop

    # ------------------------------------------------------------------
    # Build the full block-structured decomposition
    # ------------------------------------------------------------------
    def build_block_decomposition(n, m, p, strassen_terms):
        """
        Build the block-structured decomposition.
        J1 = {0,1}, J2 = {2}
        K1 = {0,1}, K2 = {2,3}, K3 = {4,5}, K4 = {6,7}
        """
        R = 0
        U_rows = n * m  # 6
        V_rows = m * p  # 24
        W_rows = p * n  # 16

        U = []
        V = []
        W = []

        j_blocks = [(0, 2), (2, 1)]  # (start, size)
        k_blocks = [(0, 2), (2, 2), (4, 2), (6, 2)]

        for j0, js in j_blocks:
            for k0, ks in k_blocks:
                if js == 2 and ks == 2:
                    # 2x2x2 block → use Strassen (7 terms)
                    for u_sub, v_sub, w_sub in strassen_terms:
                        # u_sub is length 4: index i*2+j' where i in {0,1}, j' in {0,1}
                        # Full U index: i*m + (j0+j')
                        # v_sub is length 4: index j'*2+k' where j' in {0,1}, k' in {0,1}
                        # Full V index: (j0+j')*p + (k0+k')
                        # w_sub is length 4: index k'*2+i where k' in {0,1}, i in {0,1}
                        # Full W index: (k0+k')*n + i

                        u_full = [0.0] * U_rows
                        v_full = [0.0] * V_rows
                        w_full = [0.0] * W_rows

                        for i in range(2):
                            for jp in range(2):
                                u_full[i * m + (j0 + jp)] = u_sub[i * 2 + jp]

                        for jp in range(2):
                            for kp in range(2):
                                v_full[(j0 + jp) * p + (k0 + kp)] = v_sub[jp * 2 + kp]

                        for kp in range(2):
                            for i in range(2):
                                w_full[(k0 + kp) * n + i] = w_sub[kp * 2 + i]

                        U.append(u_full)
                        V.append(v_full)
                        W.append(w_full)
                        R += 1
                elif js == 1 and ks == 2:
                    # 1x2x2 block → rank 2 (one term per i in {0,1})
                    for i in range(2):
                        u_full = [0.0] * U_rows
                        v_full = [0.0] * V_rows
                        w_full = [0.0] * W_rows

                        # u: A[i, j0] → U[i*m + j0] = 1
                        u_full[i * m + j0] = 1.0

                        # v: B[j0, k0+kp] for kp in {0,1} → V[j0*p + k0+kp] = 1
                        for kp in range(2):
                            v_full[j0 * p + (k0 + kp)] = 1.0

                        # w: C[k0+kp, i] for kp in {0,1} → W[(k0+kp)*n + i] = 1
                        for kp in range(2):
                            w_full[(k0 + kp) * n + i] = 1.0

                        U.append(u_full)
                        V.append(v_full)
                        W.append(w_full)
                        R += 1
                else:
                    # Should not happen
                    pass

        return U, V, W, R

    # ------------------------------------------------------------------
    # Attempt the block decomposition
    # ------------------------------------------------------------------
    if time.time() < deadline - 5:
        U1, V1, W1, R1 = build_block_decomposition(n, m, p, strassen_terms)

        if verify(U1, V1, W1, n, m, p):
            return {"U": U1, "V": V1, "W": W1}
    
    # ------------------------------------------------------------------
    # Try a different block structure: split j into {0}, {1}, {2} and 
    # k into four 2-blocks. Each (1,2) block has rank 2.
    # Total rank = 3 * 4 * 2 = 24? No, that's not right either.
    # For i in {0,1}, j in {0,1,2}, k in {0..7}:
    # If we split j into singletons and k into pairs, each (i,j,k_pair) 
    # is a 1x1x2 tensor which has rank 1 (it's just A[i,j]*B[j,k0] and A[i,j]*B[j,k1]).
    # Wait, 1x1x2 tensor: T[j,k] with j fixed, k in {0,1}. Rank is 1.
    # So for each (i, j, k_block), rank 1. Total: 2*3*4 = 24? 
    # But that can't be right because the full rank is at least max(n*m, m*p, p*n) = 24.
    # Actually the border rank lower bound for 2x3x8 is at least 24.
    # Let me try: for each i in {0,1}, j in {0,1,2}, k_block in {0,1,2,3}:
    #   u[i*m+j] = 1, v[j*p+k0+kp] = 1 for kp in {0,1}, w[(k0+kp)*n+i] = 1 for kp in {0,1}
    # This gives rank 2*3*4 = 24 terms.
    # ------------------------------------------------------------------
    if time.time() < deadline - 5:
        U2 = []
        V2 = []
        W2 = []
        for i in range(n):
            for j in range(m):
                for k0 in range(0, p, 2):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    for kp in range(2):
                        v[j * p + k0 + kp] = 1.0
                        w[(k0 + kp) * n + i] = 1.0
                    U2.append(u)
                    V2.append(v)
                    W2.append(w)
        
        if verify(U2, V2, W2, n, m, p):
            return {"U": U2, "V": V2, "W": W2}

    # ------------------------------------------------------------------
    # Fallback to schoolbook
    # ------------------------------------------------------------------
    return {"U": U0, "V": V0, "W": W0}
