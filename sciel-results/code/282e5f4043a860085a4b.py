import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140

    # --- Build the known rank-33 decomposition as a starting point ---
    # Structure: schoolbook on A[:,0]*B[0,:] (12 terms) + 3 Strassen 2x2x2 blocks (21 terms)
    rank33 = 33
    U33 = np.zeros((n * m, rank33))
    V33 = np.zeros((m * p, rank33))
    W33 = np.zeros((p * n, rank33))
    col = 0
    for k in range(p):
        for i in range(n):
            U33[i * m, col] = 1.0
            V33[k, col] = 1.0
            W33[k * n + i, col] = 1.0
            col += 1
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        # M1
        U33[m+1, col] += 1; U33[m+2, col] += 1
        V33[p+k0, col] += 1; V33[2*p+k1, col] += 1
        W33[k0*n, col] += 1; W33[k1*n+1, col] += 1
        col += 1
        # M2
        U33[m+1, col] += 1; U33[m+2, col] += 1
        V33[p+k0, col] += 1
        W33[k0*n+1, col] += 1; W33[k1*n+1, col] += -1
        col += 1
        # M3
        U33[1, col] += 1
        V33[p+k1, col] += 1; V33[2*p+k1, col] += -1
        W33[k1*n, col] += 1; W33[k1*n+1, col] += 1
        col += 1
        # M4
        U33[m+2, col] += 1
        V33[2*p+k0, col] += 1; V33[p+k0, col] += -1
        W33[k0*n, col] += 1; W33[k0*n+1, col] += 1
        col += 1
        # M5
        U33[1, col] += 1; U33[2, col] += 1
        V33[2*p+k1, col] += 1
        W33[k0*n, col] += -1; W33[k1*n, col] += 1
        col += 1
        # M6
        U33[m+1, col] += 1; U33[1, col] += -1
        V33[p+k0, col] += 1; V33[p+k1, col] += 1
        W33[k1*n+1, col] += 1
        col += 1
        # M7
        U33[2, col] += 1; U33[m+2, col] += -1
        V33[2*p+k0, col] += 1; V33[2*p+k1, col] += 1
        W33[k0*n, col] += 1
        col += 1

    def verify(U, V, W):
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U33, V33, W33):
        # Fallback schoolbook
        rs = n*m*p
        Us = np.zeros((n*m, rs)); Vs = np.zeros((m*p, rs)); Ws = np.zeros((p*n, rs))
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    Us[ii*m+jj, c] = 1; Vs[jj*p+kk, c] = 1; Ws[kk*n+ii, c] = 1
                    c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    best_U, best_V, best_W, best_rank = U33.copy(), V33.copy(), W33.copy(), rank33

    # --- Attempt rank reduction via column elimination ---
    # Strategy: try to find pairs of columns that can be merged, or a column that can be
    # expressed as a linear combination of others (zero rank contribution).
    # Use a greedy approach: for each pair (a,b), check if removing them and adjusting
    # others preserves the tensor identity.

    # More practically: try all pairs of columns and check if their sum/difference
    # can be absorbed. We use a linear-algebra test on the coefficient space.
    #
    # Tensor equation: for all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # We have 36 such equations and rank*3 variables per column.
    #
    # Simplest reduction test: can column r be written as a linear combination of
    # other columns such that the tensor equation still holds? This is generally
    # hard. Instead, we try: for each pair (a, b), remove both columns and see if
    # the remaining columns can be scaled/shifted to compensate. This is a
    # constrained optimization over the remaining rank-31 columns.
    #
    # Given time budget, we do a fast randomized search for rank-32:
    # Try to find a column r that is "redundant" — i.e., there exists a vector
    # c such that adding c_r to column r and adjusting other columns preserves
    # the identity. We check if the column space has dimension < rank.

    # Check column space rank of each factor
    for attempt in range(200):
        if time.time() > deadline - 10:
            break

        # Random perturbation: try to find a linear dependence among columns
        # by checking if a random linear combination of columns gives zero in
        # the "tensor product" sense.
        r = np.random.randint(rank33)
        # Check if column r can be removed: solve for coefficients in other columns
        # such that the tensor contribution of column r is matched.
        # This is a system of 36 equations in 2*rank33 unknowns (too many).
        # Instead, check simple case: is column r identically zero in some factor?
        # (Already checked by construction — no zeros in our 33-column construction.)

        # Try pairwise column merging via sign flip:
        # If U[:,a] = -U[:,b], V[:,a] = V[:,b], W[:,a] = -W[:,b], then
        # column a contributes the same as column b with a sign, so we can
        # combine them if the signs work out.
        # Generalize: check if there's a sign pattern that makes two columns
        # contribute oppositely, allowing cancellation.
        for a in range(rank33):
            for b in range(a+1, rank33):
                # Check if U[:,a] = -s*U[:,b] for some s in {1,-1}
                ua, ub = U33[:, a], U33[:, b]
                va, vb = V33[:, a], V33[:, b]
                wa, wb = W33[:, a], W33[:, b]

                # Try: can we set U_new = U - U_a*e_a - U_b*e_b and adjust?
                # Simple test: is (U_a * V_a * W_a) = (U_b * V_b * W_b) as tensors?
                # If so, they contribute the same and one can be removed.
                # Tensor of column r: T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
                ta = U33[:, a].reshape(n, m) * V33[:, a].reshape(m, p)  # not directly
                # Actually: T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
                # For two columns to be "same tensor", we need
                # U_a[i,j]*V_a[j,k]*W_a[k,i] = U_b[i,j]*V_b[j,k]*W_b[k,i] for all i,j,k
                # This is a strong condition. Let's just check a few (i,j,k) tuples.
                same = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ta_val = U33[i*m+j, a] * V33[j*p+k, a] * W33[k*n+i, a]
                            tb_val = U33[i*m+j, b] * V33[j*p+k, b] * W33[k*n+i, b]
                            if abs(ta_val - tb_val) > 1e-12:
                                same = False
                                break
                        if not same:
                            break
                    if not same:
                        break
                if same:
                    # Remove column b (keep a)
                    new_rank = rank33 - 1
                    U_new = np.delete(U33, b, axis=1)
                    V_new = np.delete(V33, b, axis=1)
                    W_new = np.delete(W33, b, axis=1)
                    if verify(U_new, V_new, W_new):
                        if new_rank < best_rank:
                            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank
                        break
            if best_rank < rank33:
                break

    # --- Try rank-32 via a specific structured reduction ---
    # The 12 schoolbook terms for A[:,0]*B[0,:] might be reducible.
    # A[:,0] is a 2x1 vector, B[0,:] is a 1x6 vector. Their outer product
    # in the tensor sense is 2*6 = 12 rank-1 terms. The minimal rank of the
    # 2x1x6 tensor (outer product of a 2-vector and a 6-vector) is 2+6-1=7
    # by the Winograd result for rectangular multiplication.
    # So we can potentially replace the 12 schoolbook terms with 7 terms!
    # This would give rank 7 + 21 = 28.

    # Winograd for 2x1x6: multiply a (2x1) matrix by (1x6) matrix.
    # A = [a0; a1], B = [b0 b1 b2 b3 b4 b5]
    # C = [a0*b0, a0*b1, a0*b2, a0*b3, a0*b4, a0*b5;
    #      a1*b0, a1*b1, a1*b2, a1*b3, a1*b4, a1*b5]
    # This is the outer product of a (2,) vector and a (6,) vector.
    # Tensor rank of outer product of sizes r, s is r+s-1.
    # For r=2, s=6: rank = 7.

    # Winograd construction for 2x6 outer product:
    # Let u0 = a0, u1 = a0 + a1
    # Let v0 = b0, v1 = b0 + b1, v2 = b0 + b1 + b2, ..., v5 = b0+...+b5
    # Then C[i,k] = sum of appropriate differences.
    # Standard Winograd for 1xN by N: uses N-1+1 = N terms.
    # For 2x6: use 2+6-1 = 7 terms.

    # Construct:
    # U factors (2 values: a0, a1):
    #   r=0: a0
    #   r=1: a0 + a1
    #   r=2..6: need to cover the 6 b's with 5 more terms
    # Actually, the Winograd scheme for (r+1) x (s+1) outer product uses r+s+1 terms.
    # For 2 x 6: 2+6-1 = 7 terms.
    # Let's use the standard FTA-based approach:
    # A' = [a0, a1] -> DFT-like: [a0+a1, a0-a1] (2 points)
    # B' = [b0..b5] -> 6-point "cumulative" transform
    # Product in transformed domain, then inverse.
    # But 2 and 6 are not the same size, so we need a different approach.

    # Simpler: explicit rank-7 decomposition of 2x1x6 outer product.
    # C[i,k] = a_i * b_k
    # Use:
    #   t0 = a0*b0
    #   t1 = (a0+a1)*b0
    #   t2 = a0*b1
    #   t3 = (a0+a1)*b1
    #   ... this gives 2*6=12, not 7.

    # The rank of the tensor T[i,k] = a_i * b_k (i=0..1, k=0..5) is indeed 7.
    # Decomposition:
    # Let s_j = b_0 + b_1 + ... + b_j for j=0..5 (6 partial sums)
    # Let d_j = s_j - s_{j-1} = b_j (differences)
    # C[i,k] = a_i * b_k = a_i * (s_k - s_{k-1})
    # This doesn't directly give rank-7.

    # Correct rank-(r+s-1) decomposition:
    # For a (r+1)-vector x and (s+1)-vector y, outer product rank = r+s+1.
    # Wait, I had it backwards. The rank of the outer product tensor
    # T[i,k] = x_i * y_k in R^{(r+1) x (s+1)} as a 2-tensor (matrix) is min(r+1,s+1).
    # But as a 3-tensor T[i,j,k] where j is the shared index... there IS no shared index
    # in a pure outer product. The tensor here is T[i,k] = a_i * b_k, which is a
    # 2-dimensional tensor (matrix), and its matrix rank is min(2,6)=2.
    #
    # But in our problem, the tensor is T[i,j,k] = A[i,j]*B[j,k]. For j=0 only:
    # T[i,0,k] = A[i,0]*B[0,k]. This is a 3-tensor with indices (i,0,k).
    # The "rank" as a 3-tensor is the matrix rank of the unfolding, which is 2.
    # But in the Strassen sense (multiplicative complexity), we need rank-1 terms
    # of the form U[i,j,r]*V[j,k,r]*W[k,i,r].
    # For the j=0 slice: U[i,0,r]*V[0,k,r]*W[k,i,r] must sum to A[i,0]*B[0,k].
    # This is the outer product of a 2-vector and a 6-vector, implemented as
    # a sum of rank-1 tensors in 3D. The multiplicative complexity of
    # multiplying a (2x1) by (1x6) matrix is 2*6 = 12 (no shortcuts since
    # there's no structure to exploit — each a_i * b_k is independent).
    #
    # So the 12 terms are necessary for the j=0 slice. The rank-33 is likely optimal
    # for this block structure.

    # --- Try a completely different block structure ---
    # What if we use Winograd on the j-index instead?
    # The tensor is T[i,j,k] = A[i,j]*B[j,k], i=0..1, j=0..2, k=0..5.
    # Apply a 3-point transform in j:
    # j' = 0: A'[i,0] = A[i,0]+A[i,1]+A[i,2], B'[0,k] = B[0,k]
    # j' = 1: A'[i,1] = A[i,0]-A[i,1], B'[1,k] = B[1,k]-B[2,k]  (not standard)
    # This is getting complex. Given the time, stick with rank 33.

    # Final verification and return
    if best_rank < rank33:
        if verify(best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U33.tolist(), "V": V33.tolist(), "W": W33.tolist()}
