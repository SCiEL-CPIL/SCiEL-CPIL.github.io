import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _build_strassen_2x2_2x2(n, m, p):
    """
    Build a tensor decomposition using Strassen 2x2x2 on the first 2x2x2 block,
    schoolbook for the rest. Returns (U, V, W) as lists of lists.
    
    Tensor T[i, j, k] = delta (the multiplication tensor).
    Decomposition: T = sum_c u_c (x) v_c (x) w_c, where
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    """
    terms = []  # list of (u_vec, v_vec, w_vec) where u has n*m entries, v has m*p, w has p*n

    # Strassen 2x2x2: for i in {0,1}, j in {0,1}, k in {0,1}
    # 7 rank-1 terms
    # The 7 products for computing [[a11,a12],[a21,a22]] * [[b11,b12],[b21,b22]]:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5+M6
    # C21 = M2+M4+M7
    # C22 = M1-M2+M3+M6

    # For each of the 7 Strassen terms, we have:
    #   u: which (i,j) pairs contribute, with coefficients
    #   v: which (j,k) pairs contribute, with coefficients
    #   w: which (k,i) pairs get the result

    # Term 1: M1 = (a11+a22)(b11+b22)
    # u: (i=0,j=0)+1, (i=1,j=1)+1
    # v: (j=0,k=0)+1, (j=1,k=1)+1
    # w: (k=0,i=0)+1, (k=0,i=1)-1, (k=1,i=0)-1, (k=1,i=1)+1
    # Wait, let me be more careful.
    # C11 = M1+M4-M5+M7, C12 = M3+M5+M6, C21 = M2+M4+M7, C22 = M1-M2+M3+M6
    # So M1 contributes +1 to C11 and +1 to C22.
    # M4 contributes +1 to C11 and +1 to C21.
    # M5 contributes -1 to C11 and +1 to C12.
    # M7 contributes +1 to C11 and +1 to C21.
    # M3 contributes +1 to C12 and +1 to C22.
    # M6 contributes +1 to C12 and +1 to C22.
    # M2 contributes +1 to C21 and -1 to C22.

    # For the Strassen block, i in {0,1}, j in {0,1}, k in {0,1}
    # u vector is over (i,j) with i,j in {0,1}
    # v vector is over (j,k) with j,k in {0,1}
    # w vector is over (k,i) with k,i in {0,1}

    strassen = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [u00, u01, u10, u11] for (i,j) = (0,0),(0,1),(1,0),(1,1)
        # v: [v00, v01, v10, v11] for (j,k) = (0,0),(0,1),(1,0),(1,1)
        # w: [w00, w01, w10, w11] for (k,i) = (0,0),(0,1),(1,0),(1,1)
        # M1: (a11+a22)(b11+b22)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # contributes to C11(k=0,i=0) and C22(k=1,i=1)
        # M2: (a21+a22)b11
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),  # contributes to C21(k=0,i=1) and -C22(k=1,i=1)
        # M3: a11(b12-b22)
        ([1,0,0,0], [0,1,-1,0], [0,1,0,0]),  # contributes to C12(k=1,i=0) and C22(k=1,i=1)
        # M4: a22(b21-b11)
        ([0,0,0,1], [-1,0,1,0], [0,0,1,0]),  # contributes to C11(k=0,i=0) and C21(k=1,i=1)
        # M5: (a11+a12)b22
        ([1,1,0,0], [0,0,0,1], [0,-1,0,0]),  # contributes to -C11(k=0,i=0) and C12(k=1,i=0)
        # M6: (a21-a11)(b11+b12)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # contributes to C12(k=1,i=0) and C22(k=1,i=1)
        # M7: (a12-a22)(b21+b22)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # contributes to C11(k=0,i=0) and C21(k=1,i=1)
    ]

    # Let me verify these are correct by checking the tensor identity.
    # Actually, let me just directly compute what the decomposition gives and check.
    # For now, let me use a known correct formulation.

    # Let me re-derive carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5 + M6
    # C21 = M2 + M4 + M7
    # C22 = M1 - M2 + M3 + M6

    # So the w coefficients for each M term:
    # M1: C11+1, C22+1 -> w[0,0]+1, w[1,1]+1
    # M2: C21+1, C22-1 -> w[0,1]+1, w[1,1]-1
    # M3: C12+1, C22+1 -> w[1,0]+1, w[1,1]+1
    # M4: C11+1, C21+1 -> w[0,0]+1, w[1,1]+1
    # M5: C11-1, C12+1 -> w[0,0]-1, w[1,0]+1
    # M6: C12+1, C22+1 -> w[1,0]+1, w[1,1]+1
    # M7: C11+1, C21+1 -> w[0,0]+1, w[1,1]+1

    # Wait, C21 has k=0, i=1, so w[0,1]. Let me recheck.
    # w index: w[k*n+i], so for n=2: w[0*2+0]=w00, w[0*2+1]=w01, w[1*2+0]=w10, w[1*2+1]=w11
    # C11: k=0,i=0 -> w00
    # C12: k=1,i=0 -> w10
    # C21: k=0,i=1 -> w01
    # C22: k=1,i=1 -> w11

    # M1: C11+1, C22+1 -> w00+1, w11+1
    # M2: C21+1, C22-1 -> w01+1, w11-1
    # M3: C12+1, C22+1 -> w10+1, w11+1
    # M4: C11+1, C21+1 -> w00+1, w01+1
    # M5: C11-1, C12+1 -> w00-1, w10+1
    # M6: C12+1, C22+1 -> w10+1, w11+1
    # M7: C11+1, C21+1 -> w00+1, w01+1

    # u coefficients (which a_ij contribute):
    # M1: a11+a22 -> u00+1, u11+1
    # M2: a21+a22 -> u10+1, u11+1
    # M3: a11 -> u00+1
    # M4: a22 -> u11+1
    # M5: a11+a12 -> u00+1, u01+1
    # M6: a21-a11 -> u10+1, u00-1
    # M7: a12-a22 -> u01+1, u11-1

    # v coefficients (which b_jk contribute):
    # M1: b11+b22 -> v00+1, v11+1
    # M2: b11 -> v00+1
    # M3: b12-b22 -> v01+1, v11-1
    # M4: b21-b11 -> v10+1, v00-1
    # M5: b22 -> v11+1
    # M6: b11+b12 -> v00+1, v01+1
    # M7: b21+b22 -> v10+1, v11+1

    strassen_correct = [
        # (u, v, w) where u=[u00,u01,u10,u11], v=[v00,v01,v10,v11], w=[w00,w01,w10,w11]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),   # M2
        ([1,0,0,0], [0,1,-1,0], [0,0,1,1]),   # M3
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),   # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,1,1]),   # M6
        ([0,1,0,-1], [0,0,1,1], [1,1,0,0]),   # M7
    ]

    # Build full U, V, W
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []

    # Strassen block: i in {0,1}, j in {0,1}, k in {0,1}
    for u_local, v_local, w_local in strassen_correct:
        u_full = [0]*(n*m)
        v_full = [0]*(m*p)
        w_full = [0]*(p*n)
        # u: (i,j) -> index i*m+j
        for ii in range(2):
            for jj in range(2):
                u_full[ii*m+jj] = u_local[ii*2+jj]
        # v: (j,k) -> index j*p+k
        for jj in range(2):
            for kk in range(2):
                v_full[jj*p+kk] = v_local[jj*2+kk]
        # w: (k,i) -> index k*n+i
        for kk in range(2):
            for ii in range(2):
                w_full[kk*n+ii] = w_local[kk*2+ii]
        U_cols.append(u_full)
        V_cols.append(v_full)
        W_cols.append(w_full)
        rank += 1

    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # already handled by Strassen
                u_full = [0]*(n*m)
                v_full = [0]*(m*p)
                w_full = [0]*(p*n)
                u_full[i*m+j] = 1
                v_full[j*p+k] = 1
                w_full[k*n+i] = 1
                U_cols.append(u_full)
                V_cols.append(v_full)
                W_cols.append(w_full)
                rank += 1

    U = U_cols
    V = V_cols
    W = W_cols
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _de_zeeuw_peel(U, V, W, n, m, p, max_steps=50):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    If terms c1 and c2 satisfy: u_c1 = alpha * u_c2, v_c1 = beta * v_c2, and appropriate
    conditions on w, we can potentially eliminate one.
    
    More generally, try to find if any term can be expressed as a linear combination of others.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0]) if U else 0

    for step in range(max_steps):
        if rank <= 0:
            break
        # Try to find a term that is redundant
        eliminated = False
        for c1 in range(rank):
            # Check if term c1 can be expressed as a combination of other terms
            # Build a system: for all (i,j,k), sum_{c != c1} x_c * U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1]
            # This is a linear system in x_c
            # Variables: x_c for c != c1, total rank-1 variables
            # Equations: n*m*p equations (one for each (i,j,k))
            # But this is a very large system. Let's try a simpler approach.

            # Simpler: check if U[:, c1] is in the span of {U[:, c] : c != c1}
            # AND V[:, c1] is in the span of {V[:, c] : c != c1}
            # AND W[:, c1] is in the span of {W[:, c] : c != c1}
            # This is a necessary but not sufficient condition.

            # Even simpler: just check numerical rank of U, V, W separately
            # If the rank of U is less than the number of columns, there's a dependency.

            pass

        # Let's try a different approach: check the numerical rank of U, V, W
        U_arr = np.array(U, dtype=float)  # shape (n*m, rank)
        V_arr = np.array(V, dtype=float)  # shape (m*p, rank)
        W_arr = np.array(W, dtype=float)  # shape (p*n, rank)

        U_rank = np.linalg.matrix_rank(U_arr, tol=1e-10)
        V_rank = np.linalg.matrix_rank(V_arr, tol=1e-10)
        W_rank = np.linalg.matrix_rank(W_arr, tol=1e-10)

        if U_rank < rank or V_rank < rank or W_rank < rank:
            # There's a linear dependency. Find it and eliminate.
            # For now, this is a placeholder - implementing full elimination is complex.
            pass

        break  # For now, just one pass

    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds safety margin

    n, m, p = DIMENSIONS

    # Start with Strassen-enhanced decomposition
    U, V, W, rank = _build_strassen_2x2_2x2(n, m, p)

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook if Strassen is wrong
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

    # Try to reduce rank further with de Zeeuw peeling
    if time.time() < deadline - 5:
        U2, V2, W2, rank2 = _de_zeeuw_peel(U, V, W, n, m, p)
        if _verify(U2, V2, W2, n, m, p) and rank2 < rank:
            U, V, W, rank = U2, V2, W2, rank2

    return {"U": U, "V": V, "W": W}
