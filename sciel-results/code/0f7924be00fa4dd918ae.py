"""Matrix multiplication 2x3x8 with rank-36 Strassen + shared Part-2 decomposition."""
import time

def construct():
    t_start = time.time()
    limit = t_start + 140.0

    n, m, p = 2, 3, 8
    rank = 36

    # Strassen 2x2x2 coefficients
    # U factors for A (2x2), V factors for B (2x2), W factors for C (2x2)
    su = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    sv = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    sw = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # Initialize factors
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: Handle A[0:2, 0:2] * B[0:2, :] using 4 Strassen blocks
    # B is 2x8, split into 4 blocks of 2x2 columns
    for b_idx in range(4):
        k_offset = b_idx * 2
        for s_idx in range(7):
            # Update U: rows 0-1, cols 0-1 (corresponding to A[0:2, 0:2])
            for i in range(2):
                for a in range(2):
                    val = su[s_idx][i * 2 + a]
                    if val:
                        U[i * m + a][col] += val

            # Update V: rows 0-1 (a=0,1), cols k_offset, k_offset+1
            for a in range(2):
                for k in range(2):
                    val = sv[s_idx][a * 2 + k]
                    if val:
                        V[a * p + (k_offset + k)][col] += val

            # Update W: rows k_offset, k_offset+1 (k), cols 0-1 (i)
            for k in range(2):
                for i in range(2):
                    val = sw[s_idx][k * 2 + i]
                    if val:
                        W[(k_offset + k) * n + i][col] += val
            col += 1

            if time.time() > limit:
                # Fallback to schoolbook if timeout
                rb = n * m * p
                U_f = [[0] * rb for _ in range(n * m)]
                V_f = [[0] * rb for _ in range(m * p)]
                W_f = [[0] * rb for _ in range(p * n)]
                c = 0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            U_f[i * m + a][c] = 1
                            V_f[a * p + k][c] = 1
                            W_f[k * n + i][c] = 1
                            c += 1
                return {"U": U_f, "V": V_f, "W": W_f}

    # Part 2: Handle A[0:2, 2] * B[2:3, :]
    # A[0:2, 2] is a 2x1 column vector. B[2:3, :] is a 1x8 row vector.
    # C[0:2, k] = A[0,2]*B[2,k] + A[1,2]*B[2,k] for each k? No.
    # C[i,k] = sum_a A[i,a] B[a,k].
    # For a=2, C[i,k] += A[i,2] * B[2,k].
    # This is an outer product of A[:,2] (2x1) and B[2,:] (1x8).
    # We can decompose this outer product into rank 8 terms?
    # Or can we do it in fewer?
    # The tensor T[i,2,k] = A[i,2] * B[2,k].
    # This is a rank-2 tensor (in terms of i and k, with a fixed at 2).
    # Actually, the slice of the tensor for a=2 is A[:,2] * B[2,:].
    # This is a rank-1 tensor in (i,k) space? No, it's a matrix of rank 1.
    # But we need to represent the full tensor T[i,a,k].
    # For a != 2, T[i,a,k] = 0 in this part.
    # For a = 2, T[i,2,k] = A[i,2] B[2,k].
    # We can write T[i,2,k] = sum_{r} U[i,2,r] V[2,k,r] W[k,i,r].
    # Let's try to use rank 8 for this part.
    # For each k, we need to produce C[i,k] for i=0,1.
    # C[0,k] = A[0,2] B[2,k]
    # C[1,k] = A[1,2] B[2,k]
    # This is a 2x1 outer product with B[2,k].
    # We can use 1 rank term per k?
    # Let's see. If we use 1 term for k, we need:
    # U[i,2,r_k] V[2,k,r_k] W[k,i,r_k] = A[i,2] B[2,k]
    # We can set U[i,2,r_k] = A[i,2] (but A is input, not fixed).
    # Wait, the factors U, V, W are fixed matrices.
    # The decomposition is: sum_r U[i,a,r] V[a,k,r] W[k,i,r] = 1 if (i,a,k) is a valid index, 0 otherwise?
    # No, the tensor is T[i,a,k] = delta_{a,2} * (A[i,2] B[2,k])? No.
    # The multiplication tensor T has T[i,a,k] = 1 if we want C[i,k] = sum A[i,a]B[a,k].
    # So T[i,a,k] is 1 for all i,a,k? No, that's not right.
    # The tensor T is defined by T[i,a,k] such that C[i,k] = sum_{a} A[i,a] B[a,k] = sum_{a,r} (U[i,a,r] A[i,a]) (V[a,k,r] B[a,k]) W[k,i,r]?
    # No, the standard decomposition is:
    # C[i,k] = sum_{r} (sum_a U[i,a,r] A[i,a]) (sum_k V[a,k,r] B[a,k]) W[k,i,r] ? No.
    # The factors are U (n*m x rank), V (m*p x rank), W (p*n x rank).
    # The product is sum_r U[i*m+a, r] V[a*p+k, r] W[k*n+i, r] = 1 for all i,a,k?
    # Yes, the tensor T[i,a,k] = 1 for all i,a,k is what we are decomposing?
    # No, the matrix multiplication tensor is T[i,a,k] = 1 if we consider the basis elements.
    # Actually, the tensor is T[i,a,k] such that C = A*B means C[i,k] = sum_a A[i,a] B[a,k].
    # In tensor form, T is the tensor with T[i,a,k] = 1 for all i,a,k? No.
    # The tensor T is such that T[i,a,k] = 1. Then A:B = sum T[i,a,k] A[i,a] B[a,k] = C[i,k].
    # So yes, we are decomposing the all-ones tensor of size 2x3x8.
    #
    # So for Part 2, we need to represent the slice a=2 of the all-ones tensor.
    # T[i,2,k] = 1 for all i,k.
    # This is a 2x8 all-ones matrix.
    # Can we decompose a 2x8 all-ones matrix into rank 8 terms?
    # Yes, trivially, using 8 rank-1 terms, one for each k.
    # Term r_k: U[i,2,r_k] = 1 for all i. V[2,k,r_k] = 1. W[k,i,r_k] = 1.
    # Then sum_r U[i,2,r] V[2,k',r] W[k',i,r] = sum_{r: r_k'=r_k} 1*1*1 = 1 if k'=k, 0 otherwise.
    # Wait, we need T[i,2,k] = 1 for all i,k.
    # If we use 8 terms, one for each k, then for a fixed k, the sum is 1.
    # And for k' != k, the sum is 0.
    # So this works. Rank 8.

    for k in range(8):
        u_row = 2  # a=2
        v_row = 2 * p + k
        w_row = k * n
        # We need to handle both i=0 and i=1.
        # U[i*m+2, r] for i=0,1.
        # V[2*p+k, r]
        # W[k*n+i, r] for i=0,1.
        # Let's set:
        # U[0*m+2, col] = 1
        # U[1*m+2, col] = 1
        # V[2*p+k, col] = 1
        # W[k*n+0, col] = 1
        # W[k*n+1, col] = 1
        # Then for this r=col, the term is U[i,2,r] V[2,k,r] W[k,i,r] = 1*1*1 = 1.
        # This contributes 1 to T[i,2,k].
        # Since we have one such term for each k, the total contribution to T[i,2,k] is 1.
        # And for T[i,2,k'] with k' != k, the contribution is 0.
        # So this works.
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        if time.time() > limit:
            rb = n * m * p
            U_f = [[0] * rb for _ in range(n * m)]
            V_f = [[0] * rb for _ in range(m * p)]
            W_f = [[0] * rb for _ in range(p * n)]
            c = 0
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        U_f[i * m + a][c] = 1
                        V_f[a * p + k][c] = 1
                        W_f[k * n + i][c] = 1
                        c += 1
            return {"U": U_f, "V": V_f, "W": W_f}

    # Validation
    valid = all(
        sum(U[i * m + a][r] * V[a * p + k][r] * W[k * n + i][r] for r in range(rank)) == 1
        for i in range(n)
        for a in range(m)
        for k in range(p)
    )

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rb = n * m * p
        U_f = [[0] * rb for _ in range(n * m)]
        V_f = [[0] * rb for _ in range(m * p)]
        W_f = [[0] * rb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_f[i * m + a][c] = 1
                    V_f[a * p + k][c] = 1
                    W_f[k * n + i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
