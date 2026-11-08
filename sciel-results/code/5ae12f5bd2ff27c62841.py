"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Schoolbook as guaranteed fallback
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    school_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_school
    best_result = school_result

    # The multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k] to get C[i,k]
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # This is a 3-tensor of shape (n*m, m*p, p*n) = (8, 28, 14)
    # with T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i are consistent (same j, same k)
    # Actually T is the multiplication tensor: T[a,b,c] = 1 iff there exist i,j,k with
    # a = i*m+j, b = j*p+k, c = k*n+i, else 0.
    # Since each (i,j,k) gives exactly one (a,b,c) triple, T is a permutation-like tensor.

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find low-rank decomposition
    # Strategy: coordinate descent on rank r, entries from a small grid
    # Start with small ranks and increase

    grid_values = [0, 0.5, 1, 1.5, 2, 2.5, 3]

    def frob_residual(U, V, W, r):
        """Compute ||tensor(U,V,W) - T||_F"""
        residual = 0.0
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # outer product uc[None,None,:] * vc[None,:,None] * wc[:,None,None]
            # But we only need to compare with T, so accumulate
            pass
        # More efficient: build the full tensor
        # tensor = sum_c np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        # For small r this is fine
        tensor = np.zeros((n * m, m * p, p * n))
        for c in range(r):
            tensor += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(tensor - T)

    def coordinate_descent(U, V, W, r, max_iters=500, tol=1e-6):
        """Local search: fix two matrices, optimize the third by coordinate updates."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        prev_res = frob_residual(U, V, W, r)
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for c in range(r):
                # Optimize column c of U
                # Given V[:,c] and W[:,c], find best u in grid to minimize residual
                # residual contribution: sum_c' [outer(U_c', V_c', W_c')] - T
                # Fix all other columns, optimize U[:,c]
                # This is: min_u ||sum_{c'} u_{c'} ⊗ V_{c'} ⊗ W_{c'} - T||
                # = min_u ||(sum_{c'≠c} ...) + u ⊗ V_c ⊗ W_c - T||
                # For fixed u, the tensor is linear in u.
                # We want to minimize ||A + u * B - T||_F^2 where B = V_c ⊗ W_c (flattened)
                # = ||A - T + u*B||^2 = ||(A-T)||^2 + 2u * <A-T, B> + u^2 ||B||^2
                # Unconstrained min: u* = -<A-T, B> / ||B||^2
                # Constrained to grid: pick nearest grid value
                # But entries are per-row, so we do coordinate descent per row.

                # Per-row coordinate descent for U[:,c]:
                for a_idx in range(n * m):
                    # Current residual with U[a_idx, c] fixed
                    # Changing U[a_idx,c] by delta changes tensor by delta * (e_a ⊗ V_c ⊗ W_c)
                    # where e_a is unit vector
                    # So we want to minimize over u in grid:
                    # ||(current_tensor - T) + (u - old_u) * e_a ⊗ V_c ⊗ W_c||^2
                    # = ||res_old + (u-old_u)*B_a||^2 where B_a = e_a ⊗ V_c ⊗ W_c
                    # Expand: ||res_old||^2 + 2(u-ou)*<res_old, B_a> + (u-ou)^2||B_a||^2
                    # min at u = ou - <res_old, B_a>/||B_a||^2
                    # Compute B_a = e_a ⊗ V_c ⊗ W_c: shape (n*m, m*p, p*n)
                    # B_a[a, b, cc] = V_c[b] * W_c[cc]
                    B_a = np.outer(V[:, c], W[:, c])  # shape (m*p, p*n)
                    # res_old at position a: res_full[a, :, :]
                    # We need the full residual tensor
                    # This is expensive to recompute each time. Let's compute it once per column update.

                # Recompute residual tensor once
                tensor = np.zeros((n * m, m * p, p * n))
                for c2 in range(r):
                    tensor += np.einsum('a,b,c->abc', U[:, c2], V[:, c2], W[:, c2])
                res = tensor - T

                # Optimize each row of U[:, c]
                for a_idx in range(n * m):
                    B_a = np.outer(V[:, c], W[:, c])  # (m*p, p*n)
                    res_a = res[a_idx, :, :]  # (m*p, p*n)
                    # <res, B_a> = sum(res_a * B_a)
                    dot = np.sum(res_a * B_a)
                    norm_sq = np.sum(B_a * B_a)
                    if norm_sq < 1e-12:
                        continue
                    old_u = U[a_idx, c]
                    # optimal continuous u = old_u - dot / norm_sq
                    u_opt = old_u - dot / norm_sq
                    # snap to grid
                    best_grid = min(grid_values, key=lambda g: abs(g - u_opt))
                    if best_grid != old_u:
                        # Check improvement
                        U[a_idx, c] = best_grid
                        new_dot = np.sum((res_a + (best_grid - old_u) * B_a) ** 2)
                        old_dot = np.sum(res_a ** 2)
                        if new_dot < old_dot - 1e-10:
                            res[a_idx, :, :] = res_a + (best_grid - old_u) * B_a
                            tensor[a_idx, :, :] += (best_grid - old_u) * B_a
                            improved = True
                        else:
                            U[a_idx, c] = old_u

                # Same for V[:, c]
                for b_idx in range(m * p):
                    B_b = np.outer(U[:, c], W[:, c])  # (n*m, p*n)
                    res_b = res[:, b_idx, :]  # (n*m, p*n)
                    dot = np.sum(res_b * B_b)
                    norm_sq = np.sum(B_b * B_b)
                    if norm_sq < 1e-12:
                        continue
                    old_v = V[b_idx, c]
                    v_opt = old_v - dot / norm_sq
                    best_grid = min(grid_values, key=lambda g: abs(g - v_opt))
                    if best_grid != old_v:
                        V[b_idx, c] = best_grid
                        new_dot = np.sum((res_b + (best_grid - old_v) * B_b) ** 2)
                        old_dot = np.sum(res_b ** 2)
                        if new_dot < old_dot - 1e-10:
                            res[:, b_idx, :] = res_b + (best_grid - old_v) * B_b
                            tensor[:, b_idx, :] += (best_grid - old_v) * B_b
                            improved = True
                        else:
                            V[b_idx, c] = old_v

                # Same for W[:, c]
                for cc_idx in range(p * n):
                    B_w = np.outer(U[:, c], V[:, c])  # (n*m, m*p)
                    res_w = res[:, :, cc_idx]  # (n*m, m*p)
                    dot = np.sum(res_w * B_w)
                    norm_sq = np.sum(B_w * B_w)
                    if norm_sq < 1e-12:
                        continue
                    old_w = W[cc_idx, c]
                    w_opt = old_w - dot / norm_sq
                    best_grid = min(grid_values, key=lambda g: abs(g - w_opt))
                    if best_grid != old_w:
                        W[cc_idx, c] = best_grid
                        new_dot = np.sum((res_w + (best_grid - old_w) * B_w) ** 2)
                        old_dot = np.sum(res_w ** 2)
                        if new_dot < old_dot - 1e-10:
                            res[:, :, cc_idx] = res_w + (best_grid - old_w) * B_w
                            tensor[:, :, cc_idx] += (best_grid - old_w) * B_w
                            improved = True
                        else:
                            W[cc_idx, c] = old_w

            if not improved:
                break

        final_res = frob_residual(U, V, W, r)
        return U, V, W, final_res

    # Try ranks from small to large
    for r in [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            break

        # Multiple random restarts
        n_restarts = 3 if r <= 16 else 2
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            np.random.seed(42 + r * 100 + restart)
            U = np.random.choice(grid_values, size=(n * m, r))
            V = np.random.choice(grid_values, size=(m * p, r))
            W = np.random.choice(grid_values, size=(p * n, r))

            # Scale to roughly match: each entry of T is 1, and there are n*m*m*p*p*n...
            # Actually T has n*m*p ones. Each rank-1 term contributes at most (max|u|*max|v|*max|w|) per entry.
            # Just run coordinate descent.
            U, V, W, final_res = coordinate_descent(U, V, W, r, max_iters=300)

            if final_res < 1e-6:
                # Found exact decomposition!
                # Snap to exact half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                # Verify
                tensor = np.zeros((n * m, m * p, p * n))
                for c in range(r):
                    tensor += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                if np.allclose(tensor, T, atol=1e-9):
                    # Check if we can reduce rank by removing zero columns
                    # and verify entries are half-integers
                    for mat in [U, V, W]:
                        assert np.allclose(mat * 2, np.round(mat * 2)), "Not half-integer"

                    # Try to reduce rank
                    for r2 in range(1, r + 1):
                        if time.time() > deadline - 5:
                            break
                        # Check if first r2 columns suffice
                        U2, V2, W2 = U[:, :r2], V[:, :r2], W[:, :r2]
                        tensor2 = np.zeros((n * m, m * p, p * n))
                        for c in range(r2):
                            tensor2 += np.einsum('a,b,c->abc', U2[:, c], V2[:, c], W2[:, c])
                        if np.allclose(tensor2, T, atol=1e-9):
                            r = r2
                            U, V, W = U2, V2, W2
                            break

                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    # Convert to clean integer/half-integer representation
                    U_list = [[(int(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-9 else x) for x in row] for row in U]
                    V_list = [[(int(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-9 else x) for x in row] for row in V]
                    W_list = [[(int(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-9 else x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break

            if time.time() > deadline - 10:
                break

    # If no better found, try a structured approach: Kronecker product decomposition
    # For 2x4 x 4x7, we can think of it as block structure.
    # A is 2x4 = 2 blocks of 2x2, B is 4x7 = 2 blocks of 2x7
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using Strassen-like: for 2x2 by 2x7, rank is at most 7*2 = 14 per pair?
    # Actually for M_{2,2,7}, rank ≤ 7 (if we can do 2x2 with 7 mults and reuse)
    # But that's for square. For 2x2 by 2x7: rank = 14 (7 for each of 2 rows of output? No.)
    # M_{2,2,7}: at most 14, at least 7. Known: rank(M_{2,2,7}) = 13 or 14.

    # Let's try a specific structured decomposition for 2x4 x 4x7
    # Split into two 2x2 x 2x7 problems (for the two row-blocks of A)
    # Each 2x2 x 2x7 has rank at most 14
    # Total: 28
    # This is better than 56!

    # Actually, let me construct a rank-28 decomposition:
    # A = [[A1, A2], [A3, A4]] where each Ai is 1x2
    # B = [[B1], [B2]] where B1, B2 are 2x7
    # C[0,k] = A1@B1 + A2@B2
    # C[1,k] = A3@B1 + A4@B2

    # This is just schoolbook: 2*2*7 = 28 multiplications for each output row? No.
    # C[0,k] = sum_{j=0}^{3} A[0,j]*B[j,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # That's 4 terms per k, 7 k's = 28 for row 0, same for row 1 = 56 total. That's schoolbook.

    # To get 28, we'd need to share multiplications between rows.
    # (A[0,0]+A[1,0]) * B[0,k] gives contribution to both C[0,k] and C[1,k].
    # Let s0 = A[0,0]+A[1,0], d0 = A[0,0]-A[1,0]
    # Then A[0,0] = (s0+d0)/2, A[1,0] = (s0-d0)/2
    # C[0,k] += A[0,0]*B[0,k] = (s0+d0)/2 * B[0,k]
    # C[1,k] += A[1,0]*B[0,k] = (s0-d0)/2 * B[0,k]
    # So we need: s0*B[0,k] and d0*B[0,k], which is 2 multiplications per k per column-block.
    # That's still 2*7 = 14 per column pair, times 2 column pairs = 28. Same as schoolbook!

    # The savings come from Strassen: 7 instead of 8 for 2x2.
    # For 2x2 by 2x7:
    # C = [[c11, c12, ..., c17], [c21, c22, ..., c27]]
    # = [[a11,a12],[a21,a22]] * [[b11,...,b17],[b21,...,b27]]
    # c1k = a11*b1k + a12*b2k
    # c2k = a21*b1k + a22*b2k
    # Strassen:
    # p1 = (a21-a11)(b12-b11)  -- this is for 2x2 square, not directly applicable
    # For 2x2 by 2x7, we can use:
    # p1 = a11*b1, p2 = a12*b2, p3 = (a11+a12)*b1, p4 = (a21+a22)*b2, p5 = (a22-a11)*(b1+b2), p6 = (a11-a21)*b2, p7 = (a12-a22)*b1
    # where b1, b2 are row-vectors of length 7.
    # c1 = p3 + p2 - p5 - p7  (wait, let me recheck)
    # Standard Strassen for 2x2:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)b11
    # p7 = (a12-a22)(b21+b11)
    # c11 = p1+p4-p5+p7
    # c12 = p3+p5
    # c21 = p2+p4
    # c22 = p1-p2+p3+p6

    # For 2x2 by 2x7, the "b" entries are vectors of length 7.
    # p1 = (a11+a22) * (b1_vec + b2_vec)  -> 7 mults
    # p2 = (a21+a22) * b1_vec -> 7 mults
    # p3 = a11 * (b1_vec - b2_vec) -> 7 mults
    # p4 = a22 * (b2_vec - b1_vec) -> 7 mults
    # p5 = (a11+a12) * b2_vec -> 7 mults
    # p6 = (a21-a11) * b1_vec -> 7 mults
    # p7 = (a12-a22) * (b2_vec + b1_vec) -> 7 mults
    # Total: 49 multiplications for 2x2 by 2x7!
    # Schoolbook is 4*7 = 28. So Strassen is WORSE here because the "blocks" are 1x7, not 1x1.
    # Strassen helps when both dimensions are >= 2.

    # So for 2x4 x 4x7, the best known is probably schoolbook 56, or maybe a bit less
    # with clever sharing. The rank of M_{2,4,7} is between 32 and 56.

    # Given the difficulty, return the best we found.
    return best_result
# EVOLVE-BLOCK-END
