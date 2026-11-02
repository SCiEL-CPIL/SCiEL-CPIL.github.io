import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Build the verified rank-71 Strassen-based decomposition
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]

    total_strassen_rank = len(blocks) * 7
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True

    schoolbook_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    schoolbook_count += 1

    total_rank = total_strassen_rank + schoolbook_count

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Convert to numpy for efficient operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]

    def verify(Ua, Va, Wa):
        """Check if the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(Ua.shape[1]):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Verify initial decomposition
    if not verify(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        sb_rank = n * m * p
        U_sb = [[0.0] * sb_rank for _ in range(n * m)]
        V_sb = [[0.0] * sb_rank for _ in range(m * p)]
        W_sb = [[0.0] * sb_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][c] = 1.0
                    V_sb[j*p+k][c] = 1.0
                    W_sb[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Local search: try to find linearly dependent columns
    # Two columns r1, r2 are "redundant" if there exist coefficients a, b
    # such that a*outer(u_r1, v_r1, w_r1) + b*outer(u_r2, v_r2, w_r2) = 0
    # This means u_r1/v_r1/w_r1 are proportional to u_r2/v_r2/w_r2
    
    # More generally: check if any column is a linear combination of others
    # in the tensor product space
    
    # Strategy 1: Check for proportional columns (simplest case)
    def find_proportional_pairs(Ua, Va, Wa, max_check=2000):
        """Find pairs of proportional columns."""
        rank = Ua.shape[1]
        pairs = []
        checked = 0
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1+1, rank):
                if time.time() > deadline:
                    break
                checked += 1
                if checked > max_check:
                    break
                
                # Check if column r1 is proportional to column r2
                # u_r1 should be proportional to u_r2
                u1 = Ua[:, r1]
                u2 = Ua[:, r2]
                v1 = Va[:, r1]
                v2 = Va[:, r2]
                w1 = Wa[:, r1]
                w2 = Wa[:, r2]
                
                # Find scale factor from U
                nz_u1 = np.nonzero(u1)[0]
                nz_u2 = np.nonzero(u2)[0]
                if len(nz_u1) == 0 or len(nz_u2) == 0:
                    continue
                
                # Check proportionality on U
                scale_u = None
                for idx in nz_u1:
                    if u2[idx] != 0:
                        if scale_u is None:
                            scale_u = u1[idx] / u2[idx]
                        elif abs(u1[idx] / u2[idx] - scale_u) > 1e-10:
                            break
                else:
                    if scale_u is None:
                        continue
                    # Verify on V
                    ok = True
                    for idx in nz_u2:
                        if u1[idx] != 0 and abs(u1[idx] - scale_u * u2[idx]) > 1e-10:
                            ok = False
                            break
                        elif u1[idx] == 0 and abs(u2[idx]) > 1e-10:
                            ok = False
                            break
                    if ok:
                        for idx in range(len(v1)):
                            if abs(v1[idx] - scale_u * v2[idx]) > 1e-10:
                                ok = False
                                break
                    if ok:
                        for idx in range(len(w1)):
                            if abs(w1[idx] - scale_u * w2[idx]) > 1e-10:
                                ok = False
                                break
                    if ok:
                        pairs.append((r1, r2, scale_u))
                        break  # Only one pair per r1 for now
        return pairs

    # Strategy 2: Try to find a new rank-1 term that can replace 2+ existing terms
    # This is harder. Let's try a different approach:
    # Check if removing one column and adjusting others still works.
    
    # Strategy 3: Systematic elimination - try removing each column and see
    # if the remaining columns can be adjusted (scaled) to still give correct result.
    
    # Actually, the most promising approach: check if any 2 columns sum to zero
    # (i.e., they are negatives of each other)
    def find_cancelling_pairs(Ua, Va, Wa):
        """Find pairs of columns that cancel each other."""
        rank = Ua.shape[1]
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1+1, rank):
                if time.time() > deadline:
                    break
                # Check if col r1 = -col r2
                if np.allclose(Ua[:, r1], -Ua[:, r2], atol=1e-10) and \
                   np.allclose(Va[:, r1], -Va[:, r2], atol=1e-10) and \
                   np.allclose(Wa[:, r1], -Wa[:, r2], atol=1e-10):
                    return (r1, r2)
        return None

    # Try to find and remove cancelling pairs
    cancel_pair = find_cancelling_pairs(U_arr, V_arr, W_arr)
    if cancel_pair is not None:
        r1, r2 = cancel_pair
        # Remove both columns
        keep = [c for c in range(rank) if c != r1 and c != r2]
        U_arr = U_arr[:, keep]
        V_arr = V_arr[:, keep]
        W_arr = W_arr[:, keep]
        rank -= 2
        print(f"Removed cancelling pair ({r1},{r2}), new rank: {rank}")

    # Try proportional pairs
    prop_pairs = find_proportional_pairs(U_arr, V_arr, W_arr, max_check=5000)
    for r1, r2, scale in prop_pairs:
        if time.time() > deadline:
            break
        # If col r1 = scale * col r2, we can't just remove one (they're not cancelling)
        # But if scale = 0 or -1, we could remove one
        if abs(scale - 1.0) < 1e-10:
            # They're identical, remove one
            keep = [c for c in range(U_arr.shape[1]) if c != r2]
            U_arr = U_arr[:, keep]
            V_arr = V_arr[:, keep]
            W_arr = W_arr[:, keep]
            rank = U_arr.shape[1]
            print(f"Removed duplicate column {r2}, new rank: {rank}")
            break

    # Strategy 4: Try to find a rank-1 term that, when added, allows removing 2+ columns
    # This is the key to going below 71.
    # Approach: For each pair of columns (r1, r2), check if their tensor sum
    # can be represented as a single rank-1 term.
    # T_r1 + T_r2 = u1⊗v1⊗w1 + u2⊗v2⊗w2
    # This is rank-1 iff the 3-way tensor has rank 1, which requires
    # u1, u2 to be proportional AND v1, v2 proportional AND w1, w2 proportional.
    # We already checked that above.
    
    # Strategy 5: Try replacing 3 columns with 2 columns.
    # For each triple (r1, r2, r3), check if T_r1 + T_r2 + T_r3 can be
    # written as sum of 2 rank-1 terms. This is always possible (rank ≤ 3),
    # but we need rank exactly 2, which is a constraint.
    # This is computationally expensive. Let's try a heuristic.
    
    # Strategy 6: Use the structure - look for columns that share the same
    # U-support pattern but differ in V/W, and see if they can be merged.
    
    # For now, let's also try: can we find a new rank-1 term T_new such that
    # T_new + T_r1 + T_r2 = 0 for some pair (r1, r2)?
    # Then we can remove r1 and r2 and add T_new, net change: -1.
    
    def try_add_cancel_term(Ua, Va, Wa, max_attempts=500):
        """Try to find a new rank-1 term that cancels a pair of existing columns."""
        rank = Ua.shape[1]
        attempts = 0
        for r1 in range(rank):
            if time.time() > deadline or attempts > max_attempts:
                break
            for r2 in range(r1+1, rank):
                if time.time() > deadline or attempts > max_attempts:
                    break
                attempts += 1
                
                # We want: u_new ⊗ v_new ⊗ w_new = -(u_r1 ⊗ v_r1 ⊗ w_r1 + u_r2 ⊗ v_r2 ⊗ w_r2)
                # The RHS is a 3-way tensor. We need it to be rank-1.
                # Build the tensor T = -(T_r1 + T_r2)
                # T[i,j,k] = -(Ua[i*m+j,r1]*Va[j*p+k,r1]*Wa[k*n+i,r1] + Ua[i*m+j,r2]*Va[j*p+k,r2]*Wa[k*n+i,r2])
                
                # Check if T is rank-1: all 2x2 minors of any unfolding must be zero
                # Use the U-unfolding: T_U[a, (b,c)] where a indexes U, (b,c) indexes V×W
                # Actually, simpler: check if the 3-way tensor has rank 1 by checking
                # if all entries factorize.
                
                # Build T as a 3D array
                T = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i*m+j, j*p+k, k*n+i] = -(
                                Ua[i*m+j, r1] * Va[j*p+k, r1] * Wa[k*n+i, r1] +
                                Ua[i*m+j, r2] * Va[j*p+k, r2] * Wa[k*n+i, r2]
                            )
                
                # Check if T is rank-1: T[a,b,c] = x[a]*y[b]*z[c]
                # Find a nonzero entry to determine scales
                nonzero = np.argwhere(T != 0)
                if len(nonzero) == 0:
                    continue  # T is zero, so r1 and r2 already cancel
                
                a0, b0, c0 = nonzero[0]
                val = T[a0, b0, c0]
                
                # Try to factorize
                # x[a] = T[a, b0, c0] / (y[b0]*z[c0])
                # We need to determine y and z.
                # From T[a0, b, c0] = x[a0]*y[b]*z[c0] → y[b] = T[a0,b,c0]/(x[a0]*z[c0])
                # From T[a0, b0, c] = x[a0]*y[b0]*z[c] → z[c] = T[a0,b0,c]/(x[a0]*y[b0])
                
                # This is circular. Let's use a different approach:
                # Check if all 2x2 minors of the matrix T[a0, :, :] are zero
                # and T[:, b0, :] and T[:, :, c0] have rank 1.
                
                mat1 = T[a0, :, :]  # shape (m*p, p*n)
                mat2 = T[:, b0, :]  # shape (n*m, p*n)
                mat3 = T[:, :, c0]  # shape (n*m, m*p)
                
                # Check rank 1 of each
                def is_rank1(M, tol=1e-10):
                    if M.shape[0] == 0 or M.shape[1] == 0:
                        return True
                    # All 2x2 minors should be zero
                    if M.shape[0] >= 2 and M.shape[1] >= 2:
                        # Check a few 2x2 minors
                        for i in range(min(2, M.shape[0])):
                            for j in range(min(2, M.shape[1])):
                                for i2 in range(i+1, min(i+2, M.shape[0])):
                                    for j2 in range(j+1, min(j+2, M.shape[1])):
                                        minor = M[i,j]*M[i2,j2] - M[i,j2]*M[i2,j]
                                        if abs(minor) > tol:
                                            return False
                    return True
                
                if not (is_rank1(mat1) and is_rank1(mat2) and is_rank1(mat3)):
                    continue
                
                # T is rank-1! Extract the factorization.
                # x[a] ∝ T[a, b0, c0] for fixed (b0,c0)
                # y[b] ∝ T[a0, b, c0] for fixed (a0,c0)  
                # z[c] ∝ T[a0, b0, c] for fixed (a0,b0)
                
                x = T[:, b0, c0].copy()
                y = T[a0, :, c0].copy()
                z = T[a0, b0, :].copy()
                
                # Normalize: T[a0,b0,c0] = x[a0]*y[b0]*z[c0]
                # So we need to adjust scales.
                # Let x' = x / x[a0], y' = y / y[b0], z' = z / z[c0]
                # Then T[a,b,c] should = x'[a]*y'[b]*z'[c]*T[a0,b0,c0]
                
                if abs(x[a0]) < 1e-15 or abs(y[b0]) < 1e-15 or abs(z[c0]) < 1e-15:
                    continue
                
                x_norm = x / x[a0]
                y_norm = y / y[b0]
                z_norm = z / z[c0]
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = T[i*m+j, j*p+k, k*n+i]
                            actual = x_norm[i*m+j] * y_norm[j*p+k] * z_norm[k*n+i]
                            err = abs(expected - actual)
                            if err > max_err:
                                max_err = err
                
                if max_err > 1e-8:
                    continue
                
                # We found a valid rank-1 term!
                # Add it and remove r1, r2
                new_U = np.zeros((n*m, rank - 1))
                new_V = np.zeros((m*p, rank - 1))
                new_W = np.zeros((p*n, rank - 1))
                
                new_cols = [c for c in range(rank) if c != r1 and c != r2]
                new_U[:, :] = Ua[:, new_cols]
                new_V[:, :] = Va[:, new_cols]
                new_W[:, :] = Wa[:, new_cols]
                
                # Add the new term at the end
                new_U[:, rank-1] = x_norm
                new_V[:, rank-1] = y_norm
                new_W[:, rank-1] = z_norm
                
                # Verify
                if verify(new_U, new_V, new_W):
                    return new_U, new_V, new_W, (r1, r2)
        
        return None

    result = try_add_cancel_term(U_arr, V_arr, W_arr, max_attempts=200)
    if result is not None:
        U_arr, V_arr, W_arr, removed = result
        rank = U_arr.shape[1]
        print(f"Added cancelling term, removed {removed}, new rank: {rank}")

    # Final verification
    if not verify(U_arr, V_arr, W_arr):
        # Fallback to the original rank-71
        U_final = [[U_arr[i, j] for j in range(rank)] for i in range(n*m)]
        V_final = [[V_arr[i, j] for j in range(rank)] for i in range(m*p)]
        W_final = [[W_arr[i, j] for j in range(rank)] for i in range(p*n)]
        # Rebuild from scratch
        U = [[0.0] * 71 for _ in range(n * m)]
        V = [[0.0] * 71 for _ in range(m * p)]
        W = [[0.0] * 71 for _ in range(p * n)]
        col = 0
        for i0, j0, k0 in blocks:
            add_strassen_block(i0, j0, k0, col)
            col += 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        return {"U": U, "V": V, "W": W}

    U_final = [[U_arr[i, j] for j in range(rank)] for i in range(n*m)]
    V_final = [[V_arr[i, j] for j in range(rank)] for i in range(m*p)]
    W_final = [[W_arr[i, j] for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Quick verification
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    print(f"Verified: {ok}")

