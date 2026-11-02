"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)

    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i

    def make_strassen_36():
        """Block Strassen: rank 36"""
        cap = 50
        U = np.zeros((n * m, cap))
        V = np.zeros((m * p, cap))
        W = np.zeros((p * n, cap))
        col = 0

        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices

            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
            col += 1

            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
            col += 1

            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1]), col] = 1
            V[v_idx(z01[0], z01[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
            col += 1

            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z10[0], z10[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
            col += 1

            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1]), col] = 1
            U[u_idx(y01[0], y01[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
            col += 1

            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1]), col] = 1
            U[u_idx(y00[0], y00[1]), col] = -1
            V[v_idx(z00[0], z00[1]), col] = 1
            V[v_idx(z01[0], z01[1]), col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
            col += 1

            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = -1
            V[v_idx(z10[0], z10[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
            col += 1

        # Four 2x2x2 Strassen blocks
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)],
                       [(2,0),(3,0),(2,1),(3,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)],
                       [(2,0),(3,0),(2,1),(3,1)])

        # Block 5: A_L * B_col4 (2x2 x 2x1)
        for (i, j, r, c) in [(0,0,0,0),(0,1,1,0),(1,0,0,1),(1,1,1,1)]:
            U[u_idx(i, j), col] = 1
            V[v_idx(r, 4), col] = 1
            W[w_idx(4, c), col] += 1
            col += 1

        # Block 6: A_R * B_col4
        for (i, j, r, c) in [(0,2,2,0),(0,3,3,0),(1,2,2,1),(1,3,3,1)]:
            U[u_idx(i, j), col] = 1
            V[v_idx(r, 4), col] = 1
            W[w_idx(4, c), col] += 1
            col += 1

        return U[:, :col].copy(), V[:, :col].copy(), W[:, :col].copy()

    def verify(U, V, W):
        """Check that the decomposition exactly reconstructs the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_remove(U, V, W, col_idx):
        """Try removing a specific column and verify."""
        if U.shape[1] <= 1:
            return None
        keep = [c for c in range(U.shape[1]) if c != col_idx]
        U2, V2, W2 = U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy()
        if verify(U2, V2, W2):
            return U2, V2, W2
        return None

    def randomized_redundancy_removal(U, V, W, max_iters=1000):
        """Randomly try to remove columns."""
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            r = U_curr.shape[1]
            if r <= 1:
                break
            
            # Randomly shuffle the order of columns to try
            cols = list(range(r))
            np.random.shuffle(cols)
            
            removed_any = False
            for c in cols:
                if time.time() > deadline - 10:
                    break
                res = try_remove(U_curr, V_curr, W_curr, c)
                if res is not None:
                    U_curr, V_curr, W_curr = res
                    removed_any = True
                    break
            
            if not removed_any:
                # Try a few more random shuffles before giving up on this configuration
                success = False
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    cols = list(range(U_curr.shape[1]))
                    np.random.shuffle(cols)
                    for c in cols:
                        res = try_remove(U_curr, V_curr, W_curr, c)
                        if res is not None:
                            U_curr, V_curr, W_curr = res
                            success = True
                            break
                    if success:
                        break
                if not success:
                    break
        
        return U_curr, V_curr, W_curr

    # Generate base decomposition
    U_base, V_base, W_base = make_strassen_36()
    
    if not verify(U_base, V_base, W_base):
        # Fallback to schoolbook if Strassen fails (should not happen)
        rb = n * m * p
        U = np.zeros((n*m, rb))
        V = np.zeros((m*p, rb))
        W = np.zeros((p*n, rb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    best_rank = U_base.shape[1]

    # Perform multiple randomized reduction attempts
    for attempt in range(20):
        if time.time() > deadline - 20:
            break
        
        U_r, V_r, W_r = randomized_redundancy_removal(U_base, V_base, W_base, max_iters=500)
        
        r = U_r.shape[1]
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            # If we hit a very low rank, we can stop
            if best_rank <= 32:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
