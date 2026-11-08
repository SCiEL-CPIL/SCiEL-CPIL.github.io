import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    n, m, p = 4, 4, 8
    N = n * m  # 16
    M = m * p  # 32
    P = p * n  # 32

    def verify_fast(U, V, W, tol=1e-9):
        """Fast vectorized verification of tensor decomposition."""
        # Reconstruct tensor T_rec(a,b,c) = sum_k U[a,k] V[b,k] W[c,k]
        T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
        
        # Construct target tensor T_target
        T_target = np.zeros((N, M, P))
        for a in range(N):
            i_a, j_a = divmod(a, 4)
            for b in range(M):
                r_b, c_b = divmod(b, 8)
                if j_a == r_b:
                    # c_b must match c_c, i_a must match i_c
                    # c index is c_c * 4 + i_c
                    # So c = c_b * 4 + i_a
                    c_target = c_b * 4 + i_a
                    T_target[a, b, c_target] = 1.0
        
        return np.allclose(T_rec, T_target, atol=tol)

    def build_block_strassen():
        """Build block Strassen decomposition for 4x4x8 using 2x2 Strassen sub-blocks.
        Rank = 2 halves * 7 P_k * 7 S_m = 98."""
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Strassen's 7 multiplications for 2x2 matrices
        # M0 = (A00+A11)(B00+B11)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M1 = (A10+A11)B00
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        # M2 = A00(B01-B11)
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        # M3 = A11(B10-B00)
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        # M4 = (A00+A01)B11
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        # M5 = (A10-A00)(B00+B01)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # M6 = (A01-A11)(B10+B11)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        U = np.zeros((N, 98))
        V = np.zeros((M, 98))
        W = np.zeros((P, 98))
        
        # A_specs: for each P_k (0..6), list of (coeff, block_idx)
        # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        A_specs = [
            [(1, 0), (1, 3)],    # P0: A00 + A11
            [(1, 2), (1, 3)],    # P1: A10 + A11
            [(1, 0)],            # P2: A00
            [(1, 3)],            # P3: A11
            [(1, 0), (1, 1)],    # P4: A00 + A01
            [(1, 2), (-1, 0)],   # P5: A10 - A00
            [(1, 1), (-1, 3)],   # P6: A01 - A11
        ]
        
        B_specs = [
            [(1, 0), (1, 3)],    # P0: B00 + B11
            [(1, 0)],            # P1: B00
            [(1, 1), (-1, 3)],   # P2: B01 - B11
            [(1, 2), (-1, 0)],   # P3: B10 - B00
            [(1, 3)],            # P4: B11
            [(1, 0), (1, 1)],    # P5: B00 + B01
            [(1, 2), (1, 3)],    # P6: B10 + B11
        ]
        
        # C_specs: how P_k contributes to C blocks
        # C00 = P0 + P3 - P4 + P6
        # C01 = P2 + P4
        # C10 = P1 + P3
        # C11 = P0 - P1 + P2 + P5
        C_specs = [
            [(1, 0), (1, 3)],    # P0: C00 + C11
            [(1, 2), (-1, 3)],   # P1: C10 - C11
            [(1, 1), (1, 3)],    # P2: C01 + C11
            [(1, 0), (1, 2)],    # P3: C00 + C10
            [(-1, 0), (1, 1)],   # P4: -C00 + C01
            [(1, 3)],            # P5: C11
            [(1, 0)],            # P6: C00
        ]
        
        col = 0
        for half in range(2):  # 0: B_L (cols 0-3), 1: B_R (cols 4-7)
            for k_p in range(7):  # P_k
                for m_s in range(7):  # S_m (Strassen sub-multiplication)
                    # U contribution
                    for coeff_a, block_idx in A_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_a_idx = global_r * 4 + global_c
                                a_eff_idx = r * 2 + c
                                U[global_a_idx, col] += coeff_a * U2[a_eff_idx, m_s]
                    
                    # V contribution
                    for coeff_b, block_idx in B_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_k = global_c + half * 4
                                global_b_idx = global_r * 8 + global_k
                                b_eff_idx = r * 2 + c
                                V[global_b_idx, col] += coeff_b * V2[b_eff_idx, m_s]
                    
                    # W contribution
                    for coeff_c, c_block_idx in C_specs[k_p]:
                        bi, bj = divmod(c_block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_i = bi * 2 + r
                                global_k = bj * 2 + c + half * 4
                                global_w_idx = global_k * 4 + global_i
                                p_eff_idx = r * 2 + c
                                W[global_w_idx, col] += coeff_c * W2[p_eff_idx, m_s]
                    
                    col += 1
        
        return U, V, W

    def build_simple_kronecker():
        """Simple Kronecker/product decomposition. Rank = n*m*p = 4*4*8=128."""
        rank = n * m * p  # 128
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        col = 0
        for i in range(n):
            for r in range(m):
                for k in range(p):
                    U[i * 4 + r, col] = 1.0
                    V[r * 8 + k, col] = 1.0
                    W[k * 4 + i, col] = 1.0
                    col += 1
        return U, V, W

    def try_reduce_rank(U, V, W, max_iterations=50):
        """Try to reduce rank by finding and removing linearly dependent components.
        
        Strategy: Build a matrix where each row represents one rank-1 component,
        with entries being the "signature" of that component across U, V, W.
        If two components are linearly dependent (their contributions can be
        combined), we can merge them.
        
        More practically: check if any column is a linear combination of others
        in the tensor sense. This is complex, so we use a heuristic:
        check for zero columns, then try greedy elimination.
        """
        cur_rank = U.shape[1]
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        
        # First, remove exactly zero columns
        keep = []
        for c in range(cur_rank):
            if not (np.all(U_cur[:, c] == 0) and np.all(V_cur[:, c] == 0) and np.all(W_cur[:, c] == 0)):
                keep.append(c)
        if len(keep) < cur_rank:
            U_cur = U_cur[:, keep]
            V_cur = V_cur[:, keep]
            W_cur = W_cur[:, keep]
            cur_rank = len(keep)
        
        # Try to find components that are scalar multiples of each other
        # in a way that allows merging. This is heuristic.
        # For each pair, check if U[:,i] = alpha * U[:,j], V[:,i] = beta * V[:,j], W[:,i] = gamma * W[:,j]
        # and if the product alpha*beta*gamma = 1 (so they're truly redundant)
        
        # Actually, for tensor rank reduction, we need:
        # U[:,i]*V[:,i]*W[:,i] = U[:,j]*V[:,j]*W[:,j] (element-wise in the tensor sense)
        # This means U[:,i] = alpha*U[:,j], V[:,i] = beta*V[:,j], W[:,i] = (1/(alpha*beta))*W[:,j]
        
        # This is expensive to check for all pairs. Let's try a simpler approach:
        # Build the "flattened" tensor contribution for each component and check rank.
        
        # Alternative: use the fact that if we can find a linear dependency
        # in the U columns that's mirrored in V and W appropriately, we can reduce.
        
        # For now, just return what we have after zero removal.
        return U_cur, V_cur, W_cur

    # Build candidates
    U1, V1, W1 = build_block_strassen()
    U2, V2, W2 = build_simple_kronecker()
    
    # Verify block Strassen
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    if verify_fast(U1, V1, W1, tol=1e-9):
        best_rank = 98
        best_U, best_V, best_W = U1.copy(), V1.copy(), W1.copy()
    
    # Fallback to Kronecker if Strassen failed
    if best_U is None:
        if verify_fast(U2, V2, W2, tol=1e-9):
            best_rank = 128
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
    
    # Try to reduce rank (heuristic, likely no gain for Strassen)
    if best_U is not None and time.time() < deadline - 10:
        U_reduced, V_reduced, W_reduced = try_reduce_rank(best_U, best_V, best_W)
        new_rank = U_reduced.shape[1]
        if new_rank < best_rank and verify_fast(U_reduced, V_reduced, W_reduced, tol=1e-9):
            best_rank = new_rank
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Final safety: if still no valid result, use Kronecker
    if best_U is None:
        best_U, best_V, best_W = U2, V2, W2
        best_rank = 128
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
