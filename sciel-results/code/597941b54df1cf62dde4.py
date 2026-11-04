"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(mat):
        return mat.tolist()

    def verify(result, atol=1e-8):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def als_decompose_subtensor(target_t, rank, max_iter=200, tol=1e-14, num_restarts=1, time_budget=None):
        """ALS decomposition of a given target tensor with multiple restarts."""
        Nt, Mt, Pt = target_t.shape
        best_err = np.inf
        best_UVW = None
        
        for restart in range(num_restarts):
            if time_budget is not None and time_budget() < 15:
                break
            
            # Different initialization strategies
            if restart % 4 == 0:
                U = np.random.randn(Nt, rank) * 0.1
                V = np.random.randn(Mt, rank) * 0.1
                W = np.random.randn(Pt, rank) * 0.1
            elif restart % 4 == 1:
                U = np.random.randn(Nt, rank) * 0.01
                V = np.random.randn(Mt, rank) * 0.01
                W = np.random.randn(Pt, rank) * 0.01
            elif restart % 4 == 2:
                U = np.random.randn(Nt, rank)
                V = np.random.randn(Mt, rank)
                W = np.random.randn(Pt, rank)
                U /= (np.linalg.norm(U) + 1e-10)
                V /= (np.linalg.norm(V) + 1e-10)
                W /= (np.linalg.norm(W) + 1e-10)
            else:
                # Sparse initialization
                U = np.zeros((Nt, rank))
                V = np.zeros((Mt, rank))
                W = np.zeros((Pt, rank))
                for r in range(rank):
                    U[np.random.randint(0, Nt), r] = 1.0
                    V[np.random.randint(0, Mt), r] = 1.0
                    W[np.random.randint(0, Pt), r] = 1.0
            
            prev_err = np.inf
            for iteration in range(max_iter):
                if time_budget is not None and time_budget() < 15:
                    break
                
                # Update U
                H_U = np.einsum('br,cr->bcr', V, W).reshape(Mt * Pt, rank)
                target_flat = target_t.reshape(Nt, Mt * Pt)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                H_V = np.einsum('ar,cr->acr', U, W).reshape(Nt * Pt, rank)
                target_v = target_t.transpose(1, 0, 2).reshape(Mt, Nt * Pt)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                H_W = np.einsum('ar,br->abr', U, V).reshape(Nt * Mt, rank)
                target_w = target_t.transpose(2, 0, 1).reshape(Pt, Nt * Mt)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                T = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - target_t)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            # Early exit if we found exact solution
            if best_err < 1e-12:
                break
        
        if best_UVW is not None:
            return best_UVW, best_err
        return None, np.inf

    def build_subtensor_j0to3():
        """Build the sub-tensor for j=0,1,2,3 (4x4x6 part)."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    sub[i * 5 + j, j * 6 + k, k * 4 + i] = 1.0
        return sub
    
    def build_subtensor_j4():
        """Build the sub-tensor for j=4 (4x1x6 part)."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for k in range(6):
                sub[i * 5 + 4, 4 * 6 + k, k * 4 + i] = 1.0
        return sub

    # Strategy 1: Block partitioning j=0..3 and j=4
    if time_remaining() > 60:
        sub1 = build_subtensor_j0to3()
        
        # Try rank 12 for the 4x4x6 part
        U1, V1, W1 = None, None, None
        best_err1 = np.inf
        
        for trial in range(25):
            if time_remaining() < 30:
                break
            result1 = als_decompose_subtensor(sub1, 12, max_iter=150, num_restarts=1, time_budget=time_remaining)
            if result1[0] is not None:
                U1, V1, W1 = result1[0]
                err1 = result1[1]
                if err1 < best_err1:
                    best_err1 = err1
                if err1 < 1e-12:
                    break
        
        if U1 is not None and best_err1 < 1e-10:
            # Part 2: 4x1x6 tensor -> rank 24 (schoolbook)
            U2 = np.zeros((N, 24))
            V2 = np.zeros((M, 24))
            W2 = np.zeros((P, 24))
            col = 0
            for i in range(4):
                for k in range(6):
                    a = i * 5 + 4
                    b = 4 * 6 + k
                    c = k * 4 + i
                    U2[a, col] = 1
                    V2[b, col] = 1
                    W2[c, col] = 1
                    col += 1
            
            # Combine
            U_combined = np.hstack([U1, U2])
            V_combined = np.hstack([V1, V2])
            W_combined = np.hstack([W1, W2])
            
            result_combined = {"U": U_combined, "V": V_combined, "W": W_combined}
            if verify(result_combined, atol=1e-8):
                rank_combined = U_combined.shape[1]
                if rank_combined < best_rank:
                    best = result_combined
                    best_rank = rank_combined
                    print(f"Block partitioning found rank {rank_combined}")

    # Strategy 2: Try full tensor ALS with various ranks
    for rank_try in [30, 28, 26, 24]:
        if time_remaining() < 25:
            break
        
        U_full, V_full, W_full = None, None, None
        best_err_full = np.inf
        
        for trial in range(15):
            if time_remaining() < 20:
                break
            result_full = als_decompose_subtensor(target, rank_try, max_iter=100, num_restarts=1, time_budget=time_remaining)
            if result_full[0] is not None:
                U_full, V_full, W_full = result_full[0]
                err_full = result_full[1]
                if err_full < best_err_full:
                    best_err_full = err_full
                if err_full < 1e-12:
                    break
        
        if U_full is not None and best_err_full < 1e-10:
            result_full = {"U": U_full, "V": V_full, "W": W_full}
            if verify(result_full, atol=1e-8):
                if rank_try < best_rank:
                    best = result_full
                    best_rank = rank_try
                    print(f"Full ALS found rank {rank_try}")
                    break

    # Strategy 3: Try block partitioning with j=0..2 and j=3..4
    if time_remaining() > 40:
        sub_a = np.zeros((N, M, P))
        for i in range(4):
            for j in range(3):
                for k in range(6):
                    sub_a[i * 5 + j, j * 6 + k, k * 4 + i] = 1.0
        
        sub_b = np.zeros((N, M, P))
        for i in range(4):
            for j in range(3, 5):
                for k in range(6):
                    sub_b[i * 5 + j, j * 6 + k, k * 4 + i] = 1.0
        
        # Part A: 4x3x6, try rank 10
        Ua, Va, Wa = None, None, None
        best_err_a = np.inf
        for trial in range(15):
            if time_remaining() < 25:
                break
            result_a = als_decompose_subtensor(sub_a, 10, max_iter=100, num_restarts=1, time_budget=time_remaining)
            if result_a[0] is not None:
                Ua, Va, Wa = result_a[0]
                err_a = result_a[1]
                if err_a < best_err_a:
                    best_err_a = err_a
                if err_a < 1e-12:
                    break
        
        if Ua is not None and best_err_a < 1e-10:
            # Part B: 4x2x6, try rank 12
            Ub, Vb, Wb = None, None, None
            best_err_b = np.inf
            for trial in range(15):
                if time_remaining() < 20:
                    break
                result_b = als_decompose_subtensor(sub_b, 12, max_iter=100, num_restarts=1, time_budget=time_remaining)
                if result_b[0] is not None:
                    Ub, Vb, Wb = result_b[0]
                    err_b = result_b[1]
                    if err_b < best_err_b:
                        best_err_b = err_b
                    if err_b < 1e-12:
                        break
            
            if Ub is not None and best_err_b < 1e-10:
                U_comb = np.hstack([Ua, Ub])
                V_comb = np.hstack([Va, Vb])
                W_comb = np.hstack([Wa, Wb])
                result_comb = {"U": U_comb, "V": V_comb, "W": W_comb}
                if verify(result_comb, atol=1e-8):
                    rank_comb = U_comb.shape[1]
                    if rank_comb < best_rank:
                        best = result_comb
                        best_rank = rank_comb
                        print(f"3+2 block partitioning found rank {rank_comb}")

    # Strategy 4: Final polish - try to reduce rank further
    if best_rank < n * m * p and time_remaining() > 30:
        for rank_try in range(max(20, best_rank - 5), best_rank):
            if time_remaining() < 15:
                break
            result_polish = als_decompose_subtensor(target, rank_try, max_iter=200, num_restarts=5, time_budget=time_remaining)
            if result_polish[0] is not None:
                U_p, V_p, W_p = result_polish[0]
                err_p = result_polish[1]
                if err_p < 1e-10:
                    result_polish = {"U": U_p, "V": V_p, "W": W_p}
                    if verify(result_polish, atol=1e-8):
                        best = result_polish
                        best_rank = rank_try
                        print(f"Polish found rank {rank_try}")
                        break

    # Final verification
    if not verify(best, atol=1e-7):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
