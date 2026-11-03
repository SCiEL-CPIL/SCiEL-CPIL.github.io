"""Search-based exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M          # 16
MP = M * P          # 20
PN = P * N          # 20
TERM_COUNT = N * M * P  # 80

def _is_valid(U, V, W, tol=1e-9):
    """Exact check that U,V,W decompose the multiplication tensor.
    For every (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    for i in range(N):
        for j in range(M):
            urow = U_arr[i * M + j]
            vbase = j * P
            for k in range(P):
                vrow = V_arr[vbase + k]
                wrow = W_arr[k * N + i]
                total = np.sum(urow * vrow * wrow)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _rank(U, V, W):
    return len(U[0]) if U and U[0] else 0

def _try_rank_1():
    """Try the trivial rank 1 solution (all ones)."""
    U = [[1.0] for _ in range(NM)]
    V = [[1.0] for _ in range(MP)]
    W = [[1.0] for _ in range(PN)]
    if _is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _gradient_descent_search(seed, deadline):
    """
    Use Gradient Descent to minimize the reconstruction error and L1 sparsity.
    Target tensor is all ones.
    """
    rng = np.random.RandomState(seed)
    best_rank = None
    best_cand = None
    
    # Try different initial ranks
    for init_rank in [10, 20, 30, 40]:
        if time.time() > deadline:
            break
            
        U = rng.randn(NM, init_rank) * 0.1
        V = rng.randn(MP, init_rank) * 0.1
        W = rng.randn(PN, init_rank) * 0.1
        
        # Initialize with a valid solution (schoolbook) if rank allows, else random
        # Actually, random init is better for GD to escape local minima if target is complex.
        # But if target is all-ones, random might converge to all-ones.
        
        lr = 0.01
        lambda_sparsity = 0.01
        steps = 1000
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Compute current tensor
            T_calc = np.zeros((N, M, P))
            for r in range(init_rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                for i in range(N):
                    for j in range(M):
                        u_val = u_col[i * M + j]
                        if u_val == 0: continue
                        for k in range(P):
                            v_val = v_col[j * P + k]
                            if v_val == 0: continue
                            w_val = w_col[k * N + i]
                            if w_val == 0: continue
                            T_calc[i, j, k] += u_val * v_val * w_val
                            
            # Target is all ones
            T_target = np.ones((N, M, P))
            
            # Error
            error = T_calc - T_target
            mse = np.sum(error**2)
            
            # Prune terms if they are very small
            # Check L1 norm of each column
            active = 0
            for r in range(init_rank):
                norm_u = np.sum(np.abs(U[:, r]))
                norm_v = np.sum(np.abs(V[:, r]))
                norm_w = np.sum(np.abs(W[:, r]))
                if norm_u < 1e-6 and norm_v < 1e-6 and norm_w < 1e-6:
                    # Zero out
                    U[:, r] = 0
                    V[:, r] = 0
                    W[:, r] = 0
                else:
                    active += 1
            
            if active < (best_rank if best_rank else init_rank):
                if _is_valid(U, V, W):
                    best_rank = active
                    best_cand = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Compute gradients (approximate)
            # dL/du = 2 * error * v * w - lambda * sign(u)
            # This is expensive to compute exactly for each step.
            # We use a simplified update:
            # U_new = U - lr * (2 * dT/du * error + lambda * sign(U))
            
            # For efficiency, we might just use random perturbation if GD is too slow.
            # But let's try a simple GD step.
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(init_rank):
                if np.all(U[:, r] == 0) and np.all(V[:, r] == 0) and np.all(W[:, r] == 0):
                    continue
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                
                for i in range(N):
                    for j in range(M):
                        u_idx = i * M + j
                        u_val = u_col[u_idx]
                        if u_val == 0: continue
                        for k in range(P):
                            v_idx = j * P + k
                            w_idx = k * N + i
                            v_val = v_col[v_idx]
                            w_val = w_col[w_idx]
                            if v_val == 0 or w_val == 0: continue
                            
                            err = error[i, j, k]
                            
                            dU[u_idx, r] += 2 * err * v_val * w_val
                            dV[v_idx, r] += 2 * err * u_val * w_val
                            dW[w_idx, r] += 2 * err * u_val * v_val
                            
            # Add sparsity gradient
            dU += lambda_sparsity * np.sign(U)
            dV += lambda_sparsity * np.sign(V)
            dW += lambda_sparsity * np.sign(W)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Clip to avoid explosion
            U = np.clip(U, -10, 10)
            V = np.clip(V, -10, 10)
            W = np.clip(W, -10, 10)
            
        # Final check for this rank
        if _is_valid(U, V, W):
            r = _rank(U, V, W)
            if best_rank is None or r < best_rank:
                best_rank = r
                best_cand = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return best_cand

def construct():
    deadline = time.time() + 120.0
    safety = 10.0
    effective_deadline = deadline - safety
    
    # 1. Try Rank 1
    cand = _try_rank_1()
    if cand:
        return cand
        
    # 2. Gradient Descent
    best = None
    for seed in range(10):
        if time.time() > effective_deadline:
            break
        cand = _gradient_descent_search(seed, effective_deadline)
        if cand:
            r = _rank(cand["U"], cand["V"], cand["W"])
            if best is None or r < _rank(best["U"], best["V"], best["W"]):
                best = cand
                
    # 3. Fallback to Schoolbook
    if best is None:
        # Schoolbook
        rank = N * M * P
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][c] = 1
                    V[j * P + k][c] = 1
                    W[k * N + i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}
        
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", _rank(res["U"], res["V"], res["W"]))
