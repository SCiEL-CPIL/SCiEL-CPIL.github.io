import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 5]

def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    n, m, p = DIMENSIONS
    
    # Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T_target[i,j,k] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        T_calc = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        return np.allclose(T_calc, T_target, atol=tol)

    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    def find_decomposition(rank, max_iters=200, time_limit=10.0, init_U=None, init_V=None, init_W=None):
        if time.time() > start_time + time_limit:
            return None, None, None, False
            
        if init_U is None:
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
        else:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
            
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        best_error = np.inf
        best_U, best_V, best_W = U, V, W
        
        for it in range(max_iters):
            if time.time() > start_time + time_limit:
                break
                
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            
            # Update U
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(V[:, r], W[:, r])
                Num = np.einsum('ijk,jk->i', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    U[:, r] = Num / Den
            
            # Update V
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(U[:, r], W[:, r])
                Num = np.einsum('ijk,ik->j', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    V[:, r] = Num / Den

            # Update W
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(U[:, r], V[:, r])
                Num = np.einsum('ijk,ij->k', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    W[:, r] = Num / Den
            
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_calc - T_target)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-8:
                break

        U_snap = snap_to_half_int(best_U)
        V_snap = snap_to_half_int(best_V)
        W_snap = snap_to_half_int(best_W)
        
        if check_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
            return U_snap, V_snap, W_snap, True
        
        return best_U, best_V, best_W, False

    # Try rank 17
    # Strategy: Random restarts and perturbations of rank 18
    # First, get a rank 18 solution to use as a base for perturbation
    
    # Hardcoded Rank 18
    rank18_U = np.zeros((n*m, 18))
    rank18_V = np.zeros((m*p, 18))
    rank18_W = np.zeros((p*n, 18))
    
    def add_strassen_block(U, V, W, t_offset, k_start):
        U[0][t_offset] = 1
        U[3][t_offset] = 1
        U[2][t_offset+1] = 1
        U[3][t_offset+1] = 1
        U[0][t_offset+2] = 1
        U[3][t_offset+3] = 1
        U[0][t_offset+4] = 1
        U[1][t_offset+4] = 1
        U[2][t_offset+5] = 1
        U[0][t_offset+5] = -1
        U[1][t_offset+6] = 1
        U[3][t_offset+6] = -1
        
        k0 = k_start
        k1 = k_start + 1
        V[k0][t_offset] = 1
        V[p+k1][t_offset] = 1
        V[k0][t_offset+1] = 1
        V[k1][t_offset+2] = 1
        V[p+k1][t_offset+2] = -1
        V[p+k0][t_offset+3] = 1
        V[k0][t_offset+3] = -1
        V[p+k1][t_offset+4] = 1
        V[k0][t_offset+5] = 1
        V[k1][t_offset+5] = 1
        V[p+k0][t_offset+6] = 1
        V[p+k1][t_offset+6] = 1
        
        W[k0*n][t_offset] = 1
        W[k1*n+1][t_offset] = 1
        W[k0*n+1][t_offset+1] = 1
        W[k1*n+1][t_offset+1] = -1
        W[k1*n][t_offset+2] = 1
        W[k1*n+1][t_offset+2] = 1
        W[k0*n][t_offset+3] = 1
        W[k0*n+1][t_offset+3] = 1
        W[k0*n][t_offset+4] = -1
        W[k1*n][t_offset+4] = 1
        W[k1*n+1][t_offset+5] = 1
        W[k0*n][t_offset+6] = 1

    add_strassen_block(rank18_U, rank18_V, rank18_W, 0, 0)
    add_strassen_block(rank18_U, rank18_V, rank18_W, 7, 2)
    
    t = 14
    rank18_U[0][t] = 1
    rank18_V[0*p + 4][t] = 1
    rank18_W[4*n + 0][t] = 1
    t += 1
    rank18_U[1][t] = 1
    rank18_V[1*p + 4][t] = 1
    rank18_W[4*n + 0][t] = 1
    t += 1
    rank18_U[2][t] = 1
    rank18_V[0*p + 4][t] = 1
    rank18_W[4*n + 1][t] = 1
    t += 1
    rank18_U[3][t] = 1
    rank18_V[1*p + 4][t] = 1
    rank18_W[4*n + 1][t] = 1

    # Verify rank 18
    if check_decomposition(rank18_U, rank18_V, rank18_W):
        pass # It is valid

    # Search for rank 17
    # We try to drop one term from rank 18 and re-optimize
    # Or just random restarts
    num_restarts = 20
    found_rank17 = False
    best_rank17_solution = None
    
    for i in range(num_restarts):
        if time.time() > deadline - margin:
            break
            
        # Initialization strategy:
        # 1. Random
        # 2. Perturbed Rank 18 (drop one column, add noise)
        if i < 10:
            # Perturb Rank 18
            U_init = rank18_U[:, :17].copy()
            V_init = rank18_V[:, :17].copy()
            W_init = rank18_W[:, :17].copy()
            # Add noise
            U_init += 0.1 * np.random.randn(n*m, 17)
            V_init += 0.1 * np.random.randn(m*p, 17)
            W_init += 0.1 * np.random.randn(p*n, 17)
            U, V, W, success = find_decomposition(17, max_iters=100, time_limit=5.0, init_U=U_init, init_V=V_init, init_W=W_init)
        else:
            U, V, W, success = find_decomposition(17, max_iters=100, time_limit=5.0)
            
        if success:
            found_rank17 = True
            best_rank17_solution = (U, V, W)
            break

    if found_rank17:
        U, V, W = best_rank17_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 18
        return {"U": rank18_U.tolist(), "V": rank18_V.tolist(), "W": rank18_W.tolist()}
