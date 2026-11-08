import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    
    # Build target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(k',i') and i==i', j==j', k==k'
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1.0

    def get_rank26():
        rank = 26
        U = np.zeros((9, rank), dtype=np.float64)
        V = np.zeros((9, rank), dtype=np.float64)
        W = np.zeros((9, rank), dtype=np.float64)
        
        col = 0
        # Strassen for top-left 2x2
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
        
        # a b^T
        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
        
        # A2 b
        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
        
        # a b3
        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
        
        # a^T B2
        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
        
        # a3 b^T
        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
        
        # a^T b
        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
        
        # a3 b3
        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        
        return U, V, W

    def check_decomposition(U, V, W):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-6)

    def rank_of_decomposition(U, V, W):
        nonzero = np.any(np.abs(U) > 1e-9, axis=0) & np.any(np.abs(V) > 1e-9, axis=0) & np.any(np.abs(W) > 1e-9, axis=0)
        return np.sum(nonzero)

    U0, V0, W0 = get_rank26()
    best_rank = 26
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    if check_decomposition(U0, V0, W0):
        best_rank = rank_of_decomposition(U0, V0, W0)

    # Try to improve using real-valued optimization
    # We will try to find a decomposition of rank r < best_rank
    # by starting with a random initialization or a perturbation of the known one
    # and using Nelder-Mead or similar to minimize error.
    
    def optimize_decomposition(r, init_U, init_V, init_W, max_iter=2000):
        # Flatten parameters
        # We only optimize non-zero columns? No, we optimize all r columns.
        # To encourage sparsity, we can add a L1 penalty, but we need EXACT decomposition.
        # So we just minimize squared error.
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            size = 9 * r
            Ux = x[:size].reshape(9, r)
            Vx = x[size:2*size].reshape(9, r)
            Wx = x[2*size:3*size].reshape(9, r)
            return Ux, Vx, Wx
        
        def error(x):
            Ux, Vx, Wx = unflatten(x)
            recon = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
            return np.sum((recon - T)**2)
        
        x0 = flatten(init_U, init_V, init_W)
        
        # Use scipy.optimize.minimize if available, otherwise simple gradient descent
        try:
            from scipy.optimize import minimize
            res = minimize(error, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
            x_best = res.x
            Ux, Vx, Wx = unflatten(x_best)
            err = error(x_best)
            if err < 1e-6:
                return Ux, Vx, Wx, err
        except:
            pass
            
        # Fallback: Simple coordinate descent
        Ux, Vx, Wx = unflatten(x0)
        current_err = error(flatten(Ux, Vx, Wx))
        
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            # Pick random element
            mat_choice = np.random.randint(3)
            row = np.random.randint(9)
            col = np.random.randint(r)
            
            # Try small perturbation
            delta = 0.1 * np.random.choice([-1, 1])
            
            if mat_choice == 0:
                Ux[row, col] += delta
                new_err = error(flatten(Ux, Vx, Wx))
                if new_err < current_err:
                    current_err = new_err
                else:
                    Ux[row, col] -= delta
            elif mat_choice == 1:
                Vx[row, col] += delta
                new_err = error(flatten(Ux, Vx, Wx))
                if new_err < current_err:
                    current_err = new_err
                else:
                    Vx[row, col] -= delta
            else:
                Wx[row, col] += delta
                new_err = error(flatten(Ux, Vx, Wx))
                if new_err < current_err:
                    current_err = new_err
                else:
                    Wx[row, col] -= delta
                    
            if current_err < 1e-6:
                return Ux, Vx, Wx, current_err
                
        return Ux, Vx, Wx, current_err

    # Try to find a rank 23 decomposition
    # Initialize with random values or a subset of the rank 26 solution
    rng = np.random.RandomState(42)
    
    for r in range(23, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random integers in {-1, 0, 1}
            U_t = rng.randint(-1, 2, size=(9, r)).astype(np.float64)
            V_t = rng.randint(-1, 2, size=(9, r)).astype(np.float64)
            W_t = rng.randint(-1, 2, size=(9, r)).astype(np.float64)
            
            # Optimization
            U_opt, V_opt, W_opt, err = optimize_decomposition(r, U_t, V_t, W_opt if False else W_t, max_iter=500)
            
            if err < 1e-6:
                # Try to snap to integers
                U_snap = np.round(U_opt)
                V_snap = np.round(V_opt)
                W_snap = np.round(W_opt)
                
                if check_decomposition(U_snap, V_snap, W_snap):
                    actual_rank = rank_of_decomposition(U_snap, V_snap, W_snap)
                    if actual_rank < best_rank:
                        best_rank = actual_rank
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        # If we found a better rank, we can try to go lower
                        # But let's just break and return for now to save time
                        break
        
        if best_rank == 23:
            break # 23 is known to be optimal or close to it for this size

    # Final polish: try to reduce rank of the best found solution
    # by checking if any column can be removed
    if time.time() < deadline - 5:
        # This is a heuristic: if a column is very small, it might be removable
        # But we need EXACT decomposition.
        pass

    def to_list(arr):
        return arr.tolist()
    
    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)
    
    recon = np.einsum('ir,jr,kr->ijk', np.array(U_final), np.array(V_final), np.array(W_final))
    if not np.allclose(recon, T):
        U_fb, V_fb, W_fb = get_rank26()
        U_final = to_list(U_fb)
        V_final = to_list(V_fb)
        W_final = to_list(W_fb)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
