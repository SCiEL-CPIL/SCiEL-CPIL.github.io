import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    def objective(params, U_dim, V_dim, W_dim, T):
        # params is flattened U, V, W
        # R is the number of columns
        # We need to know R to reshape
        # Pass R via closure or global? Better to use closure.
        pass

    def optimize_to_rank(U_init, V_init, W_init, target_rank, max_iter=200, lr=0.01):
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        # Flatten for scipy
        def loss_and_grad(params):
            U = params[0:U_dim*R].reshape(U_dim, R)
            V = params[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
            W = params[U_dim*R+V_dim*R:].reshape(W_dim, R)
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            grad = np.concatenate([gradU.flatten(), gradV.flatten(), gradW.flatten()])
            return loss, grad

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B for better convergence
        res = minimize(loss_and_grad, x0, jac=True, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        x_opt = res.x
        U_opt = x_opt[0:U_dim*R].reshape(U_dim, R)
        V_opt = x_opt[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
        W_opt = x_opt[U_dim*R+V_dim*R:].reshape(W_dim, R)
        
        return U_opt, V_opt, W_opt

    # First, verify rank-46 works
    U46, V46, W46 = build_rank46()
    if validate(U46, V46, W46):
        # Try to find rank-45 solution
        best_result = None
        
        # Try dropping each single column
        for col_to_zero in range(46):
            if time.time() > deadline - 10:
                break
            
            mask = [i for i in range(46) if i != col_to_zero]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            # Initialize with the values, no noise to preserve structure
            U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 45, max_iter=50)
            
            if U_opt is not None:
                valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if valid:
                    return {
                        "U": [[float(x) for x in row] for row in U_s],
                        "V": [[float(x) for x in row] for row in V_s],
                        "W": [[float(x) for x in row] for row in W_s]
                    }
        
        # If single column drop failed, try random pairs (Rank 44)
        # This is expensive, so only if time remains
        if time.time() < deadline - 30:
            import itertools
            col_pairs = list(itertools.combinations(range(46), 2))
            np.random.shuffle(col_pairs)
            for pair in col_pairs[:10]:
                if time.time() > deadline - 5:
                    break
                mask = [i for i in range(46) if i not in pair]
                U_test = U46[:, mask].copy()
                V_test = V46[:, mask].copy()
                W_test = W46[:, mask].copy()
                
                U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 44, max_iter=30)
                
                if U_opt is not None:
                    valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                    if valid:
                        return {
                            "U": [[float(x) for x in row] for row in U_s],
                            "V": [[float(x) for x in row] for row in V_s],
                            "W": [[float(x) for x in row] for row in W_s]
                        }

        # Return rank-46 solution
        return {
            "U": [[float(x) for x in row] for row in U46],
            "V": [[float(x) for x in row] for row in V46],
            "W": [[float(x) for x in row] for row in W46]
        }
    else:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
# EVOLVE-BLOCK-END
