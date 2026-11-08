import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def contract(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

    def is_exact(U, V, W, atol=1e-6):
        prod = contract(U, V, W)
        return np.allclose(prod, T, atol=atol)

    def to_output(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P2
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            # P3
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P4
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            # P5
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            # P6
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P7
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

    U46, V46, W46 = build_rank46()
    if not is_exact(U46, V46, W46):
        # Fallback to trivial rank 50
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
        return to_output(U, V, W)

    def try_lower_rank(target_rank):
        R = target_rank
        # Order of dropping columns: edge columns first (28-45), then strassen (0-27)
        drop_order = list(range(28, 46)) + list(range(0, 28))
        
        for drop_idx in drop_order:
            if time.time() > deadline - 15:
                break
            
            mask = [i for i in range(46) if i != drop_idx]
            U_init = U46[:, mask]
            V_init = V46[:, mask]
            W_init = W46[:, mask]
            
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Perturb
                noise = 0.05
                U_curr = U_init + np.random.randn(*U_init.shape) * noise
                V_curr = V_init + np.random.randn(*V_init.shape) * noise
                W_curr = W_init + np.random.randn(*W_init.shape) * noise
                
                params = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
                
                def objective(params):
                    U = params[:U_dim*R].reshape(U_dim, R)
                    V = params[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    W = params[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    prod = contract(U, V, W)
                    return np.sum((prod - T)**2)
                
                try:
                    res = opt.minimize(objective, params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                except Exception:
                    continue
                
                if res.fun < 1e-6:
                    U_best = res.x[:U_dim*R].reshape(U_dim, R)
                    V_best = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    W_best = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_best * 2) / 2
                    V_snap = np.round(V_best * 2) / 2
                    W_snap = np.round(W_best * 2) / 2
                    
                    if is_exact(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
        
        return None

    # Try rank 45
    res45 = try_lower_rank(45)
    if res45 is not None:
        return to_output(*res45)

    # Try rank 44? (Optional, if time permits)
    if time.time() < deadline - 20:
        res44 = try_lower_rank(44)
        if res44 is not None:
            return to_output(*res44)

    return to_output(U46, V46, W46)

# EVOLVE-BLOCK-END
