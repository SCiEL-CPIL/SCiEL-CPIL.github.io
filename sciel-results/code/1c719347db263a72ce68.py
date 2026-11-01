import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Helper to check if a decomposition is valid and roundable
    def check_and_round(U, V, W):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Helper to perform ALS for a given rank
    def try_rank(R, max_iters=100, n_restarts=10):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart * 1000 + R)
            
            # Initialize with small random values
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Update W
                # For each c, solve v_ab . w_c = T_abc for all (a,b)
                # This is a linear system for W.
                # Let's build the system A w = b
                # w is vector of size pn*R
                # Each non-zero (a,b,c) gives an equation:
                # sum_r U[ar] V[br] W[cr] = T[abc]
                # This can be written as:
                # sum_r (U[ar] V[br]) * W[cr] = 1
                # Let coeff_r = U[ar] V[br]
                # The term involving W is sum_r coeff_r * W[cr]
                # This is a dot product of coeff (length R) and W[:, c] (length R)? No.
                # W is (pn, R). W[cr] is element (c, r).
                # So for fixed a,b,c, the equation is:
                # sum_r (U[ar] * V[br]) * W[c, r] = 1
                # This is linear in W.
                
                # We can solve for W using least squares over all non-zero entries.
                # Number of equations: 24
                # Number of unknowns: pn * R = 12 * R
                # If R=20, 240 unknowns. 24 equations. Underdetermined.
                # We want the minimum norm solution or just any solution.
                # np.linalg.lstsq gives min norm.
                
                A_mat = np.zeros((24, pn * R))
                b_vec = np.zeros(24)
                
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    # Coefficient for W[c, r]
                                    A_mat[idx, c * R + r] = U[a, r] * V[b, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                
                # Solve A_mat * W_flat = b_vec
                try:
                    W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W = W_flat.reshape(pn, R)
                except:
                    continue

                # Update U
                # Equation: sum_r U[ar] (V[br] W[cr]) = 1
                A_mat = np.zeros((24, nm * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, a * R + r] = V[b, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                
                try:
                    U_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U = U_flat.reshape(nm, R)
                except:
                    continue

                # Update V
                # Equation: sum_r (U[ar] W[cr]) V[br] = 1
                A_mat = np.zeros((24, mp * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, b * R + r] = U[a, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                
                try:
                    V_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V = V_flat.reshape(mp, R)
                except:
                    continue
                
                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_calc - T)
                if err < 1e-8:
                    result = check_and_round(U, V, W)
                    if result is not None:
                        return result

        return None

    # Try to find lower ranks
    # Start from 20 down to 15
    for R in range(20, 14, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R, max_iters=50, n_restarts=5)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 21 (Strassen based)
    # Strassen for 2x2x2 has rank 7.
    # We have 3 blocks of 2x2x2 (for k=0,1; k=2,3; k=4,5).
    # Total rank 21.
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U[l, r] = step['u'][l]
                k0 = 2 * g
                k1 = 2 * g + 1
                # b index: 6*j + k
                # l=0 -> j=0, k=k0
                # l=1 -> j=0, k=k1
                # l=2 -> j=1, k=k0
                # l=3 -> j=1, k=k1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V[b_map[l], r] = step['v'][l]
                # c index: 2*k + i
                # l=0 -> i=0, k=k0 -> c=2*k0+0
                # l=1 -> i=0, k=k1 -> c=2*k1+0
                # l=2 -> i=1, k=k0 -> c=2*k0+1
                # l=3 -> i=1, k=k1 -> c=2*k1+1
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W[c_map[l], r] = step['w'][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
