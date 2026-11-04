import numpy as np
import time
import random
import scipy.optimize as so

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def find_3x3x3(rank):
        # U, V, W are 9 x rank
        # Reshape to 3x3xrank
        def error(x):
            U = x[:9*rank].reshape(3, 3, rank)
            V = x[9*rank:18*rank].reshape(3, 3, rank)
            W = x[18*rank:27*rank].reshape(3, 3, rank)
            
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            T = np.zeros((3, 3, 3))
            for c in range(rank):
                # u_c is 3x3, v_c is 3x3, w_c is 3x3
                # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is element-wise product of U_c, V_c (transposed? no, V[j,k]), W_c (transposed? W[k,i] -> W.T[i,k])
                # Let's check indices:
                # U[i,j,c]
                # V[j,k,c]
                # W[k,i,c]
                # So for fixed c, we multiply U_c[i,j] * V_c[j,k] * W_c[k,i].
                # This is the trace of U_c * V_c * W_c? No.
                # It is the sum over c of (U_c \circ V_c \circ W_c^T?)
                # Let's just use einsum or loops for safety and clarity, it's small.
                # T += np.einsum('ij,jk,ki->ijk', U[:,:,c], V[:,:,c], W[:,:,c])
                # But W is indexed W[k,i], so W[:,:,c] is W[k,i].
                # So we need W_c[k,i].
                # Let Wc = W[:,:,c]. Then Wc[k,i] is Wc.T[i,k]? No, Wc[k,i] is just Wc.
                # Wait, W is stored as W[k*n+i, c]. Reshaped to (3,3,rank) -> W[k,i,c].
                # So W[:,:,c] is W[k,i].
                # So we need U[i,j] * V[j,k] * W[k,i].
                # This is exactly np.einsum('ij,jk,ki->ijk', U[:,:,c], V[:,:,c], W[:,:,c])
                T += np.einsum('ij,jk,ki->ijk', U[:,:,c], V[:,:,c], W[:,:,c])
            
            # Target is 1.0 everywhere
            return np.sum((T - 1.0)**2)

        # Use multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                return None
                
            x0 = np.random.uniform(-1, 1, 27*rank)
            
            try:
                res = so.minimize(error, x0, method='Powell', options={'maxiter': 500, 'xtol': 1e-8, 'disp': False})
            except Exception:
                continue
                
            x = res.x
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            valid = True
            T = np.zeros((3, 3, 3))
            Ur = U_r.reshape(3, 3, rank)
            Vr = V_r.reshape(3, 3, rank)
            Wr = W_r.reshape(3, 3, rank)
            for c in range(rank):
                T += np.einsum('ij,jk,ki->ijk', Ur[:,:,c], Vr[:,:,c], Wr[:,:,c])
            
            if not np.allclose(T, 1.0, atol=1e-9):
                continue
            
            return U_r, V_r, W_r
        
        return None

    # Try to find 3x3x3 decomposition
    for r in range(23, 27):
        if time.time() > deadline:
            break
        
        found = False
        # More restarts for lower ranks
        max_restarts = 50 if r == 23 else 30
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Call find_3x3x3 which does its own restarts? 
            # No, let's modify find_3x3x3 to take a single restart or just call it once?
            # The current find_3x3x3 does 20 restarts internally.
            # Let's just call it once per loop iteration, but increase internal restarts?
            # Or just call find_3x3x3(r) which does 20 restarts.
            # If it fails, try again.
            
            res = find_3x3x3(r)
            if res is not None:
                U1, V1, W1 = res
                # Construct 3x3x6
                total_rank = 2 * r
                U_new = np.zeros((9, total_rank))
                V_new = np.zeros((18, total_rank))
                W_new = np.zeros((18, total_rank))
                
                # Block 1: k=0,1,2
                U_new[:, 0:r] = U1
                V_new[0:9, 0:r] = V1
                W_new[0:9, 0:r] = W1
                
                # Block 2: k=3,4,5
                U_new[:, r:2*r] = U1
                V_new[9:18, r:2*r] = V1
                W_new[9:18, r:2*r] = W1
                
                # Verify final solution
                valid = True
                # T[i,j,k] for 3x3x6
                # U: 9 x 2r
                # V: 18 x 2r
                # W: 18 x 2r
                # Indices: U[i*3+j], V[j*6+k], W[k*3+i]
                # k in 0..5
                # If k in 0..2, V uses first 9 rows, W uses first 9 rows.
                # If k in 3..5, V uses last 9 rows, W uses last 9 rows.
                # U is same for both blocks.
                
                # We can verify by checking both blocks separately since they are decoupled in V and W.
                # Block 1: k=0,1,2
                T1 = np.zeros((3, 3, 3))
                Ur = U_new[:, 0:r].reshape(3, 3, r)
                Vr1 = V_new[0:9, 0:r].reshape(3, 3, r)
                Wr1 = W_new[0:9, 0:r].reshape(3, 3, r)
                for c in range(r):
                    T1 += np.einsum('ij,jk,ki->ijk', Ur[:,:,c], Vr1[:,:,c], Wr1[:,:,c])
                if not np.allclose(T1, 1.0, atol=1e-9):
                    valid = False
                
                if valid:
                    # Block 2: k=3,4,5
                    T2 = np.zeros((3, 3, 3))
                    Vr2 = V_new[9:18, r:2*r].reshape(3, 3, r)
                    Wr2 = W_new[9:18, r:2*r].reshape(3, 3, r)
                    for c in range(r):
                        T2 += np.einsum('ij,jk,ki->ijk', Ur[:,:,c], Vr2[:,:,c], Wr2[:,:,c])
                    if not np.allclose(T2, 1.0, atol=1e-9):
                        valid = False
                
                if valid:
                    best_rank = total_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found = True
                    break
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
