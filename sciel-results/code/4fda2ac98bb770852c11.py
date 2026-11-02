import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor T
    # T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a, b, c] = 1 if i=i', j=j', k=k', else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def snap_half_int(arr):
        # Snap to nearest 0.5
        return np.round(arr * 2) / 2.0

    def verify(U, V, W, tol=1e-8):
        # Check if sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        # Efficient check
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_comp - T)) < tol

    def solve_als(U_init, V_init, W_init, R, max_iters=100):
        # U, V, W are (nm, R), (mp, R), (pn, R)
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_cost = np.inf
        for it in range(max_iters):
            # Solve for U
            # For each a, A_a u_a^T = T[a, :, :].flatten()
            # A_a[(b, c), r] = V[b, r] W[c, r]
            for a in range(nm):
                A_a = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
                t_a = T[a, :, :].reshape(-1)
                # Solve A_a x = t_a
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A_a, t_a, rcond=None)
            
            # Solve for V
            for b in range(mp):
                B_b = (U[:, None, :] * W[None, :, :]).reshape(-1, R)
                t_b = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank, sv = np.linalg.lstsq(B_b, t_b, rcond=None)
            
            # Solve for W
            for c in range(pn):
                C_c = (U[:, None, :] * V[None, :, :]).reshape(-1, R)
                t_c = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank, sv = np.linalg.lstsq(C_c, t_c, rcond=None)
            
            # Calculate cost
            T_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            cost = np.sum((T_comp - T)**2)
            
            if cost < 1e-10:
                break
            if abs(prev_cost - cost) < 1e-12:
                break
            prev_cost = cost
            
        # Try snapping
        U_s = snap_half_int(U)
        V_s = snap_half_int(V)
        W_s = snap_half_int(W)
        
        if verify(U_s, V_s, W_s, tol=1e-8):
            return U_s, V_s, W_s, cost
            
        return None, None, None, cost

    def generate_initialization(R, seed):
        np.random.seed(seed)
        
        # Strategy: Geometric invariance / Random linear reparametrization
        # We start with a random base tensor and then apply random linear transformations
        # to the factor spaces. This helps escape local minima associated with the
        # schoolbook structure or simple random starts.
        
        # Random base
        U_base = np.random.randn(nm, R)
        V_base = np.random.randn(mp, R)
        W_base = np.random.randn(pn, R)
        
        # Apply random linear transformations (reparametrizations)
        # U_new = U_base @ P, V_new = V_base @ Q, W_new = W_base @ S
        # where P, Q, S are invertible matrices such that sum_r U_base[:,r] P_rk V_base[:,r] Q_rk W_base[:,r] S_rk 
        # is equivalent to the original tensor up to a transformation.
        # Actually, the tensor T = sum_r u_r v_r w_r.
        # If we let U' = U P, V' = V Q, W' = W S, then T' = sum_k (sum_r U[:,r] P[r,k]) (sum_s V[:,s] Q[s,k]) (sum_t W[:,t] S[t,k])
        # This is NOT the same tensor unless P, Q, S are related.
        # The invariance is: sum_r u_r v_r w_r = sum_k (sum_r u_r P[r,k]) (sum_s v_s Q[s,k]) (sum_t w_t S[t,k])
        # This holds if P Q S = I (elementwise product? No, matrix product).
        # Let's just use random orthogonal matrices or random invertible matrices and see if ALS can recover the structure.
        # A better approach: Initialize U, V, W with random values, but ensure they are not all zero.
        # The "geometric invariant" idea suggests that the manifold of rank-R tensors is a quotient space.
        # We can sample points on this manifold by taking a generic tensor and projecting?
        # For now, let's stick to random initialization but with a twist: 
        # We initialize with a mix of "schoolbook-like" and random components.
        
        # Alternative: Use the schoolbook decomposition as a starting point but perturb it.
        # This might be better for low ranks?
        
        # Let's try a different initialization:
        # 1. Create a random rank-R tensor.
        # 2. Project it onto the variety of rank-R tensors? (Hard)
        # 3. Just use random initialization but with multiple scales.
        
        # Let's try initializing with random orthogonal matrices scaled.
        U_init = np.random.randn(nm, R)
        V_init = np.random.randn(mp, R)
        W_init = np.random.randn(pn, R)
        
        # Scale to avoid numerical issues
        U_init /= np.linalg.norm(U_init)
        V_init /= np.linalg.norm(V_init)
        W_init /= np.linalg.norm(W_init)
        
        return U_init, V_init, W_init

    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 20 to 23
    for R in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on remaining time
        # R=20 is the most important. Give it more restarts.
        if R == 20:
            num_restarts = 50
        elif R == 21:
            num_restarts = 20
        else:
            num_restarts = 10
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Use the new initialization strategy
            U_init, V_init, W_init = generate_initialization(R, seed=42 + R * 100 + restart)
            
            U, V, W, cost = solve_als(U_init, V_init, W_init, R, max_iters=200)
            if U is not None:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook rank 24
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
