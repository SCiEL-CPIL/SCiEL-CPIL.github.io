"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def to_list(M):
        """Convert numpy array to list of lists, trying to round to integers/half-integers"""
        L = []
        for row in M:
            L.append([])
            for val in row:
                v = float(val)
                # Check if close to integer
                if abs(v - round(v)) < 1e-6:
                    L[-1].append(int(round(v)))
                # Check if close to half-integer
                elif abs(v - (round(2*v)/2.0)) < 1e-6:
                    L[-1].append(round(2*v)/2.0)
                else:
                    L[-1].append(v)
        return L
    
    # Schoolbook decomposition (rank = n*m*p = 84)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    # Alternating minimization for rank-r decomposition
    def alt_minimize(U, V, W, T, max_iter=200, tol=1e-10):
        """Alternating minimization to find rank-r decomposition"""
        nm, r = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            # Fix V and W, solve for U
            # For each a: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is a linear system for each a
            # But it's not straightforward because V and W are fixed
            
            # Better approach: use gradient descent or normal equations
            # Let's use a simple approach: for each rank component, update one matrix at a time
            
            # Actually, let's use a different strategy: 
            # Fix two matrices, solve for the third using least squares
            
            # Fix V and W, solve for U
            # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # For fixed b,c: T[a,b,c] = sum_r U[a,r] * M_bc[r] where M_bc[r] = V[b,r]*W[c,r]
            # This gives a linear system for U[a,:] for each a
            
            # More efficient: vectorize
            # Reshape T to (nm, mp*pn)
            T_2d = T.reshape(nm, mp * pn)
            
            # Build matrix B where B[a, b*pn+c, r] = V[b,r] * W[c,r]
            # Then T_2d[a, :] = U[a, :] @ B[a, :, :]... no that's not right
            
            # Let me think differently. For fixed V and W:
            # We want to find U such that einsum('ar,br,cr->abc', U, V, W) = T
            # This is linear in U. We can solve it using the normal equations.
            
            # The Jacobian with respect to U is complex. Let's use a simpler approach:
            # Use scipy's least squares on the full system
            
            # Actually, for small problems, let's just use numpy lstsq
            # Flatten the constraint: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # This is a linear system in U. Let's build it.
            # Number of equations: nm * mp * pn = 12 * 28 * 21 = 7056
            # Number of unknowns: nm * r = 12 * r
            
            # For r < 84, this is an overdetermined system
            
            # Build the matrix A where A[(a,b,c), (a',r)] = 1 if a==a' else 0, times V[b,r]*W[c,r]
            # Actually: A[(a,b,c), (a',r)] = V[b,r]*W[c,r] if a==a' else 0
            
            # This is block diagonal with blocks of size (mp*pn) x r for each a
            
            # Let's solve it block by block
            U_new = np.zeros_like(U)
            for a in range(nm):
                # For this a, we have equations: for all b,c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Build the matrix M of shape (mp*pn, r)
                M = np.zeros((mp * pn, r))
                for b in range(mp):
                    for c in range(pn):
                        M[b * pn + c, :] = V[b, :] * W[c, :]
                
                # Solve M @ U[a,:] = T[a,:,:].flatten()
                rhs = T[a, :, :].flatten()
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    U_new[a, :] = sol
                except:
                    U_new[a, :] = U[a, :]
            
            U = U_new
            
            # Similarly fix U and W, solve for V
            V_new = np.zeros_like(V)
            for b in range(mp):
                M = np.zeros((nm * pn, r))
                for a in range(nm):
                    for c in range(pn):
                        M[a * pn + c, :] = U[a, :] * W[c, :]
                rhs = T[:, b, :].reshape(nm, pn).flatten()
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    V_new[b, :] = sol
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for c in range(pn):
                M = np.zeros((nm * mp, r))
                for a in range(nm):
                    for b in range(mp):
                        M[a * mp + b, :] = U[a, :] * V[b, :]
                rhs = T[:, :, c].reshape(nm, mp).flatten()
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    W_new[c, :] = sol
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Check convergence
            S = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(S - T) / (np.linalg.norm(T) + 1e-10)
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try to reduce rank
    # Start from just below schoolbook rank and go down
    for target_rank in range(best_rank - 1, max(30, best_rank - 30), -1):
        if time.time() > DEADLINE:
            break
        
        # Multiple random restarts
        for attempt in range(5):
            if time.time() > DEADLINE:
                break
            
            # Initialize by taking a subset of schoolbook columns and adding noise
            cols = np.random.choice(best_rank, target_rank, replace=False)
            Ur = U_s[:, cols].copy()
            Vr = V_s[:, cols].copy()
            Wr = W_s[:, cols].copy()
            
            # Add small noise to escape local minima
            noise_scale = 0.05
            Ur += np.random.randn(nm, target_rank) * noise_scale
            Vr += np.random.randn(mp, target_rank) * noise_scale
            Wr += np.random.randn(pn, target_rank) * noise_scale
            
            Ur, Vr, Wr, err = alt_minimize(Ur, Vr, Wr, T, max_iter=100)
            
            if err < 1e-8:
                # Verify and try to round
                U_clean = np.round(Ur * 2) / 2
                V_clean = np.round(Vr * 2) / 2
                W_clean = np.round(Wr * 2) / 2
                
                if verify(U_clean, V_clean, W_clean):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_clean, V_clean, W_clean
                    break
                elif verify(Ur, Vr, Wr):
                    best_rank = target_rank
                    best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
                    break
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
