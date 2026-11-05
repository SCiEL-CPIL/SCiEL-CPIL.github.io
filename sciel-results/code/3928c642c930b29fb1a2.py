import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 4
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                g = k*n + i
                T[a, b, g] = 1.0
                
    start_time = time.time()
    timeout = 110.0
    
    # Precompute b_g vectors for all g
    b_vectors = []
    for g in range(p*n):
        b_g = np.zeros(n*m * m*p)
        idx = 0
        for a in range(n*m):
            for b in range(m*p):
                b_g[idx] = T[a, b, g]
                idx += 1
        b_vectors.append(b_g)
    
    # Strategy: try ranks from 25 to 36 with improved search
    # Use structured initialization inspired by known constructions
    
    for r in range(25, 37):
        if time.time() - start_time > timeout:
            break
        
        # More attempts for lower ranks
        max_attempts = 2000 if r < 28 else (1000 if r < 32 else 300)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Different initialization strategies
            if attempt % 5 == 0:
                # Sparse with {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.15, 0.7, 0.15])
                V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.15, 0.7, 0.15])
            elif attempt % 5 == 1:
                # Denser
                U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.25, 0.5, 0.25])
                V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.25, 0.5, 0.25])
            elif attempt % 5 == 2:
                # Random integers in {-2,-1,0,1,2}
                U = np.random.randint(-2, 3, size=(n*m, r))
                V = np.random.randint(-2, 3, size=(m*p, r))
            elif attempt % 5 == 3:
                # Block-structured: try to create patterns
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for c in range(r):
                    # Randomly assign blocks
                    U[np.random.randint(0, n*m, size=min(3, r//3)), c] = np.random.choice([-1, 1], size=min(3, r//3))
                    V[np.random.randint(0, m*p, size=min(4, r//3)), c] = np.random.choice([-1, 1], size=min(4, r//3))
            else:
                # Pure random
                U = np.random.randint(-1, 2, size=(n*m, r))
                V = np.random.randint(-1, 2, size=(m*p, r))
            
            # Solve for W using vectorized approach
            W = np.zeros((p*n, r))
            valid = True
            
            # Precompute U_rep and V_rep for efficient M_g construction
            # M_g[a*12+b, c] = U[a,c] * V[b,c]
            # This is the element-wise product of U[a,:] and V[b,:]
            
            for g in range(p*n):
                # Build M_g efficiently using broadcasting
                # U has shape (9, r), V has shape (12, r)
                # We want M_g of shape (108, r) where M_g[(a,b), c] = U[a,c]*V[b,c]
                
                # U_rep: repeat each row of U 12 times -> shape (108, r)
                U_rep = np.repeat(U, m*p, axis=0)  # (108, r)
                # V_rep: tile V 9 times -> shape (108, r)
                V_rep = np.tile(V, (n*m, 1))  # (108, r)
                
                M_g = U_rep * V_rep  # (108, r)
                b_g = b_vectors[g]
                
                # Solve M_g @ w_g = b_g
                try:
                    # Use lstsq for overdetermined system
                    w_g, residuals, rank_g, sv = np.linalg.lstsq(M_g, b_g, rcond=None)
                    
                    # Check residual
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-7:
                            valid = False
                            break
                    else:
                        # Verify manually
                        residual = np.linalg.norm(M_g @ w_g - b_g)
                        if residual > 1e-7:
                            valid = False
                            break
                    
                    W[g, :] = w_g
                    
                except Exception:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            # Verify exactness
            T_approx = np.einsum('ac,bc,gc->abg', U_s, V_s, W_s)
            
            if np.allclose(T_approx, T, atol=1e-9):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    # Fallback: schoolbook rank 36
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
