import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = nm * mp
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return U, V, W
    
    # Greedy tensor decomposition
    U_list = []
    V_list = []
    W_list = []
    T_remaining = T.copy()
    
    max_terms = nm * mp  # don't exceed schoolbook
    
    for term_idx in range(max_terms):
        if time.time() > deadline - 5:
            break
        
        if np.max(np.abs(T_remaining)) < 1e-10:
            break
        
        # Find the best rank-1 term to subtract
        # A rank-1 term is u * v * w (outer product)
        # We want to maximize the correlation with T_remaining
        
        # Try all possible unit vectors (or a subset)
        # For simplicity, try vectors with entries in {-1, 0, 1}
        
        best_score = -np.inf
        best_u = None
        best_v = None
        best_w = None
        best_scale = 0
        
        # Sample some candidate vectors
        np.random.seed(term_idx)
        
        for _ in range(1000):
            u = np.random.choice([-1, 0, 1], size=nm)
            v = np.random.choice([-1, 0, 1], size=mp)
            w = np.random.choice([-1, 0, 1], size=pn)
            
            # Compute the outer product
            term = np.einsum('a,b,c->abc', u, v, w)
            
            # Compute the scale factor
            dot = np.sum(T_remaining * term)
            norm_sq = np.sum(term * term)
            
            if norm_sq == 0:
                continue
            
            scale = dot / norm_sq
            
            # Compute the residual after subtracting scale * term
            residual = T_remaining - scale * term
            score = np.sum(residual * residual)
            
            if score < best_score:
                best_score = score
                best_u = u
                best_v = v
                best_w = w
                best_scale = scale
        
        if best_u is None:
            break
        
        # Subtract the best term
        T_remaining -= best_scale * np.einsum('a,b,c->abc', best_u, best_v, best_w)
        
        # Store the term (scale the vectors)
        U_list.append(best_scale * best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        # Check if we've converged
        if np.max(np.abs(T_remaining)) < 1e-10:
            break
    
    rank = len(U_list)
    
    if rank == 0 or rank >= nm * mp:
        # Use schoolbook
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Round to half-integers
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    U_rounded = np.round(2 * U_arr) / 2.0
    V_rounded = np.round(2 * V_arr) / 2.0
    W_rounded = np.round(2 * W_arr) / 2.0
    
    # Verify
    T_dec = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
    
    if not np.allclose(T_dec, T, atol=1e-9):
        # Rounding broke the decomposition, use schoolbook
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U_rounded.tolist(),
        "V": V_rounded.tolist(),
        "W": W_rounded.tolist()
    }
