import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute target tensor T as a list of (u, v) indices for each w
    # T[u, v, w] = 1 if j(u)==j(v) and i(u)==i(w) and k(v)==k(w)
    # w = k*n + i
    # i = w % n, k = w // n
    # u = i*m + j
    # v = j*p + k
    
    target_pairs = []
    for w in range(pn):
        i_w = w % n
        k_w = w // n
        pairs = []
        for j in range(m):
            u = i_w * m + j
            v = j * p + k_w
            pairs.append((u, v))
        target_pairs.append(pairs)
        
    # b_w vectors
    b_vectors = []
    for w in range(pn):
        b = np.zeros(nm * mp)
        for u, v in target_pairs[w]:
            b[u * mp + v] = 1.0
        b_vectors.append(b)
        
    deadline = time.time() + 140
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank), dtype=int)
    best_V = np.zeros((mp, best_rank), dtype=int)
    best_W = np.zeros((pn, best_rank), dtype=int)
    
    # Initialize with schoolbook
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                best_U[u, idx] = 1
                best_V[v, idx] = 1
                best_W[w, idx] = 1
                idx += 1
                
    # Search for lower rank
    # Try ranks from 99 down to 50
    # We will use random U, V with entries in {0, 1}
    # To increase chance of integer W, we can try to make U, V sparse
    
    for r in range(99, 49, -1):
        if time.time() > deadline:
            break
            
        attempts = 200 if r > 80 else 100
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Random sparse U, V
            # Density 0.2
            U = (np.random.rand(nm, r) < 0.2).astype(int)
            V = (np.random.rand(mp, r) < 0.2).astype(int)
            
            W = np.zeros((pn, r))
            valid = True
            
            for w in range(pn):
                # A_w[u, v, l] = U[u, l] * V[v, l]
                # Flatten u, v to row index
                # A_w is (nm*mp) x r
                # We can compute A_w efficiently
                # A_w = np.einsum('ul,vl->uvl', U, V).reshape(nm*mp, r)
                
                # Optimization: Only check if the system is solvable
                # Use lstsq
                try:
                    A_w = np.einsum('ul,vl->uvl', U, V).reshape(nm * mp, r)
                    W_w, residuals, rank, sv = np.linalg.lstsq(A_w, b_vectors[w], rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check residual
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        valid = False
                        break
                else:
                    res = np.linalg.norm(A_w @ W_w - b_vectors[w])
                    if res > 1e-9:
                        valid = False
                        break
                
                # Check integer/half-integer
                W_w_scaled = W_w * 2
                if not np.allclose(W_w_scaled, np.round(W_w_scaled), atol=1e-6):
                    valid = False
                    break
                    
                W[w] = W_w
                
            if valid:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break # Found a solution for this r, try lower r
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
