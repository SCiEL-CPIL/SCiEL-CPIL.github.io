import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy: Decompose 4x4*4x7 using block structure
    # Split the 4x4 matrix into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split the 4x7 matrix into 2x2 blocks of size 2x(7/2=3.5) -- not clean
    # Alternative: Use 4 separate 2x2 multiplications with overlapping structure
    
    # Better approach: Use Kronecker product structure
    # 4 = 2*2, so we can write:
    # A (4x4) = kron(A_left, A_right) where each is 2x2? No, that's not how matrix mult works.
    
    # Correct approach: Block decomposition
    # A is 4x4, B is 4x7
    # Split A into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into four 2x(7/2) blocks -- but 7 is odd, so this doesn't work cleanly.
    
    # Alternative: Use 1x4 and 4x1 splits
    # Or: Use the fact that 4x4*4x7 can be done via multiple 2x2*2x3 and 2x2*2x4
    
    # Let me try a different structural approach:
    # Use rank-1 terms systematically based on index patterns
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # For n=4, m=4, p=7: T[i*4+j, j*7+k, k*4+i] = 1
    
    # Try to find a structured low-rank decomposition using symmetry
    
    # Method: Group indices and use polynomial-like decomposition
    # Consider fixing some indices and varying others
    
    # Actually, let's try the simplest structured approach:
    # Use Kronecker products of smaller known decompositions
    
    # 4x4*4x7 = (2x2 * 2x2) * (2x2 * 2x4) approximately?
    # This is getting complex. Let me try a direct structured peeling.
    
    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Structured peeling: Try all possible sign patterns in a systematic way
    # For each position, determine the optimal sign based on remaining tensor
    
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_terms = 200  # Cap on number of terms to try
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        # Determine u, v, w by greedy sign selection
        # Start with u = sign of marginal sums
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        # Initialize: set u[a] = sign(sum_{b,c} R[a,b,c])
        u_init = np.sign(np.sum(R, axis=(1,2)))
        v_init = np.sign(np.sum(R, axis=(0,2)))
        w_init = np.sign(np.sum(R, axis=(0,1)))
        
        # Use these as starting point for alternating maximization
        u = u_init.copy()
        v = v_init.copy()
        w = w_init.copy()
        
        # Alternating maximization with multiple restarts
        best_score = 0
        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            if restart == 0:
                cu, cv, cw = u.copy(), v.copy(), w.copy()
            else:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update u
                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)
                
                # Update v
                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)
                
                # Update w
                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)
                
                cu, cv, cw = new_u, new_v, new_w
                
                if np.array_equal(cu, new_u) and np.array_equal(cv, new_v) and np.array_equal(cw, new_w):
                    break
            
            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_score <= 0:
            # Can't make progress, break
            break
        
        # Subtract the term
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    # Check if we completed the decomposition
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        # Verify exactness
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            if rank < n * m * p:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
