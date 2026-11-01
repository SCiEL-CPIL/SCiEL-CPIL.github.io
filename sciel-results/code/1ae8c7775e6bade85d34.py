import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140
    
    # We will search for ranks 16, 15, 14 using discrete random search
    # This is much faster than continuous optimization for small ranks
    # and respects the integer/half-integer constraint directly.
    
    for rank in [16, 15, 14]:
        if time.time() > deadline - 10:
            break
            
        max_trials = 5000
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Random initialization with values in {-1, -0.5, 0, 0.5, 1}
            # Sparse is better
            vals = np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.5, 1.0])
            probs = np.array([0.05, 0.1, 0.6, 0.6, 0.6, 0.1, 0.05]) # Roughly, need to normalize
            # Simpler: use uniform random and round
            U = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(nm, rank))
            V = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(mp, rank))
            W = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(pn, rank))
            
            # Compute R
            R = np.einsum('as,bs,cs->abc', U, V, W)
            
            if np.allclose(R, T, atol=1e-9):
                # Convert to lists of floats/ints
                U_out = U.tolist()
                V_out = V.tolist()
                W_out = W.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
                
            # Local refinement: try to fix errors
            # If not allclose, we can try to adjust specific entries
            # This is expensive, so we only do it if we are close
            diff = R - T
            if np.sum(diff**2) < 5.0: # Close enough to try refinement
                # Try flipping signs or adding 0.5 to random entries
                for _ in range(10):
                    # Pick a random matrix, entry, and delta
                    mat_idx = np.random.randint(3)
                    r_idx, c_idx = np.random.randint(nm) if mat_idx==0 else (np.random.randint(mp), np.random.randint(rank)) if mat_idx==1 else (np.random.randint(pn), np.random.randint(rank))
                    
                    if mat_idx == 0:
                        old_val = U[r_idx, c_idx]
                        U[r_idx, c_idx] = old_val + np.random.choice([-0.5, 0.5])
                        # Clip to reasonable range
                        if U[r_idx, c_idx] > 2: U[r_idx, c_idx] = 2
                        if U[r_idx, c_idx] < -2: U[r_idx, c_idx] = -2
                    elif mat_idx == 1:
                        old_val = V[r_idx, c_idx]
                        V[r_idx, c_idx] = old_val + np.random.choice([-0.5, 0.5])
                    else:
                        old_val = W[r_idx, c_idx]
                        W[r_idx, c_idx] = old_val + np.random.choice([-0.5, 0.5])
                        
                    R = np.einsum('as,bs,cs->abc', U, V, W)
                    if np.allclose(R, T, atol=1e-9):
                        U_out = U.tolist()
                        V_out = V.tolist()
                        W_out = W.tolist()
                        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
