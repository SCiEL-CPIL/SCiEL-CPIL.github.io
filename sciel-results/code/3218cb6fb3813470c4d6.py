import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Start with the known rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Efficient check: for each column r, check if removing it still gives valid decomposition
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Without column r: T'[a,b,c] = T[a,b,c] - U[a,r]*V[b,r]*W[c,r]
    # We need T'[a,b,c] = 0 for all a,b,c where T[a,b,c] = 0, and T'[a,b,c] = 1 where T[a,b,c] = 1
    
    # More efficiently: check if U[:,r]*V[:,r]*W[:,r] is in the span of the other rank-1 tensors
    # This is equivalent to checking if the outer product is in the column space
    
    # For each r, compute the contribution and check if it's redundant
    def is_redundant(r, U_m, V_m, W_m):
        """Check if column r is redundant using linear algebra."""
        # The remaining columns must sum to T
        # This is equivalent to: vec(T) is in the column space of the matrix
        # whose columns are vec(rank-1 tensors for r' != r)
        
        # Vectorize T
        T_flat = T.flatten()
        
        # Form the matrix of remaining rank-1 tensors (vectorized)
        # Each column is vec(U[:,r'] * V[:,r'] * W[:,r'])
        # We can compute this as: for each r' != r, the outer product U[:,r'] ⊗ V[:,r'] ⊗ W[:,r'] flattened
        
        # More efficient: use the fact that vec(U*V*W) = (W ⊗ U^T) vec(V) for appropriate reshaping
        # But let's just compute directly for now
        
        mask = np.ones(U_m.shape[1], dtype=bool)
        mask[r] = False
        cols = np.where(mask)[0]
        
        if len(cols) == 0:
            return np.allclose(T, 0, atol=1e-9)
        
        # Build the matrix: each column is the vectorized rank-1 tensor
        # Shape: (n*m*m*p*p*n, len(cols))
        # This is too large for direct computation
        
        # Alternative: check using random projections
        # Pick random vectors and check the bilinear form
        rng = np.random.RandomState(42)
        for _ in range(10):
            # Random test: pick random a, b, c indices
            a = rng.randint(n * m)
            b = rng.randint(m * p)
            c = rng.randint(p * n)
            
            # Check if U[a,r']*V[b,r']*W[c,r'] summed over r'!=r equals T[a,b,c]
            sum_rem = 0.0
            for rp in cols:
                sum_rem += U_m[a, rp] * V_m[b, rp] * W_m[c, rp]
            
            if abs(sum_rem - T[a, b, c]) > 1e-9:
                return False
        
        return True
    
    # Try removing columns greedily
    best_rank = rank
    r = 0
    while r < rank:
        if time.time() > deadline:
            break
        if is_redundant(r, U_np, V_np, W_np):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank -= 1
            # Don't restart; move to next column (which is now at index r)
        else:
            r += 1
    
    # Convert back to lists with integer/half-integer entries
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
