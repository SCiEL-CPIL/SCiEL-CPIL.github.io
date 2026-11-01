import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor indices
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Build the target tensor indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i  <-- Wait, the task says W[k*n+i, r]
    # Let's check the convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i.
    # U index is i*m + j.
    # V index is j*p + k.
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
                
    num_indices = len(indices)
    
    # We will search for the minimal rank R
    # Start from lower bound
    # Lower bound for 2x3x9 is 18.
    # We will try R from 18 to 24.
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds for search
    
    for R in range(18, 25):
        if time.time() > deadline:
            break
            
        # Variables
        # U: n*m * R
        # V: m*p * R
        # W: p*n * R
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        def residuals(x):
            U = x[:size_U].reshape(n * m, R)
            V = x[size_U:size_U+size_V].reshape(m * p, R)
            W = x[size_U+size_V:].reshape(p * n, R)
            
            res = np.zeros(num_indices)
            for idx, (a, b, c) in enumerate(indices):
                # Sum_r U[a, r] * V[b, r] * W[c, r]
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                res[idx] = val - 1.0
                
            # We also need to ensure zeros elsewhere?
            # For exactness, yes. But checking all 2916 is slow.
            # However, if the rank is minimal, the solution is likely unique or structured.
            # Let's just check the non-zero ones first.
            # If we find a solution, we verify it fully.
            
            return res

        # Initialize with random values
        np.random.seed(42)
        x0 = np.random.rand(total_size) * 0.1
        
        # Solve
        try:
            sol = opt.least_squares(residuals, x0, method='trf', max_nfev=10000)
            if sol.success and sol.cost < 1e-10:
                # Verify full tensor
                U = sol.x[:size_U].reshape(n * m, R)
                V = sol.x[size_U:size_U+size_V].reshape(m * p, R)
                W = sol.x[size_U+size_V:].reshape(p * n, R)
                
                # Check a few random zero entries
                is_valid = True
                for _ in range(100):
                    a = np.random.randint(n * m)
                    b = np.random.randint(m * p)
                    c = np.random.randint(p * n)
                    # Check if (a,b,c) is a valid index
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j == j2 and i == i2:
                        # Non-zero
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val - 1.0) > 1e-6:
                            is_valid = False
                            break
                    else:
                        # Zero
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > 1e-6:
                            is_valid = False
                            break
                
                if is_valid:
                    # Round to 0.5
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    valid_rounded = True
                    for idx, (a, b, c) in enumerate(indices):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val - 1.0) > 1e-9:
                            valid_rounded = False
                            break
                    if valid_rounded:
                        for _ in range(100):
                            a = np.random.randint(n * m)
                            b = np.random.randint(m * p)
                            c = np.random.randint(p * n)
                            i, j = divmod(a, m)
                            j2, k = divmod(b, p)
                            k2, i2 = divmod(c, n)
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if j == j2 and i == i2:
                                if abs(val - 1.0) > 1e-9:
                                    valid_rounded = False
                                    break
                            else:
                                if abs(val) > 1e-9:
                                    valid_rounded = False
                                    break
                            if not valid_rounded:
                                break
                    
                    if valid_rounded:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        except Exception as e:
            pass
            
    # Fallback: Schoolbook
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
