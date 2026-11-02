import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 17
    target_rank = 17
    
    # Deadline guard
    start_time = time.time()
    timeout = 10.0  # Search for 10 seconds
    
    # Schoolbook fallback
    def schoolbook():
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

    # Try to find rank 17 decomposition
    rng = np.random.default_rng(42)
    
    for _ in range(10000):
        if time.time() - start_time > timeout:
            break
            
        # Random V and W with integer entries -1, 0, 1
        V = rng.integers(-1, 2, size=(m * p, target_rank))
        W = rng.integers(-1, 2, size=(p * n, target_rank))
        
        # Solve for U
        # For each (i, j), we have a 3x3 system for U[i*m+j, 14:17]
        # We set U[i*m+j, 0:14] = 0
        U = np.zeros((n * m, target_rank))
        
        valid = True
        for i in range(n):
            for j in range(m):
                # System for k=0,1,2
                # Eq k: sum_{r=14}^{16} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                A_sub = np.zeros((3, 3))
                for k in range(p):
                    for r_idx, r in enumerate([14, 15, 16]):
                        A_sub[k, r_idx] = V[j * p + k, r] * W[k * n + i, r]
                
                try:
                    # Solve A_sub * u = [1, 1, 1]
                    u_sol = np.linalg.solve(A_sub, np.ones(3))
                    U[i * m + j, 14] = u_sol[0]
                    U[i * m + j, 15] = u_sol[1]
                    U[i * m + j, 16] = u_sol[2]
                except np.linalg.LinAlgError:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Round U to nearest half-integers
        U_rounded = np.round(U * 2) / 2.0
        
        # Verify decomposition
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_rounded[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        error += abs(val - 1.0)
                        break
                if error > 0:
                    break
            if error > 0:
                break
        
        if error == 0.0:
            # Found a valid rank 17 decomposition with half-integers
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return schoolbook()
# EVOLVE-BLOCK-END
