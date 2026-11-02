import numpy as np
import random
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    def schoolbook():
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

    # Target rank 10
    target_rank = 10
    deadline = time.time() + 110  # 110s limit, 10s margin
    
    # Precompute tensor T indices
    # T[i, j, k] = 1
    # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    # We fix U and V, solve for W.
    # For each (i, k), we have:
    # sum_l (U[i*m+j, l] * V[j*p+k, l]) * W[k*n+i, l] = 1 for j=0..m-1
    
    best_solution = None

    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Random sparse U, V
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        
        for l in range(target_rank):
            # Pick random non-zero entries
            u_idx = random.sample(range(n * m), k=random.randint(1, 3))
            v_idx = random.sample(range(m * p), k=random.randint(1, 3))
            
            for idx in u_idx:
                U[idx, l] = random.choice([-1, 1])
            for idx in v_idx:
                V[idx, l] = random.choice([-1, 1])
        
        # Build system for W
        # Unknowns: W has shape (p*n, target_rank)
        # We solve for each row of W independently?
        # Row r of W corresponds to (k, i) where r = k*n + i.
        # Equations for this row:
        # For each j in 0..m-1:
        # sum_l (U[i*m+j, l] * V[j*p+k, l]) * W[r, l] = 1
        
        W_solution = np.zeros((p * n, target_rank), dtype=float)
        valid = True
        
        for k in range(p):
            for i in range(n):
                r = k * n + i
                # Matrix A for this row: m x target_rank
                A = np.zeros((m, target_rank), dtype=float)
                for j in range(m):
                    for l in range(target_rank):
                        A[j, l] = U[i * m + j, l] * V[j * p + k, l]
                
                # b vector: m x 1
                b = np.ones(m, dtype=float)
                
                # Solve A * W_row = b
                # Using least squares
                try:
                    W_row, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if solution is exact
                    if np.allclose(A @ W_row, b, atol=1e-9):
                        # Check if W_row is close to integers or half-integers
                        # Multiply by 2 to check integers
                        W_row_scaled = W_row * 2
                        W_row_rounded = np.round(W_row_scaled)
                        if np.allclose(W_row_scaled, W_row_rounded, atol=1e-6):
                            W_solution[r, :] = W_row_rounded / 2.0
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Verify exactness over integers/half-integers
            # Convert to fractions to be safe
            # Check if entries are integer or half-integer
            U_int = U.astype(float)
            V_int = V.astype(float)
            W_int = W_solution
            
            exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for l in range(target_rank):
                            s += U_int[i * m + j, l] * V_int[j * p + k, l] * W_int[k * n + i, l]
                        if abs(s - 1.0) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                # Convert to lists of floats (integers or x.5)
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                
                # Clean up floats to be exact integers or .5
                def clean(x):
                    if abs(x - round(x)) < 1e-9:
                        return int(round(x))
                    else:
                        return x
                
                U_list = [[clean(val) for val in row] for row in U_list]
                V_list = [[clean(val) for val in row] for row in V_list]
                W_list = [[clean(val) for val in row] for row in W_list]
                
                return {"U": U_list, "V": V_list, "W": W_list}

    return schoolbook()
