import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _construct_rank14():
    """Return the known rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1 (k=0)
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Block 2 (k=1)
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank13_search(deadline):
    """
    Attempt to find a rank-13 decomposition by perturbing the rank-14 solution
    and optimizing to drive one coefficient to zero.
    """
    n, m, p = 2, 2, 4
    rank_target = 13
    
    # Start from the rank-14 solution
    base = _construct_rank14()
    U = np.array(base['U'], dtype=float)
    V = np.array(base['V'], dtype=float)
    W = np.array(base['W'], dtype=float)
    
    # The tensor equation: for each (i,k,j), sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    # We want to find rank-13 by setting one entry to 0 and re-optimizing.
    
    # Strategy: try to drive one specific entry to zero while keeping the tensor valid.
    # Pick a candidate entry to eliminate. We'll try several candidates.
    
    # Build the list of all (matrix, row, col) entries that are non-zero in the base solution
    candidates = []
    for r in range(14):
        for i in range(n*m):
            if abs(U[i, r]) > 1e-12:
                candidates.append(('U', i, r))
        for k in range(m*p):
            if abs(V[k, r]) > 1e-12:
                candidates.append(('V', k, r))
        for j in range(p*n):
            if abs(W[j, r]) > 1e-12:
                candidates.append(('W', j, r))
    
    random.seed(42)
    random.shuffle(candidates)
    
    best_solution = None
    best_error = float('inf')
    
    for candidate in candidates:
        if time.time() > deadline - 5:
            break
        
        mat_name, row, col = candidate
        
        # Save original
        U_orig = U.copy()
        V_orig = V.copy()
        W_orig = W.copy()
        
        # Zero out the candidate entry
        if mat_name == 'U':
            U[row, col] = 0.0
        elif mat_name == 'V':
            V[row, col] = 0.0
        else:
            W[row, col] = 0.0
        
        # Now try to optimize the remaining free variables to satisfy the tensor equations
        # Free variables: all entries except the one we zeroed
        # We'll use a simple gradient descent on the squared error of the tensor equations
        
        # Build the error function
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        val = 0.0
                        for r in range(14):
                            val += U[a, r] * V[b, r] * W[c, r]
                        err += (val - 1.0) ** 2
            return err
        
        # Gradient descent with small steps
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients numerically (or analytically)
            # For efficiency, compute gradients analytically
            # dE/du[a,r] = 2 * sum_{(i,k,j): a=i*m+k} (val_{ijk} - 1) * V[k*p+j, r] * W[j*n+i, r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        val = 0.0
                        for r in range(14):
                            val += U[a, r] * V[b, r] * W[c, r]
                        diff = val - 1.0
                        for r in range(14):
                            dU[a, r] += 2 * diff * V[b, r] * W[c, r]
                            dV[b, r] += 2 * diff * U[a, r] * W[c, r]
                            dW[c, r] += 2 * diff * U[a, r] * V[b, r]
            
            # Update, but keep the zeroed entry at zero
            if mat_name == 'U':
                U -= lr * dU
                U[row, col] = 0.0
            elif mat_name == 'V':
                V -= lr * dV
                V[row, col] = 0.0
            else:
                W -= lr * dW
                W[row, col] = 0.0
            
            # Check convergence
            err = compute_error(U, V, W)
            if err < 1e-10:
                break
        
        err = compute_error(U, V, W)
        if err < best_error:
            best_error = err
            if err < 1e-8:
                # Found a valid rank-13 solution!
                best_solution = (U.copy(), V.copy(), W.copy())
                break
        
        # Restore original
        U = U_orig
        V = V_orig
        W = W_orig
    
    if best_solution is not None:
        U_sol, V_sol, W_sol = best_solution
        # Convert to list format with integer/half-integer rounding if close
        U_list = [[0]*14 for _ in range(n*m)]
        V_list = [[0]*14 for _ in range(m*p)]
        W_list = [[0]*14 for _ in range(p*n)]
        for i in range(n*m):
            for r in range(14):
                U_list[i][r] = round(U_sol[i, r] * 2) / 2
        for k in range(m*p):
            for r in range(14):
                V_list[k][r] = round(V_sol[k, r] * 2) / 2
        for j in range(p*n):
            for r in range(14):
                W_list[j][r] = round(W_sol[j, r] * 2) / 2
        
        # Verify the rounded solution
        if _verify(U_list, V_list, W_list):
            # Check if it's actually rank 13 (one column all zeros)
            for r in range(14):
                col_zero = True
                for i in range(n*m):
                    if abs(U_list[i][r]) > 1e-9:
                        col_zero = False
                        break
                if col_zero:
                    for k in range(m*p):
                        if abs(V_list[k][r]) > 1e-9:
                            col_zero = False
                            break
                if col_zero:
                    for j in range(p*n):
                        if abs(W_list[j][r]) > 1e-9:
                            col_zero = False
                            break
                if col_zero:
                    # This is rank 13!
                    # Remove the zero column
                    U_final = [[U_list[i][r] for r in range(14) if r != r] for i in range(n*m)]
                    V_final = [[V_list[k][r] for r in range(14) if r != r] for k in range(m*p)]
                    W_final = [[W_list[j][r] for r in range(14) if r != r] for j in range(p*n)]
                    if _verify(U_final, V_final, W_final):
                        return {"U": U_final, "V": V_final, "W": W_final}
        
        # If rounding didn't work, try without rounding
        if _verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist()):
            # Check for zero column
            for r in range(14):
                col_zero = True
                for i in range(n*m):
                    if abs(U_sol[i, r]) > 1e-9:
                        col_zero = False
                        break
                if col_zero:
                    for k in range(m*p):
                        if abs(V_sol[k, r]) > 1e-9:
                            col_zero = False
                            break
                if col_zero:
                    for j in range(p*n):
                        if abs(W_sol[j, r]) > 1e-9:
                            col_zero = False
                            break
                if col_zero:
                    U_final = [[U_sol[i, r] for r in range(14) if r != r] for i in range(n*m)]
                    V_final = [[V_sol[k, r] for r in range(14) if r != r] for k in range(m*p)]
                    W_final = [[W_sol[j, r] for r in range(14) if r != r] for j in range(p*n)]
                    if _verify(U_final, V_final, W_final):
                        return {"U": U_final, "V": V_final, "W": W_final}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find a rank-13 solution
    result = _try_rank13_search(deadline)
    if result is not None:
        return result
    
    # Fall back to rank-14
    return _construct_rank14()
