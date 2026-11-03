import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    diag_indices = []
    zero_indices = []
    
    # Target tensor T is 1 at (i,j), (j,k), (k,i)
    # We need to map these to flat indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # Create a set of diagonal tuples
    diag_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                diag_set.add((a, b, c))
    
    # Generate all indices
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                all_indices.append((a, b, c))
    
    for idx in all_indices:
        if idx in diag_set:
            diag_indices.append(idx)
        else:
            zero_indices.append(idx)
            
    diag_indices = np.array(diag_indices)
    zero_indices = np.array(zero_indices)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Compute R
        R = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check diagonal
        for a, b, c in diag_indices:
            if abs(R[a, b, c] - 1.0) > tol:
                return False
                
        # Check zeros
        for a, b, c in zero_indices:
            if abs(R[a, b, c]) > tol:
                return False
        return True

    def make_result(U, V, W):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up
        for lst in [U_list, V_list, W_list]:
            for row in lst:
                for idx in range(len(row)):
                    rounded = round(row[idx] * 2) / 2.0
                    if abs(row[idx] - rounded) < 1e-8:
                        row[idx] = rounded
        return {'U': U_list, 'V': V_list, 'W': W_list}

    best_result = None
    best_rank = float('inf')
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = make_result(U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank

    # Search for lower rank
    # Try ranks from 40 down to 21
    for target_rank in range(40, 20, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Multiple trials
        max_trials = 5
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # SGD
            lr = 0.01
            steps = 2000
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Forward
                R = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Error
                E = np.zeros_like(R)
                E[diag_indices[:, 0], diag_indices[:, 1], diag_indices[:, 2]] = R[diag_indices[:, 0], diag_indices[:, 1], diag_indices[:, 2]] - 1.0
                E[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]] = R[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]]
                
                loss = np.sum(E**2)
                
                if loss < 1e-10:
                    break
                    
                # Gradients
                grad_U = 2 * np.einsum('abc,br,cr->ar', E, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', E, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', E, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            # Check if converged
            R = np.einsum('ar,br,cr->abc', U, V, W)
            E_diag = R[diag_indices[:, 0], diag_indices[:, 1], diag_indices[:, 2]] - 1.0
            E_zero = R[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]]
            loss = np.sum(E_diag**2) + np.sum(E_zero**2)
            
            if loss < 1e-8:
                # Try rounding
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    result = make_result(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    best_result = result
                    best_rank = target_rank
                    break

    return best_result
# EVOLVE-BLOCK-END
