import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 20s safety margin
    
    n, m, p = DIMENSIONS
    # Indices
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try ranks from lower bound to 32
    # Lower bound: max(n*m, m*p, p*n) = max(8, 16, 8) = 16
    # Also, the rank must be at least the matrix rank of any flattening
    
    # Precompute the nonzero entries
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nonzero.append((u_idx, v_idx, w_idx))
    
    num_nonzero = len(nonzero)  # 32
    
    # Try to find a solution for each rank
    best_result = None
    
    for rank in range(16, 33):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 1000 + rank)
            
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.zeros((p * n, rank))
            
            # Alternating minimization
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W: for each w_idx, we have a linear system
                # For each (u_idx, v_idx, w_idx) in nonzero:
                #   sum_c W[w_idx, c] * U[u_idx, c] * V[v_idx, c] = 1
                # Group by w_idx
                w_solutions = {}
                for w_idx in range(p * n):
                    # Collect all equations involving this w_idx
                    eqs = []
                    for (ui, vi, wi) in nonzero:
                        if wi == w_idx:
                            coeff = U[ui, :] * V[vi, :]  # shape (rank,)
                            eqs.append(coeff)
                    
                    if eqs:
                        A_mat = np.array(eqs)  # shape (num_eqs, rank)
                        b_vec = np.ones(len(eqs))
                        # Solve least squares
                        try:
                            sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W[w_idx, :] = sol
                        except:
                            W[w_idx, :] = 0
                    else:
                        W[w_idx, :] = 0
                
                # Check zero constraints
                # For all (u_idx, v_idx, w_idx) not in nonzero, sum_c U[u_idx,c]*V[v_idx,c]*W[w_idx,c] should be 0
                # This is expensive to check fully, so let's just check the nonzero constraints first
                max_err = 0
                for (ui, vi, wi) in nonzero:
                    val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
                    err = abs(val - 1)
                    max_err = max(max_err, err)
                
                if max_err < 1e-6:
                    # Check zero constraints (sample)
                    zero_ok = True
                    for _ in range(100):
                        ui = np.random.randint(0, n * m)
                        vi = np.random.randint(0, m * p)
                        wi = np.random.randint(0, p * n)
                        # Check if this is a nonzero entry
                        is_nonzero = any(ui == a and vi == b and wi == c for (a, b, c) in nonzero)
                        if not is_nonzero:
                            val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
                            if abs(val) > 1e-4:
                                zero_ok = False
                                break
                    if zero_ok:
                        converged = True
                        break
                
                # Update U
                for u_idx in range(n * m):
                    eqs = []
                    for (ui, vi, wi) in nonzero:
                        if ui == u_idx:
                            coeff = V[vi, :] * W[wi, :]
                            eqs.append(coeff)
                    if eqs:
                        A_mat = np.array(eqs)
                        b_vec = np.ones(len(eqs))
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[u_idx, :] = sol
                        except:
                            U[u_idx, :] = 0
                    else:
                        U[u_idx, :] = 0
                
                # Update V
                for v_idx in range(m * p):
                    eqs = []
                    for (ui, vi, wi) in nonzero:
                        if vi == v_idx:
                            coeff = U[ui, :] * W[wi, :]
                            eqs.append(coeff)
                    if eqs:
                        A_mat = np.array(eqs)
                        b_vec = np.ones(len(eqs))
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V[v_idx, :] = sol
                        except:
                            V[v_idx, :] = 0
                    else:
                        V[v_idx, :] = 0
            
            if converged:
                # Verify fully
                full_ok = True
                for ui in range(n * m):
                    for vi in range(m * p):
                        for wi in range(p * n):
                            val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
                            is_nonzero = any(ui == a and vi == b and wi == c for (a, b, c) in nonzero)
                            if is_nonzero:
                                if abs(val - 1) > 1e-4:
                                    full_ok = False
                                    break
                            else:
                                if abs(val) > 1e-4:
                                    full_ok = False
                                    break
                        if not full_ok:
                            break
                    if not full_ok:
                        break
                
                if full_ok:
                    # Convert to lists
                    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n * m)]
                    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m * p)]
                    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p * n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
