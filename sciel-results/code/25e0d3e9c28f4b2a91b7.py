import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the list of (i,j,k) tuples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)  # 90
    
    best_rank = nm * mp // m  # n*m*p = 90
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    rank = best_rank
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    for idx, (i, j, k) in enumerate(triples):
        U[i*m+j, idx] = 1
        V[j*p+k, idx] = 1
        W[k*n+i, idx] = 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    for target_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            # Random U, V
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            
            # Build linear system for W
            # For each triple (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            
            A = np.zeros((num_triples, target_rank))
            b = np.ones(num_triples)
            for idx, (i, j, k) in enumerate(triples):
                for c in range(target_rank):
                    A[idx, c] = U[i*m+j, c] * V[j*p+k, c]
            
            # Solve A @ W_col = b for each column of W? No, W is (pn x target_rank)
            # Actually, for each (i,k), W[k*n+i, :] is a row of W.
            # The equation for (i,j,k) involves W[k*n+i, :]
            
            # Let me restructure: for each (i,k), we have multiple j's.
            # For fixed (i,k), the equations are: for each j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of m equations in target_rank unknowns (W[k*n+i, :])
            
            # Actually, it's coupled across (i,k) because U and V are shared.
            
            # Simpler: treat all W entries as unknowns. There are pn * target_rank unknowns.
            # But the equations are only num_triples = 90.
            # So we have 90 equations in pn*target_rank = 18*target_rank unknowns.
            # For target_rank < 5, 18*target_rank < 90, so overdetermined.
            # For target_rank >= 5, 18*target_rank >= 90, so underdetermined.
            
            # Let me just use least squares for the whole system.
            # Flatten: unknowns are W_flat of size pn*target_rank
            # Equation idx: sum over (c, and the specific W entry) ...
            
            # Actually, for equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # This is a dot product of (U[i*m+j,:] * V[j*p+k,:]) with W[k*n+i,:]
            
            # So the system is: for each (i,j,k), (U[i*m+j,:] * V[j*p+k,:]) @ W[k*n+i,:] = 1
            
            # This is not a standard linear system in W_flat because each equation only involves one row of W.
            
            # For each (i,k), let's collect all j's:
            # Equations: for j in range(m): (U[i*m+j,:] * V[j*p+k,:]) @ W[k*n+i,:] = 1
            # This is a system of m equations in target_rank unknowns (W[k*n+i,:])
            
            # So for each (i,k), we can solve independently!
            
            W_sol = np.zeros((pn, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build system for W[k*n+i, :]
                    A_ik = np.zeros((m, target_rank))
                    b_ik = np.ones(m)
                    for j in range(m):
                        A_ik[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve A_ik @ x = b_ik
                    try:
                        x, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                        W_sol[k*n+i, :] = x
                        # Check residual
                        if np.any(np.abs(A_ik @ x - b_ik) > 1e-6):
                            valid = False
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check all equations
                error = 0
                for (i, j, k) in triples:
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W_sol[k*n+i, :])
                    error += (val - 1) ** 2
                if error < 1e-6:
                    found = True
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    break
        
        if found:
            # Try to reduce further
            continue
        else:
            # If we couldn't find a solution for this rank, stop
            # (assuming lower ranks are harder)
            break
    
    # Round to half-integers
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    # Verify
    error = 0
    for (i, j, k) in triples:
        val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
        error += (val - 1) ** 2
    
    if error > 1e-6:
        # Rounding broke it, use original
        U_rounded = best_U
        V_rounded = best_V
        W_rounded = best_W
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_rounded]
    V_list = [[float(x) for x in row] for row in V_rounded]
    W_list = [[float(x) for x in row] for row in W_rounded]
    
    return {"U": U_list, "V": V_list, "W": W_list}
