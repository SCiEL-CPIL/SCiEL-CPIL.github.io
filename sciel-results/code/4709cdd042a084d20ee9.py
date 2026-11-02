"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix using algebraic elimination.
    """
    n, m, p = DIMENSIONS
    
    # The tensor T is defined by T[(i,j),(j,k),(k,i)] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try ranks from the theoretical lower bound upward
    for R in range(41, 65):
        U, V, W = _try_rank(R, n, m, p)
        if U is not None:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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

def _try_rank(R, n, m, p):
    """
    Try to find a rank-R decomposition using alternating least squares.
    Returns (U, V, W) as lists of lists, or None if failed.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute entries
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx))
    
    num_constraints = len(entries)
    
    # Precompute index arrays for efficiency
    u_indices = np.array([e[0] for e in entries])
    v_indices = np.array([e[1] for e in entries])
    w_indices = np.array([e[2] for e in entries])
    
    np.random.seed(12345)
    
    for attempt in range(100):
        # Initialize with small random values
        scale = 1.0 / np.sqrt(R)
        U = np.random.randn(nm, R) * scale
        V = np.random.randn(mp, R) * scale
        W = np.random.randn(pn, R) * scale
        
        success = False
        for iteration in range(500):
            # Solve for W given U, V
            # For each constraint idx: sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r] = 1
            # This is linear in W
            A_W = np.zeros((num_constraints, pn * R))
            for idx in range(num_constraints):
                u_i = u_indices[idx]
                v_i = v_indices[idx]
                w_i = w_indices[idx]
                for r in range(R):
                    A_W[idx, w_i * R + r] = U[u_i, r] * V[v_i, r]
            
            try:
                W_flat = np.linalg.lstsq(A_W, np.ones(num_constraints), rcond=None)[0]
                W_new = W_flat.reshape(pn, R)
            except:
                break
            
            # Solve for U given V, W
            A_U = np.zeros((num_constraints, nm * R))
            for idx in range(num_constraints):
                u_i = u_indices[idx]
                v_i = v_indices[idx]
                w_i = w_indices[idx]
                for r in range(R):
                    A_U[idx, u_i * R + r] = V[v_i, r] * W_new[w_i, r]
            
            try:
                U_flat = np.linalg.lstsq(A_U, np.ones(num_constraints), rcond=None)[0]
                U_new = U_flat.reshape(nm, R)
            except:
                break
            
            # Solve for V given U_new, W_new
            A_V = np.zeros((num_constraints, mp * R))
            for idx in range(num_constraints):
                u_i = u_indices[idx]
                v_i = v_indices[idx]
                w_i = w_indices[idx]
                for r in range(R):
                    A_V[idx, v_i * R + r] = U_new[u_i, r] * W_new[w_i, r]
            
            try:
                V_flat = np.linalg.lstsq(A_V, np.ones(num_constraints), rcond=None)[0]
                V_new = V_flat.reshape(mp, R)
            except:
                break
            
            # Check convergence
            diff_U = np.max(np.abs(U_new - U))
            diff_V = np.max(np.abs(V_new - V))
            diff_W = np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if max(diff_U, diff_V, diff_W) < 1e-12:
                # Round to half-integers and verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                valid = True
                for idx in range(num_constraints):
                    u_i = u_indices[idx]
                    v_i = v_indices[idx]
                    w_i = w_indices[idx]
                    val = np.sum(U_r[u_i] * V_r[v_i] * W_r[w_i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    success = True
                    break
        
        if success:
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
    
    return None
# EVOLVE-BLOCK-END
