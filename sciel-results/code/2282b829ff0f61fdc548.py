"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[r, t, s] = 1
    # U index: r*m + t
    # V index: t*p + s
    # W index: s*n + r
    
    for r in range(n):
        for t in range(m):
            for s in range(p):
                expected = 1.0
                actual = 0.0
                u_idx = r * m + t
                v_idx = t * p + s
                w_idx = s * n + r
                for k in range(rank):
                    actual += U[u_idx, k] * V[v_idx, k] * W[w_idx, k]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank14():
    """Construct the known rank-14 decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _search_rank(target_rank, n=2, m=2, p=4, time_limit=10.0):
    """
    Search for a decomposition of the given rank.
    Uses a heuristic: fix U and V to sparse random patterns, solve for W.
    """
    start_time = time.time()
    um = n * m
    vm = m * p
    wm = p * n
    
    # Tensor T indices: u, v, w
    # T[u, v, w] = 1 if valid
    # u = r*m + t
    # v = t*p + s
    # w = s*n + r
    # Valid if r in [0,n), t in [0,m), s in [0,p)
    # Note: u determines r, t. v determines t, s. w determines s, r.
    # Consistency: t from u must match t from v. s from v must match s from w. r from u must match r from w.
    
    T = np.zeros((um, vm, wm))
    for r in range(n):
        for t in range(m):
            for s in range(p):
                u = r * m + t
                v = t * p + s
                w = s * n + r
                T[u, v, w] = 1.0

    # Reshape T to matrix form for linear solve
    # We want sum_k U[u,k] V[v,k] W[w,k] = T[u,v,w]
    # Let's fix U and V. Then for each k, we have a term U[u,k] V[v,k] W[w,k].
    # This is linear in W.
    # Let A_k be the matrix of coefficients for W in the k-th term?
    # Actually, we can vectorize.
    # Let vec(W) be the unknown.
    # The equation is sum_k (U[:,k] * V[:,k]) outer W[:,k] = T
    # This is linear in W[:,k].
    
    # We will try random sparse U and V.
    # Number of non-zeros in U: roughly um * (target_rank / um) ?
    # Let's just try random dense U and V with small integers, then solve for W.
    # If U and V are fixed, the system is:
    # sum_k W[:,k] * (U[:,k] * V[:,k]) = T (tensor product)
    # This is a linear system for the entries of W.
    # Size of W: wm * target_rank.
    # Number of equations: um * vm * wm.
    # This is overdetermined if wm * target_rank < um * vm * wm.
    # For rank 12: 8*12=96 vars. 4*8*8=256 eqs.
    # We can use least squares, but we need exact solution.
    # So we need U and V such that the system is consistent and has an exact solution.
    
    # Heuristic:
    # 1. Generate random U, V with entries in {-1, 0, 1}.
    # 2. Construct the linear system for W.
    # 3. Solve using np.linalg.lstsq.
    # 4. Check if the solution is exact (residual ~ 0) and if W entries are close to integers/half-integers.
    
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Generate sparse random U, V
        # Ensure some structure?
        U_rand = np.zeros((um, target_rank))
        V_rand = np.zeros((vm, target_rank))
        
        # Fill with random -1, 0, 1
        # Density ~ 2/um?
        for k in range(target_rank):
            # Pick random indices
            idx_u = np.random.choice(um, size=2, replace=False)
            idx_v = np.random.choice(vm, size=2, replace=False)
            
            U_rand[idx_u[0], k] = 1
            U_rand[idx_u[1], k] = -1 if random.random() > 0.5 else 1
            
            V_rand[idx_v[0], k] = 1
            V_rand[idx_v[1], k] = -1 if random.random() > 0.5 else 1
            
        # Construct linear system for W
        # A * vec(W) = vec(T)
        # A has shape (um*vm*wm, wm*target_rank)
        # This is too large to construct explicitly in Python efficiently for many iterations.
        # Instead, we can solve for each column of W sequentially? No, they are coupled.
        
        # Alternative: Use the fact that T is sparse.
        # Let's try a different approach:
        # Iterate k from 0 to R-1.
        # Assume U, V are fixed.
        # We need W.
        # This is equivalent to finding W such that sum_k W_k \otimes (U_k * V_k) = T.
        
        # Let's just try to solve the linear system using a sparse solver if available?
        # scipy is available.
        
        # Construct A matrix
        # A[ (u,v,w), (w', k) ] = U[u,k] * V[v,k] * (1 if w==w' else 0)
        # This matrix is block diagonal in w? No.
        # For a fixed w, the equation is sum_k W[w,k] * (U[:,k] * V[:,k])[u,v] = T[u,v,w]
        # So for each w, we have a linear system for W[w, :] (vector of length R).
        # Let M_w be the matrix with columns U[:,k] * V[:,k] (flattened).
        # Then M_w @ W[w, :] = T[:, :, w] (flattened)
        # We can solve for W[w, :] for each w independently!
        
        W_sol = np.zeros((wm, target_rank))
        valid = True
        
        for w in range(wm):
            # Build M_w: shape (um*vm, target_rank)
            M_w = np.zeros((um * vm, target_rank))
            for k in range(target_rank):
                # Flatten U[:,k] * V[:,k]
                # Note: U and V indices must be consistent with T
                # T[u,v,w] is non-zero only if u,v,w are consistent.
                # But U and V are defined for all u,v.
                # The product U[u,k] * V[v,k] is just a vector of length um*vm.
                vec = np.zeros(um * vm)
                for u in range(um):
                    for v in range(vm):
                        vec[u * vm + v] = U_rand[u, k] * V_rand[v, k]
                M_w[:, k] = vec
                
            # RHS: T[:, :, w] flattened
            rhs = T[:, :, w].flatten()
            
            # Solve M_w @ x = rhs
            # Use least squares
            try:
                x, residuals, rank, s = np.linalg.lstsq(M_w, rhs, rcond=None)
                # Check if exact
                if np.max(np.abs(M_w @ x - rhs)) > 1e-9:
                    valid = False
                    break
                W_sol[w, :] = x
            except:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check if W_sol entries are close to integers or half-integers
        # And check the full decomposition
        # Convert to lists for _verify
        U_list = U_rand.tolist()
        V_list = V_rand.tolist()
        W_list = W_sol.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try to find rank 12
    start = time.time()
    result = _search_rank(12, n, m, p, time_limit=10.0)
    if result:
        return result
        
    # Try rank 13
    if time.time() - start < 20.0:
        result = _search_rank(13, n, m, p, time_limit=10.0)
        if result:
            return result
            
    # Fallback to rank 14
    return _construct_rank14()
# EVOLVE-BLOCK-END
