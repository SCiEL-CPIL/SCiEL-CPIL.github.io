"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def verify_fast(U, V, W, tol=1e-6):
        """Fast verification using einsum."""
        rank = U.shape[1]
        # Reconstruct tensor
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        # Check target entries
        for (a, b, c), val in target_entries.items():
            if abs(R[a, b, c] - val) > tol:
                return False
        # Check non-target: create a mask
        # For efficiency, check that R is zero where target is zero
        # Create target tensor
        T = np.zeros((nm, mp, pn))
        for (a, b, c), val in target_entries.items():
            T[a, b, c] = val
        # Check difference
        diff = R - T
        if np.max(np.abs(diff)) > tol:
            return False
        return True
    
    # Start with schoolbook
    U, V, W, rank = schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower-rank decompositions
    # Use a structured approach: try to find a solution with rank r
    # by solving a system of equations
    
    def try_find_decomposition(r, max_time=5.0):
        """Try to find a rank-r decomposition."""
        start = time.time()
        
        # The equations are:
        # For each (a,b,c) in target: sum_d U[a,d]*V[b,d]*W[c,d] = 1
        # For each (a,b,c) not in target: sum_d U[a,d]*V[b,d]*W[c,d] = 0
        
        # This is a nonlinear system. Let's use a simple approach:
        # Fix U and V, solve for W, then iterate.
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        for iteration in range(100):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            # Fix U, V; solve for W
            # For each c: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
            # This is: for each c, W[c,:] should satisfy:
            #   sum over (a,b) where T[a,b,c]=1: U[a,:]*V[b,:]*W[c,:] = 1
            #   sum over (a,b) where T[a,b,c]=0: U[a,:]*V[b,:]*W[c,:] = 0
            # This is a linear system in W[c,:]
            
            # Build linear system for each c
            # For fixed c, we have equations:
            #   For (a,b) with T[a,b,c]=1: sum_d U[a,d]*V[b,d]*W[c,d] = 1
            #   For (a,b) with T[a,b,c]=0: sum_d U[a,d]*V[b,d]*W[c,d] = 0
            # Let M_c[a,b,d] = U[a,d]*V[b,d]
            # Then: sum_d M_c[a,b,d]*W[c,d] = T[a,b,c]
            # This is a linear system: M_c @ W[c,:] = t_c where t_c is the vector of T[*,*,c]
            
            # But M_c is (nm*mp) x r, and we have nm*mp equations for r unknowns
            # Use least squares
            
            # Build the system
            # For each c, gather all (a,b) pairs
            for c in range(pn):
                # Determine which (a,b) pairs have T[a,b,c] = 1
                # T[a,b,c] = 1 iff there exists i,j,k such that:
                #   a = i*m+j, b = j*p+k, c = k*n+i
                # So for fixed c, we need to find (i,j,k) with k*n+i = c
                # i = c % n, k = c // n (if c < pn)
                # Then j must be such that a = i*m+j and b = j*p+k
                # So for each j in 0..m-1: a = i*m+j, b = j*p+k
                # Check if c = k*n+i (by construction)
                
                i = c % n
                k = c // n
                if k >= p:
                    continue
                
                # For this c, the target entries are at (a=i*m+j, b=j*p+k) for j=0..m-1
                # All other (a,b) should have T[a,b,c] = 0
                
                # Build linear system: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
                # For (a,b) in target: = 1
                # For (a,b) not in target: = 0
                
                # We have nm*mp equations for r unknowns (W[c,0],...,W[c,r-1])
                # Use least squares: minimize ||M @ W[c,:] - t||^2
                
                # M is (nm*mp) x r, M[ab, d] = U[a,d]*V[b,d]
                # t is (nm*mp), t[ab] = T[a,b,c]
                
                # This is too large to build explicitly for each c
                # Instead, use the fact that we only need to satisfy the target entries
                # and hope the non-target entries are satisfied
                
                # Simplified: only use target entries for this c
                # There are m target entries for each c
                # So we have m equations for r unknowns
                # If r > m, this is underdetermined; use least squares
                
                if r <= m:
                    # Square or overdetermined system
                    M = np.zeros((m, r))
                    t = np.ones(m)
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        for d in range(r):
                            M[j, d] = U[a, d] * V[b, d]
                    # Solve M @ W[c,:] = t
                    try:
                        W[c, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                    except:
                        pass
                else:
                    # Underdetermined: find minimum norm solution
                    M = np.zeros((m, r))
                    t = np.ones(m)
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        for d in range(r):
                            M[j, d] = U[a, d] * V[b, d]
                    # Minimum norm solution: W[c,:] = M^T (M M^T)^{-1} t
                    try:
                        Mt = M.T
                        MMt = M @ Mt
                        # Add regularization
                        MMt += 1e-10 * np.eye(r)
                        W[c, :] = Mt @ np.linalg.solve(MMt, t)
                    except:
                        pass
            
            # Now fix V, W; solve for U (similar)
            for a in range(nm):
                # Determine which (b,c) pairs have T[a,b,c] = 1
                # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
                # So i = a // m, j = a % m
                # Then for each k: b = j*p+k, c = k*n+i
                i = a // m
                j = a % m
                
                # For each k in 0..p-1: b = j*p+k, c = k*n+i
                # Target entries: (b=j*p+k, c=k*n+i) for k=0..p-1
                # All other (b,c) should be 0
                
                # Build system: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
                # For target (b,c): = 1
                # For non-target: = 0
                
                # Use only target entries
                M = np.zeros((p, r))
                t = np.ones(p)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    for d in range(r):
                        M[k, d] = V[b, d] * W[c, d]
                
                if r <= p:
                    try:
                        U[a, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                    except:
                        pass
                else:
                    try:
                        Mt = M.T
                        MMt = M @ Mt
                        MMt += 1e-10 * np.eye(r)
                        U[a, :] = Mt @ np.linalg.solve(MMt, t)
                    except:
                        pass
            
            # Now fix U, W; solve for V
            for b in range(mp):
                # Determine which (a,c) pairs have T[a,b,c] = 1
                # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
                # So j = b // p, k = b % p
                # Then for each i: a = i*m+j, c = k*n+i
                j = b // p
                k = b % p
                
                # For each i in 0..n-1: a = i*m+j, c = k*n+i
                # Target entries: (a=i*m+j, c=k*n+i) for i=0..n-1
                # All other (a,c) should be 0
                
                M = np.zeros((n, r))
                t = np.ones(n)
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    for d in range(r):
                        M[i, d] = U[a, d] * W[c, d]
                
                if r <= n:
                    try:
                        V[b, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                    except:
                        pass
                else:
                    try:
                        Mt = M.T
                        MMt = M @ Mt
                        MMt += 1e-10 * np.eye(r)
                        V[b, :] = Mt @ np.linalg.solve(MMt, t)
                    except:
                        pass
        
        # Verify
        if verify_fast(U, V, W, tol=1e-6):
            return U, V, W, r
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            result = try_find_decomposition(r, max_time=10.0)
            if result is not None:
                U_sol, V_sol, W_sol, r_sol = result
                best_rank = r_sol
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
