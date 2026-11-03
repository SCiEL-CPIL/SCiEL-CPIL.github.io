"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 145  # 145s budget, 5s safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def check_validity(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        # T[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
        # Should equal 1 if j_u==j_v and i_u==w%n and k_v==w//n, else 0
        for i in range(n):
            for k in range(p):
                # For output position (i,k), check all (j,j') pairs
                for j in range(m):
                    for jp in range(m):
                        u = i * m + j
                        v = jp * p + k
                        w = k * n + i
                        # Compute sum_c U[u,c] * V[v,c] * W[w,c]
                        val = np.dot(np.dot(U[u, :], V[v, :]), W[w, :])
                        expected = 1.0 if (j == jp) else 0.0
                        if abs(val - expected) > tol:
                            return False
        return True
    
    def build_A_matrices(U, V):
        """Build the A matrices for each w.
        For fixed w, we need: for all (u,v), sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
        This is: (U[u,:] * V[v,:]) dot W[w,:] = T[u,v,w]
        Let X_u_v = U[u,:] * V[v,:] (element-wise product, shape r)
        Then: X_u_v dot W[w,:] = T[u,v,w]
        
        For fixed w, the system is:
        For all u,v: (U[u,:] * V[v,:]) dot W[w,:] = T[u,v,w]
        
        We can write this as: M_w @ W[w,:] = b_w
        where M_w has rows indexed by (u,v) and M_w[(u,v), c] = U[u,c]*V[v,c]
        and b_w[(u,v)] = T[u,v,w]
        """
        A_list = []
        b_list = []
        for w in range(p * n):
            i_out = w % n
            k_out = w // n
            M = np.zeros((n * m * m * p, r)) if 'r' in dir() else None
            b = np.zeros(n * m * m * p)
            row = 0
            for u in range(n * m):
                for v in range(m * p):
                    M[row, :] = U[u, :] * V[v, :]
                    i_u = u // m
                    j_u = u % m
                    j_v = v // p
                    k_v = v % p
                    if i_u == i_out and k_v == k_out and j_u == j_v:
                        b[row] = 1.0
                    else:
                        b[row] = 0.0
                    row += 1
            A_list.append(M)
            b_list.append(b)
        return A_list, b_list
    
    def try_rank_als(r, num_restarts=50, max_als_iters=200):
        """Try to find a valid decomposition with rank r using ALS."""
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Structured initialization: use basis vectors with some randomness
            np.random.seed(hash((r, restart)) % (2**31))
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(max_als_iters):
                if time.time() > deadline - 15:
                    break
                
                # Update W: for each w, solve M_w @ W[w,:] = b_w
                A_list, b_list = build_A_matrices(U, V)
                for w in range(p * n):
                    try:
                        W[w, :], _, _, _ = np.linalg.lstsq(A_list[w], b_list[w], rcond=1e-10)
                    except np.linalg.LinAlgError:
                        pass
                
                # Update U: for each u, the equations involving U[u,:]
                # For each (u, v, w): U[u,c]*V[v,c]*W[w,c] contributes to T[u,v,w]
                # Fixing V and W, for fixed u:
                # sum_v sum_w (V[v,c]*W[w,c]) * U[u,c] ... no, it's element-wise
                # Actually: for fixed u, the constraint is:
                # For all v, w: sum_c U[u,c] * V[v,c] * W[w,c] = T[u,v,w]
                # Let Y_v_w = V[v,:] * W[w,:] (element-wise, shape r)
                # Then: U[u,:] dot Y_v_w = T[u,v,w] for all v,w
                # This is a linear system in U[u,:]
                Y = np.zeros((m * p * p * n, r))
                b_u = np.zeros(m * p * p * n)
                row = 0
                for v in range(m * p):
                    for w in range(p * n):
                        Y[row, :] = V[v, :] * W[w, :]
                        i_u = u // m if False else None  # will set per u
                        row += 1
                
                # This is getting complex. Let me use a different approach.
                # Instead, do coordinate descent or use the full linear system.
                
                # Let me try a simpler ALS: alternate between U and V given W, and W given U,V
                # For W given U,V: already done above
                # For U given V,W: for each u, solve linear system
                for u in range(n * m):
                    i_u = u // m
                    j_u = u % m
                    # Build system for U[u,:]
                    # Variables: U[u, 0..r-1]
                    # Equations: for each (v, w): sum_c U[u,c] * V[v,c] * W[w,c] = T[u,v,w]
                    M_u = np.zeros((m * p * p * n, r))
                    b_u_vec = np.zeros(m * p * p * n)
                    row = 0
                    for v in range(m * p):
                        for w in range(p * n):
                            M_u[row, :] = V[v, :] * W[w, :]
                            j_v = v // p
                            k_v = v % p
                            i_w = w % n
                            k_w = w // n
                            if j_u == j_v and i_u == i_w and k_v == k_w:
                                b_u_vec[row] = 1.0
                            else:
                                b_u_vec[row] = 0.0
                            row += 1
                    try:
                        U[u, :], _, _, _ = np.linalg.lstsq(M_u, b_u_vec, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        pass
                
                # For V given U,W: for each v, solve linear system
                for v in range(m * p):
                    j_v = v // p
                    k_v = v % p
                    M_v = np.zeros((n * m * p * n, r))
                    b_v_vec = np.zeros(n * m * p * n)
                    row = 0
                    for u in range(n * m):
                        for w in range(p * n):
                            M_v[row, :] = U[u, :] * W[w, :]
                            i_u = u // m
                            j_u = u % m
                            i_w = w % n
                            k_w = w // n
                            if j_u == j_v and i_u == i_w and k_v == k_w:
                                b_v_vec[row] = 1.0
                            else:
                                b_v_vec[row] = 0.0
                            row += 1
                    try:
                        V[v, :], _, _, _ = np.linalg.lstsq(M_v, b_v_vec, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        pass
                
                # Check convergence
                if check_validity(U, V, W):
                    return U, V, W, r
            return None
    
    def try_rank_structured(r):
        """Try structured initializations that respect the tensor's algebraic structure."""
        # For matrix multiplication, we can use a Kronecker-like structure
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The tensor is T[i,j,k; i',j',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        # Wait no, for (n x m) * (m x p) -> (n x p):
        # T[i,j; j',k; i',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        
        # Try initialization based on random projections of the basis
        for attempt in range(30):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(hash((r, 'struct', attempt)) % (2**31))
            
            # Initialize with structured patterns
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Random assignment of which "terms" go into which rank component
            # Each rank component c can contribute to multiple (i,j,k) terms
            for c in range(r):
                # Randomly select a subset of (i,j) for U
                num_u = max(1, min(n*m, np.random.randint(1, 4)))
                u_indices = np.random.choice(n*m, num_u, replace=False)
                for u in u_indices:
                    U[u, c] = np.random.choice([-1, 1])
                
                num_v = max(1, min(m*p, np.random.randint(1, 4)))
                v_indices = np.random.choice(m*p, num_v, replace=False)
                for v in v_indices:
                    V[v, c] = np.random.choice([-1, 1])
                
                num_w = max(1, min(p*n, np.random.randint(1, 4)))
                w_indices = np.random.choice(p*n, num_w, replace=False)
                for w in w_indices:
                    W[w, c] = np.random.choice([-1, 1])
            
            # ALS refinement
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Update W
                A_list, b_list = build_A_matrices(U, V)
                for w in range(p * n):
                    try:
                        W[w, :], _, _, _ = np.linalg.lstsq(A_list[w], b_list[w], rcond=1e-10)
                    except np.linalg.LinAlgError:
                        pass
                
                # Update U
                for u in range(n * m):
                    i_u = u // m
                    j_u = u % m
                    M_u = np.zeros((m * p * p * n, r))
                    b_u_vec = np.zeros(m * p * p * n)
                    row = 0
                    for v in range(m * p):
                        for w in range(p * n):
                            M_u[row, :] = V[v, :] * W[w, :]
                            j_v = v // p
                            k_v = v % p
                            i_w = w % n
                            k_w = w // n
                            if j_u == j_v and i_u == i_w and k_v == k_w:
                                b_u_vec[row] = 1.0
                            else:
                                b_u_vec[row] = 0.0
                            row += 1
                    try:
                        U[u, :], _, _, _ = np.linalg.lstsq(M_u, b_u_vec, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        pass
                
                # Update V
                for v in range(m * p):
                    j_v = v // p
                    k_v = v % p
                    M_v = np.zeros((n * m * p * n, r))
                    b_v_vec = np.zeros(n * m * p * n)
                    row = 0
                    for u in range(n * m):
                        for w in range(p * n):
                            M_v[row, :] = U[u, :] * W[w, :]
                            i_u = u // m
                            j_u = u % m
                            i_w = w % n
                            k_w = w // n
                            if j_u == j_v and i_u == i_w and k_v == k_w:
                                b_v_vec[row] = 1.0
                            else:
                                b_v_vec[row] = 0.0
                            row += 1
                    try:
                        V[v, :], _, _, _ = np.linalg.lstsq(M_v, b_v_vec, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        pass
                
                if check_validity(U, V, W):
                    return U, V, W, r
            return None
    
    # Get schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try lower ranks using both random and structured approaches
    # Lower bound considerations:
    # The tensor has rank at least max(n*m, m*p, p*n) = 36
    # But likely higher due to the specific structure
    
    ranks_to_try = list(range(36, 144))
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try structured approach first
        result = try_rank_structured(r)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
        
        # Try random ALS
        result = try_rank_als(r, num_restarts=20)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
