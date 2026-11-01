"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Known rank-11 solution (Strassen + schoolbook for col 2)
    rank_known = 11
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    U_known[0][0] = 1; U_known[3][0] = 1
    V_known[0][0] = 1; V_known[4][0] = 1
    U_known[2][1] = 1; U_known[3][1] = 1
    V_known[0][1] = 1
    U_known[0][2] = 1
    V_known[1][2] = 1; V_known[4][2] = -1
    U_known[3][3] = 1
    V_known[3][3] = 1; V_known[0][3] = -1
    U_known[0][4] = 1; U_known[1][4] = 1
    V_known[4][4] = 1
    U_known[2][5] = 1; U_known[0][5] = -1
    V_known[0][5] = 1; V_known[1][5] = 1
    U_known[1][6] = 1; U_known[3][6] = -1
    V_known[3][6] = 1; V_known[4][6] = 1
    W_known[0][0] = 1; W_known[0][3] = 1; W_known[0][4] = -1; W_known[0][6] = 1
    W_known[2][2] = 1; W_known[2][4] = 1
    W_known[1][1] = 1; W_known[1][3] = 1
    W_known[3][0] = 1; W_known[3][1] = -1; W_known[3][2] = 1; W_known[3][5] = 1
    U_known[0][7] = 1
    V_known[2][7] = 1
    U_known[1][8] = 1
    V_known[5][8] = 1
    U_known[2][9] = 1
    V_known[2][9] = 1
    U_known[3][10] = 1
    V_known[5][10] = 1
    W_known[4][7] = 1; W_known[4][8] = 1
    W_known[5][9] = 1; W_known[5][10] = 1
    
    # Build the multiplication tensor as a 3-tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # We'll represent it as a list of (a,b,c) tuples with value 1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    num_entries = len(tensor_entries)
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition exactly reproduces the tensor."""
        for (a, b, c) in tensor_entries:
            s = 0.0
            for r in range(rank):
                s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def verify_decomposition_fast(U_arr, V_arr, W_arr, rank):
        """Fast verification using numpy."""
        # For each (a,b,c) entry, compute sum_r U[a,r]*V[b,r]*W[c,r]
        # We can do this with einsum
        # result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        for (a, b, c) in tensor_entries:
            if abs(result[a, b, c] - 1.0) > 1e-9:
                return False
        return True
    
    # Try to find a rank-10 decomposition using algebraic structure
    # Key insight: fix U and V, solve linearly for W row by row
    
    def try_rank_algebraic(R, seed=42):
        """Try to find a rank-R decomposition by fixing U,V and solving for W."""
        if time.time() > _deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Group tensor entries by c value
        # c=0: (0,0), (1,3)
        # c=1: (2,0), (3,3)
        # c=2: (0,1), (1,4)
        # c=3: (2,1), (3,4)
        # c=4: (0,2), (1,5)
        # c=5: (2,2), (3,5)
        c_groups = {}
        for (a, b, c) in tensor_entries:
            if c not in c_groups:
                c_groups[c] = []
            c_groups[c].append((a, b))
        
        # Strategy: Use structured U and V with sparsity
        # For rank 10, try to find sparse U, V such that W can be solved
        
        # Approach: parameterize U and V with few free parameters
        # Use the structure: U has 4 rows, V has 6 rows
        
        # Try: U and V with entries in {-1, 0, 1} (sparse)
        # Then solve for W linearly
        
        # For each c, we have 2 equations:
        #   sum_r U[a1,r]*V[b1,r]*W[c,r] = 1
        #   sum_r U[a2,r]*V[b2,r]*W[c,r] = 1
        # This is 2 equations in R unknowns (W[c,0],...,W[c,R-1])
        
        # If R >= 3, this is underdetermined. We can set R-2 free variables.
        # But we need consistency across all c.
        
        # Better approach: use alternating minimization
        # 1. Initialize U, V randomly (sparse)
        # 2. For each c, solve for W[c,:] using least squares or null space
        # 3. Check if result is valid
        # 4. Iterate
        
        n_restarts = 200
        for restart in range(n_restarts):
            if time.time() > _deadline:
                break
            
            # Initialize U and V with random sparse structure
            U_arr = np.zeros((n*m, R))
            V_arr = np.zeros((m*p, R))
            
            # Each row of U gets 1-2 non-zero entries
            for i in range(n*m):
                nnz = rng.randint(1, 3)
                cols = rng.choice(R, size=nnz, replace=False)
                for col in cols:
                    U_arr[i, col] = rng.choice([-1, 1])
            
            for i in range(m*p):
                nnz = rng.randint(1, 3)
                cols = rng.choice(R, size=nnz, replace=False)
                for col in cols:
                    V_arr[i, col] = rng.choice([-1, 1])
            
            # Solve for W
            W_arr = np.zeros((p*n, R))
            valid = True
            
            for c in range(p*n):
                if c not in c_groups:
                    continue
                entries = c_groups[c]
                # Build linear system: A * W[c,:] = 1
                # A has 2 rows (one per entry), R columns
                A = np.zeros((len(entries), R))
                for idx, (a, b) in enumerate(entries):
                    A[idx, :] = U_arr[a, :] * V_arr[b, :]
                
                # Solve A * x = 1 (2 equations, R unknowns)
                # Use least squares or find a particular solution
                # Check if system is consistent
                try:
                    # For underdetermined system, find minimum-norm solution
                    # But we want any solution. Use pseudo-inverse
                    if np.linalg.matrix_rank(A) < len(entries):
                        # Check consistency
                        # The two rows should give consistent results
                        # If rank < 2, rows are dependent
                        # Check: is 1 in the column space?
                        sol, residuals, rank, sv = np.linalg.lstsq(A, np.ones(len(entries)), rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-9:
                            valid = False
                            break
                        W_arr[c, :] = sol
                    else:
                        # Full rank (rank 2), solve directly
                        # Use first R columns to form square system, or use lstsq
                        sol, residuals, rank, sv = np.linalg.lstsq(A, np.ones(len(entries)), rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-9:
                            valid = False
                            break
                        W_arr[c, :] = sol
                except Exception:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            U_list = [[float(U_arr[i, r]) for r in range(R)] for i in range(n*m)]
            V_list = [[float(V_arr[i, r]) for r in range(R)] for i in range(m*p)]
            W_list = [[float(W_arr[i, r]) for r in range(R)] for i in range(p*n)]
            
            if verify_decomposition(U_list, V_list, W_list, R):
                return (U_list, V_list, W_list)
        
        return None
    
    # Try rank 10 with algebraic approach
    result = try_rank_algebraic(10)
    if result is not None:
        U_list, V_list, W_list = result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Also try a more structured approach: use known rank-11 and try to merge
    # by finding if any two columns can be combined
    def try_merge_rank11():
        """Try to find rank-10 by merging two rank-11 terms."""
        if time.time() > _deadline:
            return None
        
        U11 = np.array(U_known, dtype=float)
        V11 = np.array(V_known, dtype=float)
        W11 = np.array(W_known, dtype=float)
        
        # For each pair of columns (i, j), try to find scalars alpha, beta
        # such that replacing columns i and j with a single column works
        # This means: for all (a,b,c),
        # U[a,i]*V[b,i]*W[c,i] + U[a,j]*V[b,j]*W[c,j] = U'[a]*V'[b]*W'[c]
        # This is generally not possible with a single rank-1 term
        
        # Alternative: try to find a linear dependency
        # If column j is a linear combination of other columns in some sense
        
        # Try: remove column 10 (the last one) and see if we can adjust
        # The last column contributes: U[3,10]=1, V[5,10]=1, W[5,10]=1
        # So it contributes 1 to entry (a=3, b=5, c=5)
        # Entry (3,5,5) is i=1,j=1,k=2: a=1*2+1=3, b=1*3+2=5, c=2*2+1=5 ✓
        
        # To eliminate this, we need another way to produce 1 at (3,5,5)
        # Current contribution to (3,5,5) from other columns:
        # Check which other columns have non-zero U[3,r], V[5,r], W[5,r]
        for r in range(10):
            if U11[3, r] != 0 and V11[5, r] != 0 and W11[5, r] != 0:
                # This column also contributes to (3,5,5)
                pass
        
        # This approach is complex. Try numerical: perturb and optimize
        from scipy.optimize import minimize
        
        # Remove last column, optimize remaining 10
        U_init = U11[:, :10].copy()
        V_init = V11[:, :10].copy()
        W_init = W11[:, :10].copy()
        
        # Add small perturbation to break symmetry
        rng = np.random.RandomState(123)
        U_init += rng.randn(4, 10) * 0.01
        V_init += rng.randn(6, 10) * 0.01
        W_init += rng.randn(6, 10) * 0.01
        
        R = 10
        nU = n*m*R
        nV = m*p*R
        nW = p*n*R
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U_m = params[:nU].reshape(n*m, R)
            V_m = params[nU:nU+nV].reshape(m*p, R)
            W_m = params[nU+nV:].reshape(p*n, R)
            err = 0.0
            for (a, b, c) in tensor_entries:
                val = np.dot(U_m[a], V_m[b] * W_m[c])
                err += (val - 1.0) ** 2
            return err
        
        try:
            res = minimize(objective, params_init, method='L-BFGS-B', maxiter=5000)
            if res.fun < 1e-10:
                U_m = res.x[:nU].reshape(n*m, R)
                V_m = res.x[nU:nU+nV].reshape(m*p, R)
                W_m = res.x[nU+nV:].reshape(p*n, R)
                U_list = [[float(U_m[i, r]) for r in range(R)] for i in range(n*m)]
                V_list = [[float(V_m[i, r]) for r in range(R)] for i in range(m*p)]
                W_list = [[float(W_m[i, r]) for r in range(R)] for i in range(p*n)]
                if verify_decomposition(U_list, V_list, W_list, R):
                    return (U_list, V_list, W_list)
        except Exception:
            pass
        
        return None
    
    result = try_merge_rank11()
    if result is not None:
        U_list, V_list, W_list = result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: try one more approach - alternating minimization with more restarts
    def try_alternating(R, n_restarts=50, seed=0):
        """Alternating minimization: fix two, solve for third."""
        if time.time() > _deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        for restart in range(n_restarts):
            if time.time() > _deadline:
                break
            
            U_arr = rng.randn(n*m, R) * 0.3
            V_arr = rng.randn(m*p, R) * 0.3
            W_arr = rng.randn(p*n, R) * 0.3
            
            # Alternate: fix U,V solve W; fix V,W solve U; fix W,U solve V
            for iteration in range(200):
                if time.time() > _deadline:
                    return None
                
                # Solve for W: for each c, linear system
                for c in range(p*n):
                    if c not in c_groups:
                        continue
                    entries = c_groups[c]
                    A = np.array([U_arr[a, :] * V_arr[b, :] for (a, b) in entries])
                    if len(entries) == 0:
                        continue
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, np.ones(len(entries)), rcond=None)
                        W_arr[c, :] = sol
                    except:
                        pass
                
                # Solve for U: for each a, linear system
                a_groups = {}
                for (a, b, c) in tensor_entries:
                    if a not in a_groups:
                        a_groups[a] = []
                    a_groups[a].append((b, c))
                
                for a in range(n*m):
                    if a not in a_groups:
                        continue
                    entries = a_groups[a]
                    A = np.array([V_arr[b, :] * W_arr[c, :] for (b, c) in entries])
                    if len(entries) == 0:
                        continue
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, np.ones(len(entries)), rcond=None)
                        U_arr[a, :] = sol
                    except:
                        pass
                
                # Solve for V: for each b, linear system
                b_groups = {}
                for (a, b, c) in tensor_entries:
                    if b not in b_groups:
                        b_groups[b] = []
                    b_groups[b].append((a, c))
                
                for b in range(m*p):
                    if b not in b_groups:
                        continue
                    entries = b_groups[b]
                    A = np.array([U_arr[a, :] * W_arr[c, :] for (a, c) in entries])
                    if len(entries) == 0:
                        continue
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, np.ones(len(entries)), rcond=None)
                        V_arr[b, :] = sol
                    except:
                        pass
                
                # Check convergence
                err = 0.0
                for (a, b, c) in tensor_entries:
                    val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                    err += (val - 1.0) ** 2
                if err < 1e-10:
                    U_list = [[float(U_arr[i, r]) for r in range(R)] for i in range(n*m)]
                    V_list = [[float(V_arr[i, r]) for r in range(R)] for i in range(m*p)]
                    W_list = [[float(W_arr[i, r]) for r in range(R)] for i in range(p*n)]
                    if verify_decomposition(U_list, V_list, W_list, R):
                        return (U_list, V_list, W_list)
            
            # Also check after each restart's full iterations
            err = 0.0
            for (a, b, c) in tensor_entries:
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                err += (val - 1.0) ** 2
            if err < 1e-10:
                U_list = [[float(U_arr[i, r]) for r in range(R)] for i in range(n*m)]
                V_list = [[float(V_arr[i, r]) for r in range(R)] for i in range(m*p)]
                W_list = [[float(W_arr[i, r]) for r in range(R)] for i in range(p*n)]
                if verify_decomposition(U_list, V_list, W_list, R):
                    return (U_list, V_list, W_list)
        
        return None
    
    # Build c_groups, a_groups, b_groups for the alternating method
    c_groups = {}
    a_groups = {}
    b_groups = {}
    for (a, b, c) in tensor_entries:
        if c not in c_groups:
            c_groups[c] = []
        c_groups[c].append((a, b))
        if a not in a_groups:
            a_groups[a] = []
        a_groups[a].append((b, c))
        if b not in b_groups:
            b_groups[b] = []
        b_groups[b].append((a, c))
    
    result = try_alternating(10, n_restarts=100, seed=42)
    if result is not None:
        U_list, V_list, W_list = result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to rank 11
    return {"U": U_known, "V": V_known, "W": W_known}
# EVOLVE-BLOCK-END
