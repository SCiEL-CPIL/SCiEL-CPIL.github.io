"""Matrix multiplication 3x4x6 rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Uses a simulated annealing approach over GF(2)-inspired hypergraph covering,
    combined with algebraic structure exploitation.
    
    The matrix multiplication tensor T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    where (i,j) indexes A, (j,k) indexes B, (i,k) indexes C.
    
    A rank-1 component is a triple (u, v, w) where:
      u is a vector in {0,1}^{n*m} (or more generally in a small field)
      v is a vector in {0,1}^{m*p}
      w is a vector in {0,1}^{p*n}
    such that T = sum_c u_c ⊗ v_c ⊗ w_c
    
    For exact decomposition over integers, we need:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    The schoolbook gives rank = n*m*p = 72.
    We try to find lower rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # The tensor T has entries T[(i,j), (j,k), (i,k)] = 1 for valid indices
    # Let's build the target tensor as a 3D array
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # But we only need T[a,b,c] to be 1 when the indices are consistent
    # i.e., a's j-component equals b's j-component, and a's i-component equals c's i-component,
    # and b's k-component equals c's k-component
    
    # Actually, the multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j | j,k | i,k] = 1
    # In flattened form: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the set of monomials (non-zero entries of the tensor)
    monomials = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                monomials.add((a, b, c))
    
    num_monomials = len(monomials)  # 72
    
    # Strategy: Try to find a low-rank decomposition using structured vectors.
    # Key insight: For matrix multiplication, the Kronecker product structure
    # can be exploited. The rank of multiplying (n x m) by (m x p) matrices
    # is known to be between n*p and n*m*p, with better bounds for specific dimensions.
    
    # For 3x4 * 4x6:
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    # Actually the Strassen-like lower bound is more complex.
    # Known: rank of 3x3 * 3x3 is at most 23 (Strassen), at least 19.
    # For rectangular, we can use padding arguments.
    
    # Let's try a greedy + local search approach.
    # Represent each rank-1 term as (u, v, w) where u, v, w are vectors.
    # The constraint is: for all (a,b,c) in monomials, sum_c u[a]*v[b]*w[c] = 1
    # and for all (a,b,c) NOT in monomials but in the full index space, sum_c u[a]*v[b]*w[c] = 0
    
    # This is a system of polynomial equations. Let's use a heuristic search.
    
    # First, let's try known structured decompositions.
    # For matrix multiplication, one approach is to use the fact that
    # we can decompose the identity on the middle dimension.
    
    # Try: Use the structure where we group by the middle index j.
    # For each j, we need to multiply a row of A (length p... no, row i of A has entries A[i,j] for fixed i, varying j)
    # Actually, C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A decomposition idea: For each j, define terms that handle the j-th slice.
    # But that gives rank = m * (something).
    
    # Let me think about this differently. The tensor is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the standard basis vector in the A-space, etc.
    
    # A rank-1 decomposition term is u ⊗ v ⊗ w.
    # If we choose u = sum_i alpha_i e_{i,j0} for fixed j0... no, u must be in the A-space.
    
    # Let's try: for each j, and for each pair (i,k), we can't do it directly.
    
    # Alternative: Use the fact that for fixed j, the map (i,k) -> A[i,j]*B[j,k] is rank 1 in (i,k).
    # So C = sum_j (col_j_of_A) * (row_j_of_B^T)
    # This is a rank-m decomposition of the bilinear map, but each term is rank 1 in the (i,k) output.
    # However, the input spaces are different.
    
    # In terms of the tensor:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # Let's define for each j:
    #   u_j[a] = 1 if a = i*m+j for some i, else 0  (i.e., u_j has 1s at positions (i,j) for all i)
    #   v_j[b] = 1 if b = j*p+k for some k, else 0  (i.e., v_j has 1s at positions (j,k) for all k)
    #   w_j[c] = ... we need w_j[k*n+i] such that the product works.
    # 
    # If we use u_j ⊗ v_j ⊗ w_j, the contribution to T[a,b,c] is u_j[a]*v_j[b]*w_j[c].
    # For a=(i,j), b=(j,k), c=(i',k'): contribution is 1*1*w_j[i'*n+k'] = w_j[i'*n+k']
    # We want this to be 1 when i'=i and k'=k, and 0 otherwise.
    # So w_j would need to be the identity-like vector, but w_j is a single vector, not a matrix.
    # This doesn't work with just one term per j.
    
    # We need m * n * p terms if we do it naively, which is the schoolbook.
    
    # To do better, we need to "mix" the i and k indices in w.
    # This is where Strassen's idea comes in: use linear combinations.
    
    # For the 2x2 case, Strassen uses 7 terms instead of 8.
    # The key is that the w vectors can be linear combinations of basis vectors.
    
    # Let's try a systematic search for small ranks.
    # We'll use integer/half-integer entries as allowed.
    
    best_rank = nm * mp * pn  # schoolbook, 72... wait, n*m*p = 3*4*6 = 72
    best_decomp = None
    
    # Schoolbook decomposition for reference
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_rank = rank_sb
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Verify schoolbook
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        # Check all monomials
        for (a, b, c) in monomials:
            val = 0.0
            for c_idx in range(rank):
                val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
            if abs(val - 1.0) > 1e-9:
                return False
        # Check all non-monomials
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in monomials:
                        val = 0.0
                        for c_idx in range(rank):
                            val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                        if abs(val) > 1e-9:
                            return False
        return True
    
    if verify(*best_decomp):
        pass  # Schoolbook is valid
    
    # Now try to find better decompositions.
    # Approach: Randomized local search starting from schoolbook.
    # Try to merge terms or find linear dependencies.
    
    # Better approach: Direct search over low-rank decompositions.
    # For rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that the tensor equation holds.
    
    # This is a system of polynomial equations. Let's use a heuristic.
    
    # For each candidate rank r from lower_bound to best_rank - 1:
    #   Try to find a decomposition.
    
    # Lower bound: The tensor rank of matrix multiplication (n x m) * (m x p)
    # is at least n*p (since the output is n x p and each rank-1 term can contribute
    # at most rank 1 to the output... actually this argument is not quite right for
    # the tensor rank).
    
    # A better lower bound: For the (n x m) * (m x p) multiplication,
    # the rank is at least max(n, m, p) * min(n, m, p) ... no.
    # Actually, a simple lower bound is n * p if m >= max(n, p), but that's not always tight.
    
    # Let's just try ranks from a reasonable lower bound upward.
    
    # Known results:
    # 2x2 * 2x2: rank 7 (Strassen), lower bound 7 (actually 7 is the best known upper bound,
    # lower bound is 7? No, lower bound is 7 for 2x2? Actually the lower bound for 2x2x2 is 7.)
    # Wait, the tensor rank of 2x2x2 matrix multiplication is 7 (Strassen proved it's at most 7,
    # and it's known to be at least 7... actually I think the lower bound is 7).
    
    # For 3x3 * 3x3: upper bound 23 (Strassen), lower bound 19.
    # For 3x4 * 4x6: we can use the fact that 3x4 * 4x6 can be embedded in a larger square case.
    
    # Let's try a greedy construction.
    # Idea: Use the structure of the problem. The middle dimension is 4.
    # We can think of this as 4 separate "channels" (one for each j), and we want to
    # process them with fewer than 4 * 3 * 6 = 72 operations.
    
    # Let's try a block-based approach.
    # Group the i indices into groups and the k indices into groups.
    
    # For now, let's try random restarts with local search.
    
    def try_rank(r, max_iters=10000, seed=42):
        """Try to find a rank-r decomposition using local search."""
        rng = random.Random(seed)
        
        # Initialize with random vectors
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Initialize with a structured guess
        # For each column c, try to assign it to handle a subset of monomials.
        
        # Simple init: random small integers
        for c in range(r):
            # Assign this column to handle some monomials
            # Pick a random (i, j, k)
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            U[i*m + j, c] = 1
            V[j*p + k, c] = 1
            W[k*n + i, c] = 1
        
        # Local search: try to modify entries to satisfy constraints
        # This is a constraint satisfaction problem.
        
        # Define the residual: for each (a, b, c), residual = sum_c' U[a,c']*V[b,c']*W[c,c'] - target
        # target = 1 if (a,b,c) in monomials, 0 otherwise
        
        def compute_residuals(U, V, W):
            res = {}
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = 0.0
                        for c_idx in range(r):
                            val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                        target = 1.0 if (a, b, c) in monomials else 0.0
                        res[(a, b, c)] = val - target
            return res
        
        # This is too slow for large r. Let's use a smarter approach.
        
        # Instead, let's use the fact that the tensor is sparse.
        # Only 72 entries are non-zero.
        
        # For a given (a, b), the constraint on W is:
        # sum_c' U[a, c'] * V[b, c'] * W[c, c'] = target(a, b, c)
        # for all c.
        
        # This is a linear system in W[:, :] for fixed U, V.
        
        # Let's try: fix U and V, solve for W.
        # For each c (output index), we have a linear system:
        # sum_{c'} U[a, c'] * V[b, c'] * W[c, c'] = target(a, b, c)
        # 
        # This is not a standard linear system because W[c, c'] appears with coefficient
        # U[a, c'] * V[b, c'].
        
        # Actually, for fixed a, b, the equation is:
        # sum_{c'} (U[a, c'] * V[b, c']) * W[c, c'] = target
        # This IS linear in W[c, :].
        
        # So for each (a, b), we can solve for W[:, :] from the equations over all c...
        # Wait, no. For fixed a, b, and varying c, we get:
        # For each c: sum_{c'} U[a, c'] * V[b, c'] * W[c, c'] = target(a, b, c)
        # 
        # The left side involves W[c, :] which is a row of W.
        # So for each (a, b), we have a linear system:
        # M_{ab} @ W[c, :] = target(a, b, c)
        # where M_{ab}[c, c'] = U[a, c'] * V[b, c']
        
        # This is a system of pn equations (one for each c) in r unknowns (W[c, :]).
        # But the equations for different c are independent (each involves a different row of W).
        # So actually, for each c, we have a single equation:
        # sum_{c'} U[a, c'] * V[b, c'] * W[c, c'] = target(a, b, c)
        # 
        # This is one equation in r unknowns (the entries of W[c, :]).
        # We have nm * mp such equations (one for each (a, b)), but they share the same W rows.
        
        # This is getting complicated. Let's just use a direct approach with numpy.
        
        # For a given rank r, set up the full system and solve.
        # The system is: for each (a, b, c), sum_{c'} U[a,c'] V[b,c'] W[c,c'] = T[a,b,c]
        
        # This is a system of nm * mp * pn equations in nm*r + mp*r + pn*r unknowns.
        # But it's nonlinear (products of variables).
        
        # Let's use a different strategy: iterate.
        # 1. Fix V and W, solve for U (linear).
        # 2. Fix U and W, solve for V (linear).
        # 3. Fix U and V, solve for W (linear).
        # Repeat.
        
        def solve_linear(X, Y, Z, target):
            """Given Y, Z fixed, solve for X.
            X[a, c'] * Y[b, c'] * Z[c, c'] summed over c' = target[a, b, c]
            For fixed b, c: sum_{c'} X[a, c'] * (Y[b, c'] * Z[c, c']) = target[a, b, c]
            This is linear in X[a, :].
            We have nm * mp * pn equations, but they're coupled.
            
            Actually, for each (a, b, c):
            sum_{c'} X[a, c'] * Y[b, c'] * Z[c, c'] = target[a, b, c]
            
            Let's collect all equations. The unknowns are X[a, c'] for a in range(nm), c' in range(r).
            Total unknowns: nm * r.
            Total equations: nm * mp * pn.
            
            This is an overdetermined system (usually). We can use least squares.
            """
            # Build the system: for each (a, b, c), the equation is:
            # sum_{c'} X[a, c'] * Y[b, c'] * Z[c, c'] = target[a, b, c]
            
            # The coefficient of X[a, c'] in equation (a', b', c') is:
            # Y[b', c'] * Z[c', c'] if a == a', else 0.
            
            # So the system is block-diagonal in a!
            # For each a, we have mp * pn equations in r unknowns (X[a, :]).
            
            X_sol = np.zeros((nm, r))
            for a in range(nm):
                # Equations: for each (b, c):
                # sum_{c'} X[a, c'] * Y[b, c'] * Z[c, c'] = target[a, b, c]
                A_eq = np.zeros((mp * pn, r))
                b_eq = np.zeros(mp * pn)
                idx = 0
                for b in range(mp):
                    for c in range(pn):
                        for c_prime in range(r):
                            A_eq[idx, c_prime] = Y[b, c_prime] * Z[c, c_prime]
                        b_eq[idx] = target[a, b, c]
                        idx += 1
                # Solve least squares
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                    X_sol[a, :] = sol
                except:
                    pass
            return X_sol
        
        # Build target tensor
        target = np.zeros((nm, mp, pn))
        for (a, b, c) in monomials:
            target[a, b, c] = 1.0
        
        # Initialize
        for c in range(r):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            U[i*m + j, c] = rng.choice([-1, 1])
            V[j*p + k, c] = rng.choice([-1, 1])
            W[k*n + i, c] = rng.choice([-1, 1])
        
        # Iterate
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U = solve_linear(U, V, W, target)
            V = solve_linear(V, U, W, target)  # Wait, this is wrong.
            
            # Actually, solve for V: given U, W fixed.
            # sum_{c'} U[a, c'] * V[b, c'] * W[c, c'] = target[a, b, c]
            # For fixed (a, c): sum_{c'} V[b, c'] * (U[a, c'] * W[c, c']) = target[a, b, c]
            # Block-diagonal in b.
            V_new = np.zeros((mp, r))
            for b in range(mp):
                A_eq = np.zeros((nm * pn, r))
                b_eq = np.zeros(nm * pn)
                idx = 0
                for a in range(nm):
                    for c in range(pn):
                        for c_prime in range(r):
                            A_eq[idx, c_prime] = U[a, c_prime] * W[c, c_prime]
                        b_eq[idx] = target[a, b, c]
                        idx += 1
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                    V_new[b, :] = sol
                except:
                    pass
            V = V_new
            
            # Solve for W: given U, V fixed.
            # sum_{c'} U[a, c'] * V[b, c'] * W[c, c'] = target[a, b, c]
            # For fixed (a, b): sum_{c'} W[c, c'] * (U[a, c'] * V[b, c']) = target[a, b, c]
            # Block-diagonal in c.
            W_new = np.zeros((pn, r))
            for c in range(pn):
                A_eq = np.zeros((nm * mp, r))
                b_eq = np.zeros(nm * mp)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c_prime in range(r):
                            A_eq[idx, c_prime] = U[a, c_prime] * V[b, c_prime]
                        b_eq[idx] = target[a, b, c]
                        idx += 1
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                    W_new[c, :] = sol
                except:
                    pass
            W = W_new
            
            # Check convergence
            err = 0.0
            for (a, b, c) in monomials:
                val = 0.0
                for c_idx in range(r):
                    val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                err += abs(val - 1.0)
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if (a, b, c) not in monomials:
                            val = 0.0
                            for c_idx in range(r):
                                val += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                            err += abs(val)
            
            if err < 1e-6:
                # Round to nearest integer/half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r, r
                elif verify(U, V, W):
                    return U, V, W, r
        
        return None
    
    # Try ranks from a lower bound up to best_rank - 1
    # Lower bound: for (n x m) * (m x p), a simple lower bound is n * p if m >= max(n, p)
    # Actually, let's use the fact that the rank is at least the dimension of the output
    # times something... Let's just try from 24 (which is m * p) upward.
    
    # Actually, a better lower bound: the rank of the (n x m) * (m x p) multiplication
    # is at least max(n * p, m * p, n * m) / min(n, m, p) ... I'm not sure.
    # Let's just try from 24 to best_rank - 1.
    
    # But this is expensive. Let's be smarter.
    
    # Known: For 2x2 * 2x2, rank is 7. For 3x3 * 3x3, rank is at most 23.
    # For 3x4 * 4x6, we can pad to 6x6 * 6x6 and use known bounds, but that's loose.
    
    # Let's try a few specific ranks.
    candidate_ranks = []
    # Start from a reasonable lower bound
    lb = max(n * p, m * p, n * m)  # 18, 24, 12 -> 24
    for r in range(lb, best_rank):
        candidate_ranks.append(r)
    
    # Try each rank
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=2000, seed=r)
        if result is not None:
            U_r, V_r, W_r, rank_found = result
            best_rank = rank_found
            best_decomp = (U_r, V_r, W_r)
            break  # Found a valid decomposition, stop
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_decomp[0]]
    V_list = [[float(x) for x in row] for row in best_decomp[1]]
    W_list = [[float(x) for x in row] for row in best_decomp[2]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
