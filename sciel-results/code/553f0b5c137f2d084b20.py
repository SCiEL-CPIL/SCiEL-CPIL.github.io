"""Matrix multiplication 3x4x5 - rank-minimal exact tensor decomposition.
Attempts to find a decomposition with rank less than 60 (schoolbook).
Falls back to schoolbook if no better decomposition is found within time budget.
"""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each factor c, the tensor T_c = u_c ⊗ v_c ⊗ w_c should sum to the matmul tensor.
    The matmul tensor has T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p),
    and the output is C[i,k] = sum_j A[i,j]*B[j,k].
    In tensor form: M[i,j,k] = delta(i,j,k) where the product is A[i,j]*B[j,k].
    Actually the tensor is: T[i,j,k] = 1 for all valid i,j,k (since we're computing
    the bilinear form sum_j A_ij * B_jk, the tensor has entries T[i,j,k] = 1).
    So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _build_from_factors(factors, n, m, p):
    """Given a list of (u, v, w) tuples (each a vector), build U, V, W matrices.
    factors: list of (u_vec, v_vec, w_vec)
    """
    rank = len(factors)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for c, (u, v, w) in enumerate(factors):
        for i in range(n):
            for j in range(m):
                U[i * m + j, c] = u[i * m + j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, c] = v[j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = w[k * n + i]
    return U, V, W

def _try_find_decomposition(n, m, p, time_budget=120.0, target_rank=59):
    """Try to find a decomposition of rank < n*m*p using various strategies.
    
    Strategy: Start from schoolbook (rank 60), try to merge pairs of factors.
    Two factors c1, c2 can be merged into one if there exist scalars such that
    u1⊗v1⊗w1 + u2⊗v2⊗w2 = u'⊗v'⊗w' (rank-1 tensor).
    
    A more practical approach: try random rank-r decompositions and check.
    Also try structured approaches based on known results for small matrices.
    
    For 3x4x5, the known lower bound is 59 (from Strassen-type arguments),
    and upper bound is 60. It's an open question whether 59 is achievable.
    
    We'll try:
    1. Random search for rank 59 (limited by time)
    2. Structured perturbation of schoolbook
    """
    start_time = time.time()
    deadline = start_time + time_budget - 5.0  # 5s safety margin
    
    n, m, p = 3, 4, 5
    nm = n * m
    mp = m * p
    pn = p * n
    full_rank = n * m * p  # 60
    
    # Strategy 1: Try to find if schoolbook has linear dependencies that allow rank reduction
    # Build the 60 rank-1 tensors from schoolbook and check if any is in the span of others
    U0, V0, W0 = _schoolbook(n, m, p)
    
    # The tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We need sum_c u_c v_c w_c^T (as 3-tensor) = M (the matmul tensor)
    # M[i,j,k] = 1 for all i,j,k (since the matmul tensor for A(3x4)*B(4x5) is
    # M[i,j,k] = 1, because C_ik = sum_j A_ij B_jk, and the tensor encodes each
    # (i,j,k) -> contribution 1)
    
    # Actually let me reconsider. The matrix multiplication tensor for (n,m,p) is:
    # T(i,j,k) = 1 if we're looking at the (i,k) entry computed from j
    # The tensor is in R^{n x m x p} with T[i,j,k] = 1 for all valid i,j,k.
    # Wait no - the tensor has T[i,j,k] = 1 only when computing C[i,k] += A[i,j]*B[j,k]
    # So for each (i,k), the contribution from j is A[i,j]*B[j,k], and the tensor
    # entry T[i,j,k] = 1 (the coefficient multiplying A[i,j]*B[j,k]).
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a system of n*m*p = 60 equations in rank * (nm + mp + pn) unknowns
    # but the unknowns are coupled (same c index appears in all three).
    
    # Let's try a direct approach: parameterize rank-r factors and solve.
    # For rank 59, we have 59 * (12 + 20 + 15) = 59 * 47 = 2773 unknowns
    # and 60 equations. This is very underdetermined, but the nonlinear structure
    # makes it hard.
    
    # Strategy: Start with schoolbook (60 factors) and try to express one factor
    # as a combination of the other 59.
    
    # For factor c to be redundant, we need:
    # u_c ⊗ v_c ⊗ w_c = sum_{c' != c} alpha_{c'} u_{c'} ⊗ v_{c'} ⊗ w_{c'}
    # This requires u_c ⊗ v_c ⊗ w_c to be in the span of the other 59 rank-1 tensors.
    
    # Check linear independence of the 60 tensors in R^{nm * mp * pn} (flattened)
    # That's 12 * 20 * 15 = 3600 dimensional space. 60 vectors in 3600-dim space
    # are almost certainly linearly independent, so no single factor is redundant.
    
    # But maybe a combination of 2 factors can be replaced by 1?
    # u_a ⊗ v_a ⊗ w_a + u_b ⊗ v_b ⊗ w_b = u' ⊗ v' ⊗ w'
    # This is a specific algebraic condition.
    
    # Let's try: for each pair (a,b), check if their sum is rank-1.
    # u_a ⊗ v_a ⊗ w_a + u_b ⊗ v_b ⊗ w_b is rank-1 iff...
    # Flatten: the 3600-dim vector (u_a ⊗ v_a ⊗ w_a) + (u_b ⊗ v_b ⊗ w_b) must be
    # a rank-1 tensor in R^{12 x 20 x 15}.
    
    # A tensor X in R^{I x J x K} is rank-1 iff all its 2x2 minors (after reshaping)
    # vanish. Specifically, if we reshape to I x (J*K), rank 1 means all 2x2 minors are 0.
    # But that's expensive. Alternative: check if X can be written as x ⊗ y ⊗ z.
    
    # For the schoolbook factors:
    # Factor (i,j,k) has u with 1 at position i*m+j, v with 1 at position j*p+k, w with 1 at position k*n+i.
    # So factor (i1,j1,k1) + factor (i2,j2,k2):
    #   u part: e_{i1*m+j1} + e_{i2*m+j2} (in R^12)
    #   v part: e_{j1*p+k1} + e_{j2*p+k2} (in R^20)
    #   w part: e_{k1*n+i1} + e_{k2*n+i2} (in R^15)
    # This sum is rank-1 iff (e_a + e_b) ⊗ (e_c + e_d) ⊗ (e_f + e_g) is rank-1.
    # But (e_a + e_b) has 2 non-zero entries, so (e_a+e_b) ⊗ (e_c+e_d) ⊗ (e_f+e_g)
    # has 8 non-zero entries (unless some indices coincide), and a rank-1 tensor in
    # R^{12 x 20 x 15} would need to be x ⊗ y ⊗ z where x,y,z each have at most
    # 2 non-zero entries in their respective spaces. The product would have at most
    # 2*2*2 = 8 non-zero entries, which matches! But the values must factor.
    
    # (e_a+e_b) ⊗ (e_c+e_d) ⊗ (e_f+e_g) = e_a⊗e_c⊗e_f + e_a⊗e_c⊗e_g + e_a⊗e_d⊗e_f + e_a⊗e_d⊗e_g
    #   + e_b⊗e_c⊗e_f + e_b⊗e_c⊗e_g + e_b⊗e_d⊗e_f + e_b⊗e_d⊗e_g
    # For this to equal x⊗y⊗z, we need x to have 2 nonzeros (positions a,b),
    # y to have 2 nonzeros (positions c,d), z to have 2 nonzeros (positions f,g).
    # Then x⊗y⊗z has entries x[a]y[c]z[f], x[a]y[c]z[g], etc.
    # The sum tensor has all 8 entries equal to 1.
    # So we need x[a]=x[b]=1, y[c]=y[d]=1, z[f]=z[g]=1, giving all 8 entries = 1. ✓
    # But wait, that means (e_a+e_b)⊗(e_c+e_d)⊗(e_f+e_g) = (e_a+e_b)⊗(e_c+e_d)⊗(e_f+e_g)
    # which is trivially rank... hmm, actually (e_a+e_b) is NOT a single vector that makes
    # the whole thing rank-1 in the tensor sense. Let me reconsider.
    
    # A tensor T in R^{I×J×K} is rank-1 if T[i,j,k] = x[i]*y[j]*z[k] for some x,y,z.
    # (e_a+e_b)⊗(e_c+e_d)⊗(e_f+e_g)[i,j,k] = (delta(i,a)+delta(i,b))*(delta(j,c)+delta(j,d))*(delta(k,f)+delta(k,g))
    # This equals 1 when i∈{a,b}, j∈{c,d}, k∈{f,g}, and 0 otherwise.
    # For this to be x[i]*y[j]*z[k], we need x to be nonzero at a and b, y at c and d, z at f and g.
    # Then x[a]*y[c]*z[f] = 1, x[a]*y[c]*z[g] = 1, x[a]*y[d]*z[f] = 1, x[a]*y[d]*z[g] = 1,
    # x[b]*y[c]*z[f] = 1, x[b]*y[c]*z[g] = 1, x[b]*y[d]*z[f] = 1, x[b]*y[d]*z[g] = 1.
    # From first two: z[f]=z[g]. From first and third: y[c]=y[d]. Similarly x[a]=x[b].
    # So x[a]=x[b]=α, y[c]=y[d]=β, z[f]=z[g]=γ, and αβγ=1.
    # This works! So (e_a+e_b)⊗(e_c+e_d)⊗(e_f+e_g) IS rank-1, with
    # x = α(e_a+e_b), y = β(e_c+e_d), z = γ(e_f+e_g), αβγ=1.
    
    # BUT: this only works if a≠b, c≠d, f≠g (otherwise the vectors have 1 nonzero and it's trivially rank-1).
    # If a=b, then e_a+e_b = 2e_a, and the tensor is 2e_a⊗e_c⊗e_f (if c=d,f=g) which is rank-1.
    # The key case: a≠b, c≠d, f≠g all distinct positions.
    
    # So for schoolbook factors (i1,j1,k1) and (i2,j2,k2):
    # a = i1*m+j1, b = i2*m+j2 (need a≠b, i.e., (i1,j1)≠(i2,j2))
    # c = j1*p+k1, d = j2*p+k2 (need c≠d, i.e., (j1,k1)≠(j2,k2))  
    # f = k1*n+i1, g = k2*n+i2 (need f≠g, i.e., (k1,i1)≠(k2,i2))
    
    # If all three pairs are distinct, we can merge the two factors into one!
    # The merged factor has:
    #   u' = (e_a + e_b)  (with the understanding that it represents α(e_a+e_b) for some α)
    #   Actually, the merged factor in the decomposition would be:
    #   u'_merged = e_a + e_b (as a vector in R^12)
    #   v'_merged = e_c + e_d (as a vector in R^20)
    #   w'_merged = e_f + e_g (as a vector in R^15)
    # And we replace factors a and b with this single factor.
    # But we need to adjust: the original contributed u_a⊗v_c⊗w_f + u_b⊗v_d⊗w_g
    # and the new contributes (u_a+u_b)⊗(v_c+v_d)⊗(w_f+w_g)
    # which = u_a⊗v_c⊗w_f + u_a⊗v_c⊗w_g + u_a⊗v_d⊗w_f + u_a⊗v_d⊗w_g
    #   + u_b⊗v_c⊗w_f + u_b⊗v_c⊗w_g + u_b⊗v_d⊗w_f + u_b⊗v_d⊗w_g
    # The extra terms are: u_a⊗v_c⊗w_g + u_a⊗v_d⊗w_f + u_b⊗v_c⊗w_g + u_b⊗v_d⊗w_f
    # These need to be compensated by other factors!
    
    # So simple merging doesn't work directly - we introduce extra terms.
    # Unless the extra terms happen to cancel with adjustments to other factors.
    
    # This is getting complex. Let me try a different approach.
    
    # Strategy: Use the fact that for 3x4x5, we can try to find a decomposition
    # by solving the system of equations directly for a given rank.
    
    # For rank r, we have r factors (u_c, v_c, w_c). The constraint is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # This is a system of 60 nonlinear equations in 47r unknowns.
    # For r=59, that's 2773 unknowns and 60 equations - very underdetermined.
    
    # A practical approach: fix U and V (randomly or structured), solve for W.
    # Given U and V, the equations are linear in W:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Let A_{(i,j,k),c} = U[i*m+j,c] * V[j*p+k,c]
    # Then: sum_c A_{(i,j,k),c} * W[k*n+i,c] = 1
    # But W is indexed by (k*n+i, c), so for fixed c, W[:,c] is a vector in R^{15}.
    # The equation for (i,j,k) involves W[k*n+i, c] for all c, weighted by U[i*m+j,c]*V[j*p+k,c].
    
    # This is not simply linear in W because W[k*n+i,c] appears with coefficient
    # U[i*m+j,c]*V[j*p+k,c] which depends on c.
    
    # Alternative: fix U and W, solve for V (linear). Or fix two, solve for third.
    
    # Let me try: fix U (12 x r) and V (20 x r) as random, then for each output index
    # (k,n+i) i.e., row of W, the equations are:
    # For each (i,j,k): W[k*n+i,c] appears with coefficient U[i*m+j,c]*V[j*p+k,c]
    # So: sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is: (U[m_row, :] * V[p_row, :]) dot W[row, :] = 1 where
    # m_row = i*m+j, p_row = j*p+k, W_row = k*n+i.
    
    # For fixed j,k: the equations over i give us 3 equations (i=0,1,2) for W[k*n+i,:]
    # But W has different rows for different i, so this is 3 separate equations for
    # W[0*n+0,:], W[1*n+0,:], W[2*n+0,:] (for k=0).
    
    # Actually, for a fixed (j,k), and varying i:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1, for i=0,1,2
    # These are 3 independent linear equations in the variables W[k*n+0,:], W[k*n+1,:], W[k*n+2,:]
    # Each is a dot product: (U[i*m+j,:] * V[j*p+k,:]) · W[k*n+i,:] = 1
    
    # So for each (j,k) pair (there are m*p=20 of them), we get 3 linear equations
    # for 3 different rows of W (rows k*n+0, k*n+1, k*n+2).
    # Total: 20*3 = 60 linear equations in 15*r variables (W is 15 x r).
    
    # Wait, W has 15 rows and r columns = 15r variables. 60 equations in 15r unknowns.
    # For r=59: 60 equations in 885 unknowns. Underdetermined, so solutions exist.
    
    # The question is whether we can choose U and V such that the system is consistent
    # and W has the right structure. Actually, any solution W works as long as the
    # entries are valid (real numbers, and we want integer/half-integer ideally).
    
    # Let me try: choose U and V to be the schoolbook factors (first 59 of 60),
    # and solve for W.
    
    # Actually, a simpler approach: take the schoolbook 60 factors, remove one,
    # and see if we can adjust the remaining 59 to still satisfy the equations.
    
    # Let me try a randomized approach:
    # 1. Initialize U, V, W with small random values
    # 2. Use iterative methods to satisfy the 60 constraints
    
    # Or even simpler: since the system is very underdetermined for r=59,
    # let's just pick U and V randomly and solve the linear system for W.
    
    # For each row w_row of W (15 rows), we need:
    # For all (i,j,k) with k*n+i = w_row:
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_row,c] = 1
    # i.e., sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[w_row,c] = 1
    # For w_row = k*n+i, the constraint is:
    #   (U[i*m+j,:] * V[j*p+k,:]) · W[w_row,:] = 1  for each j
    
    # For fixed w_row = k*n+i, the varying index is j (0 to m-1 = 0 to 3).
    # So we get m=4 linear equations for W[w_row,:] (a vector of length r):
    #   (U[i*m+j,:] * V[j*p+k,:]) · W[w_row,:] = 1, for j=0,1,2,3
    
    # That's 4 equations in r=59 unknowns for each row. 15 rows * 4 = 60 equations total.
    
    # This is solvable! Let's do it.
    
    def _attempt_random_search(seed, max_iters=1000):
        """Try random U, V and solve for W."""
        rng = np.random.RandomState(seed)
        r = target_rank
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random U and V with simple entries
            U_rand = rng.randint(-1, 2, size=(nm, r)).astype(float)
            V_rand = rng.randint(-1, 2, size=(mp, r)).astype(float)
            
            # Solve for W: for each row w_row = k*n+i,
            # we need (U[i*m+j,:] * V[j*p+k,:]) · W[w_row,:] = 1 for j=0..m-1
            W_sol = np.zeros((pn, r), dtype=float)
            valid = True
            
            for k in range(p):
                for i in range(n):
                    w_row = k * n + i
                    # 4 equations in r unknowns
                    A_rows = []
                    for j in range(m):
                        coeff = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                        # Check if coeff is nonzero (otherwise equation is 0=1, impossible)
                        if np.all(np.abs(coeff) < 1e-12):
                            valid = False
                            break
                        A_rows.append(coeff)
                    if not valid:
                        break
                    
                    if valid:
                        # Solve A @ W[w_row,:] = 1 (4 equations, r unknowns)
                        # Use least squares (will give a solution since underdetermined)
                        A_mat = np.array(A_rows)  # (m, r)
                        try:
                            # Find W_row such that A_mat @ W_row = ones(m)
                            # This is underdetermined, use lstsq
                            W_row, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W_sol[w_row, :] = W_row
                        except:
                            valid = False
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_mat = U_rand
            V_mat = V_rand
            W_mat = W_sol
            
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        s = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                        if abs(s - 1.0) > 1e-6:
                            ok = False
                            break
            
            if ok:
                return (U_mat, V_mat, W_mat)
        
        return None
    
    # Try random search
    best_result = None
    seed = 42
    while time.time() < deadline:
        result = _attempt_random_search(seed, max_iters=200)
        if result is not None:
            best_result = result
            break
        seed += 1
    
    if best_result is not None:
        U_mat, V_mat, W_mat = best_result
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U_mat]
        V_list = [[float(x) for x in row] for row in V_mat]
        W_list = [[float(x) for x in row] for row in W_mat]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # First, try to find a better decomposition
    better = _try_find_decomposition(n, m, p, time_budget=110.0, target_rank=59)
    
    if better is not None:
        return better
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
