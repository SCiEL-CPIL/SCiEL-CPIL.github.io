import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Start from the schoolbook (rank 144) and greedily merge rank-1 terms
    that are "compatible" — i.e., their supports overlap in a way that allows a
    combined rank-2 (or lower) decomposition of their sum. We use a structured
    ansatz: for each pair of schoolbook terms, check if their sum can be written
    as 2 or fewer rank-1 tensors. If so, replace the pair with the compressed form.
    
    Additionally, we try a direct algebraic construction using polynomial
    interpolation over the j-index to reduce rank.
    """
    n, m, p = 4, 4, 9
    deadline = time.time() + 110

    # Build schoolbook decomposition
    R_sb = n * m * p  # 144
    U = np.zeros((n * m, R_sb), dtype=float)
    V = np.zeros((m * p, R_sb), dtype=float)
    W = np.zeros((p * n, R_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    def verify(U, V, W):
        """Full verification: check all consistent and inconsistent indices."""
        # Consistent: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Inconsistent: T[a,b,c] = 0
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k2 = divmod(b, p)
                if j1 != j2:
                    # Check all c
                    for c in range(p * n):
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    # j1 == j2, check i and k consistency
                    for c in range(p * n):
                        k3, i3 = divmod(c, n)
                        if i1 != i3 or k2 != k3:
                            val = np.dot(U[a], V[b] * W[c])
                            if abs(val) > 1e-9:
                                return False
        return True

    # Quick check schoolbook
    if not verify(U, V, W):
        # Should never happen, but safety
        pass

    # Strategy 1: Polynomial interpolation in j to reduce rank
    # Key idea: For fixed (i, k), the tensor entry is 1 for all j in [0,4).
    # The "j-consistency" constraint means: term r contributes to U[i*m+j, r],
    # V[j'*p+k, r], W[k'*n+i', r] only when j=j', i=i', k=k'.
    #
    # Alternative view: Write the multiplication as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-R decomposition means:
    # C[i,k] = sum_{r=1}^{R} (linear_form_r(A)) * (linear_form_r(B))
    # where linear_form_r(A) depends on A entries and linear_form_r(B) on B entries.
    #
    # In tensor notation with the trace convention:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{consistent}
    #
    # Let's try a different parameterization. Consider that for each r, the
    # non-zero pattern of U[:,r], V[:,r], W[:,r] defines a "support".
    # 
    # For the schoolbook, each term r corresponds to a unique (i,j,k) triple.
    # We can try to combine terms that share i or k.
    #
    # Construction attempt: Group by i.
    # For each i in [0,4), we have m*p = 36 terms (all j, k for that i).
    # Within a fixed i, the constraint is:
    # sum_{j,k} [term for (i,j,k)] must give C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # This is just the rank of the 4x9 matrix multiplication restricted to one row
    # of A and one "column-block" of B. The rank of multiplying a 1x4 row by a 4x9
    # matrix is at most 4 (since it's just a dot product for each output).
    # Actually, C[i,k] = sum_j A[i,j] B[j,k]. For fixed i, this is 4 separate
    # products A[i,j]*B[j,k] summed over j. The bilinear rank is m=4 (each j
    # gives one product). So for each i, we need rank at least 4, and we can do
    # exactly 4 by letting each term correspond to one j.
    #
    # So: R = n * m = 4 * 4 = 16? Let's check.
    # r indexes (i, j) with r = i*m + j
    # U[i'*m+j', r] = 1 if i'=i and j'=j, else 0  -> U is identity-like
    # V[j'*p+k', r] = 1 if j'=j (for any k'), else 0
    # W[k'*n+i', r] = 1 if i'=i (for any k'), else 0
    #
    # Check consistent: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_{(i',j')} [i'=i][j'=j] * [j'=j] * [i'=i]
    # = 1 * 1 * 1 = 1. Correct!
    #
    # Check inconsistent: say a=i1*m+j1, b=j2*p+k2, c=k3*n+i3 with j1!=j2.
    # sum_r U[a,r]*V[b,r]*W[c,r]
    # = sum_{(i',j')} [i1=i'][j1=j'] * [j2=j'] * [i3=i']
    # = [i1=i3] * [j1=j'] * [j2=j'] summed over j'
    # = [i1=i3] * sum_{j'} [j1=j'][j2=j']
    # = [i1=i3] * [j1==j2]
    # If j1!=j2, this is 0. Good.
    #
    # But what if j1==j2 but i1!=i3?
    # sum = [i1=i3] * [j1==j2] = 0 if i1!=i3. Good.
    #
    # What if i1==i3, j1==j2, but k2!=k3?
    # The sum doesn't depend on k2 or k3! It would be 1, but T should be 0.
    # PROBLEM: k is not checked.
    
    # So R=16 doesn't work because k consistency isn't enforced.
    
    # We need to also enforce k consistency. Let's try R = n*m*p but with a
    # different structure. Or use polynomial in k.
    
    # Actually, let me reconsider. The issue is that for fixed (i,j), the term
    # must "know" about k. In the schoolbook, each (i,j,k) is separate.
    
    # Let's try: R = m * p = 36, r = j*p + k
    # U[i*m+j', r] = 1 if j'=j (regardless of i)
    # V[j'*p+k', r] = 1 if j'=j and k'=k
    # W[k'*n+i', r] = 1 if k'=k (regardless of i')
    #
    # Consistent: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_{(j',k')} [j=j'] * [j=j',k=k'] * [k=k']
    # = 1. Correct!
    #
    # Inconsistent with j1!=j2:
    # sum_{(j',k')} [j1=j'] * [j2=j'] * [k2=k'] 
    # = sum_{j',k'} [j1=j'][j2=j'][k2=k']
    # = [j1==j2] * sum_{k'}[k2=k'] = [j1==j2]. If j1!=j2, 0. Good.
    #
    # Inconsistent with i1!=i3 (but j1==j2, k2==k3):
    # sum = [j1=j1] * [j2=j1,k2=k2] * [k2=k2] = 1. But T=0. PROBLEM.
    
    # Same issue: i not enforced.
    
    # The fundamental problem: we need to enforce all three: i consistency,
    # j consistency, k consistency. The schoolbook does this with one term per
    # (i,j,k). Can we do better?
    
    # Let's try R = n * p = 36, r = i*p + k
    # U[i'*m+j, r] = 1 if i'=i (regardless of j)
    # V[j'*p+k', r] = 1 if k'=k (regardless of j')
    # W[k'*n+i', r] = 1 if k'=k and i'=i
    #
    # Consistent: sum_{(i',k')} [i=i'] * [k=k'] * [k=k',i=i'] = 1. Good.
    # Inconsistent j1!=j2: sum doesn't depend on j. = [i1=i3][k2=k3]. If j1!=j2
    # but i1=i3, k2=k3, this gives 1. PROBLEM.
    
    # It seems hard to enforce all three with fewer than n*m*p terms using
    # simple product structures. Let me try a more sophisticated approach.
    
    # Use the fact that m=4 is small. For each (i,k) pair, we need:
    # sum_j A[i,j]*B[j,k] = C[i,k]
    # This is a dot product of length 4. The bilinear rank of a dot product of
    # length m is m (you need m multiplications). So for each (i,k), we need
    # m "multiplication slots". There are n*p = 36 such pairs, so naively
    # 36*4 = 144. But can we share multiplication slots?
    
    # A multiplication slot r corresponds to a product (linear in A)(linear in B).
    # For slot r to contribute to C[i,k], we need:
    # U[i*m+j, r] and V[j*p+k, r] to be nonzero for the j's involved.
    
    # Let's try a structured search: use small integers and try to find a
    # solution with R < 144 by solving a system of equations.
    
    # For a given R, we have R variables per row of U, V, W. The constraints are:
    # For each consistent (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # For each inconsistent (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = 0
    
    # This is a system of trilinear equations in 3*R*n*m + R*m*p + R*p*n variables.
    # For R=144, that's 144*16 + 144*36 + 144*36 = 2304 + 5184 + 5184 = 12672 vars.
    # Too many for direct solve.
    
    # Alternative: fix U and V, solve for W linearly? No, it's trilinear.
    
    # Let me try a different strategy: exploit symmetry and use known results.
    # The rank of matrix multiplication (4,4,9) is at most the rank of (4,4,4) 
    # plus rank of (4,4,5) or similar decompositions.
    
    # Actually, let's just try to find any improvement by random search with
    # structured initializations, focusing on the j-consistency constraint.
    
    # New idea: For each j in [0,4), handle the j-block separately.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # If we could compute each A[i,j]*B[j,k] with shared structure...
    
    # Let's try: R = n * p + (m-1)*n*p = p*(n + (m-1)*n) = p*n*m = 144. Same.
    
    # (Removed: replaced by the directed support-pattern + linear solve strategy above)
    pass

    # Directed strategy: fix a sparse support pattern for the factor matrices
    # (a structured ansatz), then solve for the exact coefficients by enforcing
    # the tensor identity as a linear system. We iterate over candidate
    # support patterns of decreasing size and try to find one whose
    # coefficient system is consistent.
    #
    # Key observation: For a FIXED support pattern (which entries of U, V, W
    # are non-zero), the tensor identity
    #   sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # becomes a LINEAR system in the unknown coefficients, because for each
    # (a,b,c) only the terms r where ALL THREE of U[a,r], V[b,r], W[c,r]
    # are non-zero contribute, and each such term is a product of three
    # unknowns. Wait - that's still trilinear.
    #
    # However, if we fix TWO of the three factor matrices (say U and V), then
    # the equations become LINEAR in W. So the strategy is:
    #   1. Choose a support pattern and initial values for U and V.
    #   2. Solve the linear system for W.
    #   3. Verify the full tensor identity.
    #   4. If it fails, perturb U/V or try a different pattern.
    #
    # A cleaner approach: use the "rank-1 peeling" / iterative linear solve.
    # Fix the support (which (row, col) pairs are non-zero in each matrix).
    # For a given support, the problem is to find values. We can do this by
    # fixing U and V and solving for W linearly, then checking consistency.
    #
    # Even better: for a support where each column r of U has support S_U(r),
    # each column r of V has support S_V(r), each column r of W has support
    # S_W(r), the equation for (a,b,c) involves only r where a in S_U(r),
    # b in S_V(r), c in S_W(r). If we fix U and V, then for each (a,b,c):
    #   sum_{r: a in S_U(r), b in S_V(r), c in S_W(r)} U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is linear in the W[c,r] variables. We can set up this system and
    # solve it.
    #
    # Let's implement this with a structured support pattern.

    def try_solve_with_support(support_U, support_V, support_W, R, U_init=None, V_init=None):
        """
        Given support patterns (sets of non-zero positions) and initial U, V,
        solve for W linearly and verify.
        
        support_U: list of sets, support_U[r] = set of rows a where U[a,r] != 0
        support_V: list of sets, support_V[r] = set of rows b where V[b,r] != 0
        support_W: list of sets, support_W[r] = set of rows c where W[c,r] != 0
        """
        # Build the linear system for W
        # Variables: W[c, r] for c in support_W[r], r in range(R)
        # Equation for each (a, b, c):
        #   sum_{r: a in support_U[r], b in support_V[r], c in support_W[r]} 
        #       U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        # Map (c, r) -> variable index
        var_map = {}
        var_count = 0
        for r in range(R):
            for c in support_W[r]:
                var_map[(c, r)] = var_count
                var_count += 1
        
        if var_count == 0:
            return None, None, None
        
        # Build equations
        # For each (a, b, c) in the full tensor space
        n, m, p = 4, 4, 9
        eqs = []  # list of (coeff_dict, rhs)
        
        # Consistent indices: T = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    coeff = {}
                    for r in range(R):
                        if a in support_U[r] and b in support_V[r] and c in support_W[r]:
                            idx = var_map[(c, r)]
                            coeff[idx] = coeff.get(idx, 0) + U_init[a, r] * V_init[b, r]
                    if coeff:
                        eqs.append((coeff, 1.0))
        
        # Inconsistent indices: T = 0 (sample a subset for speed)
        # We need all of them for correctness, but let's include them all
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k2 = divmod(b, p)
                for c in range(p * n):
                    k3, i3 = divmod(c, n)
                    if j1 == j2 and i1 == i3 and k2 == k3:
                        continue  # consistent, already handled
                    coeff = {}
                    for r in range(R):
                        if a in support_U[r] and b in support_V[r] and c in support_W[r]:
                            idx = var_map[(c, r)]
                            coeff[idx] = coeff.get(idx, 0) + U_init[a, r] * V_init[b, r]
                    if coeff:
                        eqs.append((coeff, 0.0))
        
        # Build matrix and solve
        A_mat = np.zeros((len(eqs), var_count))
        b_vec = np.zeros(len(eqs))
        for eq_idx, (coeff, rhs) in enumerate(eqs):
            for var_idx, val in coeff.items():
                A_mat[eq_idx, var_idx] = val
            b_vec[eq_idx] = rhs
        
        try:
            W_vals, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return None, None, None
        
        # Check residual
        if len(eqs) > 0 and np.linalg.norm(A_mat @ W_vals - b_vec) > 1e-8:
            return None, None, None
        
        # Build W
        W_sol = np.zeros((p * n, R))
        for (c, r), idx in var_map.items():
            W_sol[c, r] = W_vals[idx]
        
        return U_init, V_init, W_sol

    # Now try structured support patterns
    # Pattern: for each r, U has support on a specific (i,j) pair, V on (j,k) pair, W on (k,i) pair
    # This is the schoolbook. To get lower rank, we need overlapping supports.
    
    # Try: R = 64, where r = (i, j, k) but with some merging
    # Actually, let's try the approach of fixing U and V to be "nice" and solving for W
    
    # Strategy: Use U and V from a reduced-rank ansatz, solve for W
    
    # Try R values from 72 down to 48
    np.random.seed(123)
    best_result = None
    
    for R in [72, 64, 56, 48]:
        if time.time() > deadline - 30:
            break
        
        # Try several random initializations of U and V with sparse structure
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            
            # Create sparse U and V: each column has 1-3 non-zeros
            U_t = np.zeros((16, R))
            V_t = np.zeros((36, R))
            
            for r in range(R):
                # U column: pick 1-2 random rows
                n_nonzero = 1 + (r % 2)
                rows = np.random.choice(16, size=n_nonzero, replace=False)
                for row in rows:
                    U_t[row, r] = 1 + (r % 3)  # values 1, 2, or 3
                # V column: pick 1-3 random rows
                n_nonzero = 1 + (r % 3)
                rows = np.random.choice(36, size=n_nonzero, replace=False)
                for row in rows:
                    V_t[row, r] = 1 + ((r + 1) % 3)
            
            # Support patterns
            support_U = [set(np.nonzero(U_t[:, r])[0]) for r in range(R)]
            support_V = [set(np.nonzero(V_t[:, r])[0]) for r in range(R)]
            # W support: all rows (dense) to give the linear system flexibility
            support_W = [set(range(36)) for r in range(R)]
            
            U_sol, V_sol, W_sol = try_solve_with_support(support_U, support_V, support_W, R, U_t, V_t)
            
            if W_sol is not None:
                if verify(U_sol, V_sol, W_sol):
                    best_result = (U_sol, V_sol, W_sol, R)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_sol, V_sol, W_sol, R_sol = best_result
        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # If we get here, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
