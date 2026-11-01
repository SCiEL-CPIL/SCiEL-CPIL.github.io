"""Rewritten program: systematic search for rank-minimal 2x3*3x4 matrix mult tensor."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Build the multiplication tensor T[a, b, c] where
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (i in 0..n-1, k in 0..p-1)
    # T[a,b,c] = 1 iff a,b,c form a valid (i,j,k) triple, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    def validate_fast(U, V, W):
        """Fast vectorized validation. U: (n*m, r), V: (m*p, r), W: (p*n, r)"""
        T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(T_hat, T)

    # Schoolbook solution: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=int)
    V_sb = np.zeros((m*p, rank_sb), dtype=int)
    W_sb = np.zeros((p*n, rank_sb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try structured patterns for various ranks
    # Pattern 1: rank = m*p = 12 (failed as shown - doesn't enforce i_a==i_c)
    # Pattern 2: rank = m*n*p / something... let's try other patterns
    
    # Try rank = n*m*p = 24 (schoolbook, always works)
    # Try rank = m * (n*p) = 24 (same as schoolbook)
    
    # Let's try a smarter pattern: for each (j, i), one component covering all k
    # rank = m*n = 6
    # Component (j,i): U[i*m+j]=1, V[j*p+k] for all k, W[k*n+i] for all k
    # This won't work because V and W would need to be correlated per k
    
    # Try rank = p*m = 12 with different structure
    # For each (j,k): U[i*m+j]=1 for all i, V[j*p+k]=1, W[k*n+i]=1 for all i
    # Already tried, fails
    
    # Let's try to find the actual minimal rank using a search approach
    # We'll try ranks from 12 to 23 and use a linear algebra / enumeration approach
    
    # Key insight: the tensor has 24 non-zero entries.
    # For a rank-r decomposition, each of the r components contributes to some subset
    # of the 24 entries (where all three factors are non-zero).
    
    # Let's try a direct approach: enumerate possible support patterns for small ranks
    # and solve for the values using linear algebra.
    
    # For rank r, we have r components. Each component has:
    # U_r: 6 entries, V_r: 12 entries, W_r: 8 entries
    # Total unknowns: r * 26
    # Constraints: 576 equations (but many are 0=0)
    
    # The non-trivial constraints are:
    # For each of the 24 (i,j,k) triples: sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i] = 1
    # For each invalid (a,b,c): sum_r U_r[a]*V_r[b]*W_r[c] = 0
    
    # This is a nonlinear system (products of variables from different components).
    # But if we fix the support patterns, it becomes linear in the "product" variables.
    
    # Alternative: use the fact that we can scale components.
    # If we fix U and V, then W is determined by linear equations.
    
    # Let's try a more practical approach: use random initialization and optimization
    # to minimize the error, checking for exact solutions.
    
    # Actually, for this small problem, let's try all "natural" structured patterns
    # and see which ranks work.
    
    # Pattern: for each j, use n*p = 8 components (schoolbook per j)
    # Total: m * n * p = 24 (schoolbook)
    
    # Can we share components between j values?
    # The i and k indices are the same across j, so maybe.
    
    # Let's try: rank = n*p = 8 (one set of components shared across all j)
    # Component (i,k): U[i*m+j] for all j, V[j*p+k] for all j, W[k*n+i] = 1
    # This would require U and V to encode j-dependence, which they can't with fixed support.
    
    # I think the minimal rank is likely higher than 12. Let's try to compute it
    # using a more systematic approach.
    
    # For now, let's try a few more structured patterns and then fall back to 24.
    
    # Pattern: rank = m * max(n,p) = 3*4 = 12, but with per-i structure
    # For each (j, i): one component
    # U[i*m+j] = 1, V[j*p+k] = 1 for all k, W[k*n+i] = 1 for all k
    # Check: T_hat[a,b,c] where a=i_a*m+j_a, b=j_b*p+k_b, c=k_c*n+i_c
    # = sum_{j,i} [j_a==j][i_a==i] * [j_b==j] * [k_b==k] * [k_c==k][i_c==i]
    # = [j_a==j_b] * [i_a==i_c] * [k_b==k_c] * (number of j,i satisfying the conditions)
    # For j_a==j_b=j and i_a==i_c=i, the sum has exactly one term (j,i), so = 1.
    # Wait, that's exactly T! Let me verify.
    
    # Component (j,i): U[i*m+j]=1, V[j*p+k]=1 for all k in 0..p-1, W[k*n+i]=1 for all k in 0..p-1
    # For (a,b,c) with a=i_a*m+j_a, b=j_b*p+k_b, c=k_c*n+i_c:
    # Component (j,i) contributes: [j_a==j][i_a==i] * [j_b==j] * [k_b==k] * [k_c==k][i_c==i]
    # where k is summed... no, V[j*p+k]=1 for ALL k, so V[b]=1 iff j_b==j (regardless of k_b)
    # Similarly W[c]=1 iff i_c==i (regardless of k_c)
    # So contribution = [j_a==j][i_a==i] * [j_b==j] * [i_c==i]
    # = [j_a==j_b==j] * [i_a==i_c==i]
    # Sum over all (j,i): this equals [j_a==j_b] * [i_a==i_c]
    # But T also requires k_b==k_c! So this pattern gives T_hat = [j_a==j_b]*[i_a==i_c], 
    # which is NOT equal to T (which also requires k_b==k_c).
    
    # So rank = m*n = 6 with this pattern doesn't work either.
    
    # The issue is that k_b and k_c need to be matched, which requires k-dependent structure.
    
    # Let's try rank = m * p = 12 with the correct per-(i,k) structure:
    # For each (j, k): U[i*m+j]=1 for all i, V[j*p+k]=1, W[k*n+i]=1 for all i
    # Already tried, gives [j_a==j_b]*[k_b==k_c] but not [i_a==i_c]
    
    # The fundamental issue: we need to match i_a==i_c AND k_b==k_c AND j_a==j_b
    # These are three independent conditions that need to be enforced.
    
    # I believe the minimal rank for 2x3*3x4 is actually 24 (schoolbook is optimal)
    # or possibly slightly less. Let me try a computational search for ranks 16-23.
    
    # For a more efficient search, let's use the following approach:
    # Fix U and V, then solve for W using linear equations.
    # For each c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b)
    # This is a linear system in W[c, :] for each c.
    
    # Let's try random U, V and check if W can be solved exactly.
    
    import random
    random.seed(42)
    
    def try_random_rank(r, n_trials=5000):
        """Try random U, V and solve for W."""
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            # Random U, V with small integer entries
            U_rand = np.random.randint(-2, 3, size=(n*m, r))
            V_rand = np.random.randint(-2, 3, size=(m*p, r))
            
            # For each c, solve: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b)
            # This is: (U * diag(V[b,:]) * W[c,:].T) = T[a,b,c] for all a,b
            # More precisely: for fixed c, we need W[c,:] such that
            # sum_r W[c,r] * (U[:,r] * V[:,r]) = T[:, :, c] (reshaped)
            # Let M_c[a,b] = T[a,b,c], and let P_r[a,b] = U[a,r]*V[b,r]
            # Then sum_r W[c,r] * P_r[a,b] = M_c[a,b]
            # This is a linear system: P * W[c,:] = M_c where P is (n*m*m*p, r)
            
            # Build P matrix: P[ab, r] = U[a,r] * V[b,r]
            P = np.zeros((n*m * m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    P[a*m*p + b, :] = U_rand[a, :] * V_rand[b, :]
            
            # For each c, solve P * W_c = T[:, :, c].flatten()
            W_sol = np.zeros((p*n, r))
            valid = True
            for c in range(p*n):
                rhs = T[:, :, c].flatten()
                # Check if rhs is in the column space of P
                # Use least squares and check residual
                try:
                    sol, residuals, rank_p, sv = np.linalg.lstsq(P, rhs, rcond=None)
                    residual = np.linalg.norm(P @ sol - rhs)
                    if residual > 1e-10:
                        valid = False
                        break
                    W_sol[c, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if W_sol has "nice" values (integers or half-integers)
            # and validate the full decomposition
            if validate_fast(U_rand, V_rand, W_sol):
                return U_rand, V_rand, W_sol
        
        return None
    
    # Try ranks from 12 to 23
    for r in range(12, 24):
        if time.time() > deadline:
            break
        result = try_random_rank(r, n_trials=2000)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_U, best_V, best_W = U_found, V_found, W_found
            break
    
    # Convert to lists for the return format
    result = {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }

    return result
