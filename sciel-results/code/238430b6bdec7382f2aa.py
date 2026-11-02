"""
Matrix multiplication 4x5 by 5x6: exact rank-minimal tensor decomposition.

Strategy:
  1. Build the full multiplication tensor T[i,j,k] (n*m*p = 120 terms).
  2. Use the rank-revealing linear algebra approach:
     - The tensor T can be seen as a linear map from A⊗B to C.
     - The schoolbook decomposition has rank 120.
     - We attempt to reduce rank by finding linear dependencies among the
       120 basis tensors via exact rational arithmetic (sympy-style with
       Python's fractions).
     - Specifically, we form the 120 x 120 Gram-like matrix of inner products
       between the 120 rank-1 tensors, find its null space over Q, and use
       that to identify which combinations of rank-1 terms can be replaced
       by fewer terms.
  3. In practice, for small matrices like 4x5x6, the known minimal rank is
     not much below 120. The literature (Strassen, etc.) gives bounds.
     For 4x5x6, the trivial upper bound is 120. Known results suggest
     the rank is at least n*m + m*p + p*n - 2*min(n,m,p) or similar,
     but exact minimal rank for 4x5x6 is not widely tabulated.
  
  Given the 120s time limit, we do:
  - Start with the schoolbook rank-120 decomposition.
  - Attempt a systematic search for rank reductions by checking if
    subsets of the 120 rank-1 tensors are linearly dependent in the
    tensor space (dimension 4*5*6=120). Since the tensor space has
    dimension exactly 120, the 120 schoolbook tensors form a basis,
    so no linear dependency exists in the full tensor space.
    
    However, the matrix multiplication tensor lives in a subspace.
    The key insight: the multiplication tensor T ∈ V*⊗W*⊗Z (dual spaces),
    and its rank as a 3-tensor is what we minimize. The 120 schoolbook
    terms are linearly independent as vectors in the 120-dim space,
    but the tensor rank (as a 3-way tensor) can be less than 120
    if there's a different decomposition.
    
  For 4x5x6, let's try a direct approach:
  - Use the fact that rank(A⊗B→C) ≥ max(n*p, m*p, n*m) = max(24,30,20) = 30.
  - Upper bound: 120 (schoolbook).
  - Try to find a decomposition with rank < 120 using randomized
    methods or structured search.
  
  Given time constraints, we'll implement:
  1. The schoolbook rank-120 as fallback.
  2. A structured attempt using block-wise optimizations.
  3. Return the best found within the time limit.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=Fraction(0)):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
    The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 if i,j,k are valid indices (which they always are in our setup),
    and 0 otherwise (but all indices are valid by construction).
    
    Actually, the multiplication tensor is defined as:
    T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
    In the standard convention, T[i,j,k] = 1 for all valid i,j,k (since each triple
    corresponds to one multiplication-addition term).
    
    Wait - the tensor for matrix multiplication C = A*B where A is n×m, B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor T ∈ R^{n×m×p} has T[i,j,k] = 1 for all i,j,k.
    The rank of this tensor is what we're minimizing.
    
    A rank-1 decomposition term is u_i * v_j * w_k where:
    - u is a vector in R^n (indexed by i)
    - v is a vector in R^m (indexed by j)  
    - w is a vector in R^p (indexed by k)
    
    But the interface uses:
    - U: shape (n*m, rank) - U[i*m+j, c]
    - V: shape (m*p, rank) - V[j*p+k, c]
    - W: shape (p*n, rank) - W[k*n+i, c]
    
    And the reconstruction is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    For the multiplication tensor, T[i,j,k] = 1 for all valid i,j,k.
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    # Check a subset of entries for speed, or all if small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(len(U[0])):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != Fraction(1):
                    return False
    return True


def _try_rank_reduction(n, m, p, max_rank_to_try, deadline):
    """
    Attempt to find a decomposition with rank < n*m*p using structured methods.
    
    For small problems, we can try:
    1. Randomized rank-1 peeling
    2. Exploiting symmetries
    3. Known algorithmic constructions
    
    Given the time limit, we'll try a few structured approaches.
    """
    base_rank = n * m * p
    U0, V0, W0 = _schoolbook_decomposition(n, m, p)
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = base_rank
    
    # For 4x5x6, let's try a few specific constructions.
    # One approach: if we can find a way to combine terms.
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k in [n]×[m]×[p].
    # This is the all-ones tensor of size n×m×p.
    # The rank of the all-ones tensor of size a×b×c over R is:
    #   rank = 1 if any of a,b,c is 1
    #   Otherwise, it's more complex.
    # 
    # Wait - is the matrix multiplication tensor the all-ones tensor?
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T satisfies: <T, A⊗B> = C for all A,B
    # T[i,j,k] = 1 for all i,j,k.
    # Yes! The multiplication tensor IS the all-ones tensor.
    #
    # The rank of the all-ones tensor J_{n,m,p} (all entries 1) over R:
    # This is a well-studied object. The rank of J_{n,m,p} is:
    # - If n=m=p=1: rank 1
    # - In general, the rank of the all-ones tensor of format n×m×p
    #   over a field of characteristic 0 is:
    #   rank(J_{n,m,p}) = min(n,m,p) if the field has enough elements?
    #   No, that's not right either.
    #
    # Actually, the all-ones tensor J_{n,m,p} can be written as:
    # J_{n,m,p} = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^n, v_c ∈ R^m, w_c ∈ R^p
    #
    # One construction: J_{n,m,p} = (1,1,...,1)⊗(1,1,...,1)⊗(1,1,...,1)
    # This is rank 1! The all-ones tensor IS rank 1!
    #
    # Wait, that can't be right for matrix multiplication. Let me reconsider.
    #
    # The matrix multiplication tensor is NOT simply the all-ones tensor in the
    # naive sense. Let me re-examine.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want T such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    #
    # And the tensor decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] * v_c[j] * w_c[k]
    #
    # So we need: sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.
    #
    # With u_c = (1,1,...,1)^T, v_c = (1,1,...,1)^T, w_c = (1,1,...,1)^T,
    # we get sum_c 1*1*1 = r (the number of terms).
    # So if r=1, we get 1. If r>1, we get r.
    #
    # So the all-ones tensor J_{n,m,p} has rank 1 over R!
    # u = (1,...,1), v = (1,...,1), w = (1,...,1) gives exactly 1.
    #
    # But wait - that would mean matrix multiplication has rank 1, which is absurd.
    # Let me re-examine the interface.
    #
    # The interface says:
    # U: shape (n*m, rank)
    # V: shape (m*p, rank)  
    # W: shape (p*n, rank)
    # Reconstruction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # This is NOT the standard tensor decomposition! In standard tensor decomposition:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # 
    # Here the indices are mixed:
    # U depends on (i,j), V depends on (j,k), W depends on (k,i)
    #
    # So the reconstruction is:
    # sum_c U[i,j]_c * V[j,k]_c * W[k,i]_c
    #
    # This is a different structure! The three factors don't have independent
    # indices. U couples i and j, V couples j and k, W couples k and i.
    #
    # This is actually the standard way to represent matrix multiplication
    # in the "trace convention" mentioned in the task.
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # The schoolbook solution uses U[i*m+j, c] = 1 only when c corresponds to (i,j,k),
    # and similarly for V and W. This gives rank n*m*p.
    #
    # Can we do better? We need to find matrices U, V, W such that
    # for every (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    #
    # This is a system of n*m*p equations (one for each (i,j,k)) in
    # rank * (n*m + m*p + p*n) unknowns.
    #
    # For n=4, m=5, p=6: 120 equations, rank * (20 + 30 + 24) = 74*rank unknowns.
    #
    # The question is: what's the minimum rank such that this system has a solution?
    #
    # Let's think about it differently. Define for each c:
    # A_c = U[:,c] reshaped to n×m
    # B_c = V[:,c] reshaped to m×p
    # C_c = W[:,c] reshaped to p×n
    #
    # Then the condition is:
    # For all i,j,k: sum_c A_c[i,j] * B_c[j,k] * C_c[k,i] = 1
    #
    # This is equivalent to:
    # sum_c trace(A_c ⊗ B_c ⊗ C_c) ... no, let me think again.
    #
    # Actually, define the "cyclic" product:
    # For fixed c, define f_c(i,j,k) = A_c[i,j] * B_c[j,k] * C_c[k,i]
    # We need sum_c f_c(i,j,k) = 1 for all i,j,k.
    #
    # Note that f_c is a 3D array. The set of all such f_c (as c varies over
    # all possible A_c, B_c, C_c) spans some subspace of R^{n×m×p}.
    #
    # The key question: what is the minimum number of such "cyclic" rank-1
    # tensors needed to sum to the all-ones tensor?
    #
    # This is the "cyclic rank" or "multiplication rank" in this specific
    # parametrization.
    #
    # For the schoolbook: A_c = E_{i,j}, B_c = E_{j,k}, C_c = E_{k,i} (standard basis)
    # gives rank n*m*p.
    #
    # Can we do better? Let's try small cases mentally.
    # For n=m=p=1: rank 1 (trivial).
    # For n=m=p=2: schoolbook gives 8. Can we do better?
    #
    # Let's try to set up the problem for n=m=p=2:
    # 8 equations, 6*rank unknowns.
    # For rank 1: 6 unknowns, 8 equations - likely no solution.
    # For rank 2: 12 unknowns, 8 equations - might have solution.
    #
    # Let me just implement a solver for the general case using linear algebra.
    # The equations are cubic (products of three variables), so it's nonlinear.
    # But we can use the following approach:
    #
    # Fix U and V, then the equations become linear in W:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is linear in the W variables! For fixed U,V, we have n*m*p equations
    # in rank*p*n unknowns (the entries of W).
    #
    # So the approach is:
    # 1. Randomly initialize U and V.
    # 2. Solve for W via least squares (exact rational or floating point).
    # 3. Check if the solution is exact.
    # 4. Iterate or restart with different U,V.
    #
    # But this is a non-convex problem. Let me try a different approach.
    #
    # Alternative: use the fact that this is equivalent to finding a
    # decomposition of the multiplication tensor in a specific format.
    #
    # Given the complexity, let me just try a few specific constructions
    # and a random search for small rank values.
    
    # For now, return the schoolbook decomposition as the best we have.
    # The all-ones tensor insight doesn't apply because of the cyclic index structure.
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for 120s limit
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = base_rank
    
    # Try to find a better decomposition
    # For 4x5x6, let's attempt a structured search
    
    # Approach: Try to find rank-r decomposition by solving the nonlinear system.
    # We'll use a simple iterative approach:
    # 1. Fix U, V randomly
    # 2. Solve for W linearly
    # 3. Check residual
    # 4. If not exact, try different U, V
    
    # Actually, let me try a more principled approach.
    # The system sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    # can be attacked by fixing two matrices and solving for the third.
    
    # Let's try rank values from low to high
    # But first, let's see if there's a known result for 4x5x6.
    
    # For now, just return schoolbook if we can't find better quickly.
    # Let me try a quick random search for rank < 120.
    
    import numpy as np
    
    # Try a few random initializations for various ranks
    # This is a long shot but worth trying within the time budget
    
    for target_rank in [30, 40, 50, 60, 70, 80, 90, 100, 110]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            
            # Solve for W: for each (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
            # This is linear in W.
            # Build the system: A_w * W_flat = b
            # where A_w has one row per (i,j,k) equation
            
            num_eqs = n * m * p
            num_w_unknowns = p * n * target_rank
            
            # For each equation (i,j,k) and each W-unknown (k',i',c):
            # The coefficient is nonzero only if k'=k and i'=i
            # A_w[(i,j,k), (k*n+i)*target_rank + c] = U[i*m+j,c] * V[j*p+k,c]
            
            # This is a sparse system. Let's build it.
            # Actually, for each (i,j,k), the equation involves only W[k*n+i, c] for all c.
            # So we can solve each (i,k) group independently? No, because the
            # equations for different j with same (i,k) share the same W variables.
            
            # Let's build the full system.
            # Row index: (i,j,k) -> i*m*p + j*p + k
            # Col index: (k,i,c) -> (k*n+i)*target_rank + c
            
            A_w = np.zeros((num_eqs, num_w_unknowns))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            w_idx = (k * n + i) * target_rank + c
                            A_w[eq_idx, w_idx] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
            
            # Solve least squares
            try:
                W_flat, residuals, rank_w, sv = np.linalg.lstsq(A_w, b, rcond=None)
                # Check if solution is exact
                W_sol = W_flat.reshape(p * n, target_rank)
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_sol[k*n+i, c] 
                                   for c in range(target_rank))
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    # Convert to exact fractions
                    U_exact = [[Fraction(round(U_rand[i*m+j, c], 12)).limit_denominator(10**6) 
                               for c in range(target_rank)] for i in range(n) for j in range(m)]
                    V_exact = [[Fraction(round(V_rand[j*p+k, c], 12)).limit_denominator(10**6) 
                               for c in range(target_rank)] for j in range(m) for k in range(p)]
                    W_exact = [[Fraction(round(W_sol[k*n+i, c], 12)).limit_denominator(10**6) 
                               for c in range(target_rank)] for k in range(p) for i in range(n)]
                    
                    # Verify exactness
                    if _verify_decomposition(U_exact, V_exact, W_exact, n, m, p):
                        best_U, best_V, best_W = U_exact, V_exact, W_exact
                        best_rank = target_rank
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    if time.time() > deadline:
        pass  # Return whatever we have
    
    # Return the best decomposition found
    if best_rank < base_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
