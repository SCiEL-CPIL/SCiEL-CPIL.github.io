"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def _build_tensor(n, m, p):
    """Build the (n*m) x (m*p) x (p*n) tensor T where T[a,b,c]=1 iff consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return bool(np.allclose(result, T, atol=tol))

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition (always valid)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = n*m*p
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank17(n, m, p):
    """Verified Strassen(2x2x2) + direct decomposition of rank 17."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = 17
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))

    def set_term(r, u, v, w):
        for a, val in u.items(): U[a, r] = val
        for b, val in v.items(): V[b, r] = val
        for c, val in w.items(): W[c, r] = val

    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2,0:2]
    set_term(0, {0:1,4:1}, {0:1,4:1}, {0:1,3:1})   # M1=(a0+a4)(b0+b4)
    set_term(1, {3:1,4:1}, {0:1},       {1:1,3:-1}) # M2=(a3+a4)b0
    set_term(2, {0:1},       {1:1,4:-1}, {2:1,3:1}) # M3=a0(b1-b4)
    set_term(3, {4:1},       {3:1,0:-1}, {0:1,1:1}) # M4=a4(b3-b0)
    set_term(4, {0:1,1:1},   {4:1},      {0:-1,2:1})# M5=(a0+a1)b4
    set_term(5, {3:1,0:-1},  {0:1,1:1},  {3:1})     # M6=(a3-a0)(b0+b1)
    set_term(6, {1:1,4:-1},  {3:1,4:1},  {0:1})     # M7=(a1-a4)(b3+b4)
    # Part 2: A1(2x2) x B1_right(2x1) -> C[:,2]
    set_term(7,  {0:1}, {2:1}, {4:1})
    set_term(8,  {1:1}, {5:1}, {4:1})
    set_term(9,  {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    # Part 3: A2(2x1) x B2(1x3) -> C[0:2,0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    return U, V, W

def _try_rank16_structured(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition by fixing a structured sparsity pattern
    and solving the resulting linear system.
    
    Strategy: The tensor has 18 nonzero entries. Each rank-1 term with support
    size s_u * s_v * s_w contributes to at most that many entries. If we use
    terms with small supports (2-3 nonzeros per factor), we can cover the 18
    entries with 16 terms if some terms share contributions.
    
    We try several structured patterns and solve for the values.
    """
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    # Get the nonzero entries of T
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nonzero.append((a, b, c))
    
    # For rank 16, we need 16 rank-1 terms.
    # Each term r has U[:,r], V[:,r], W[:,r].
    # The contribution to T[a,b,c] from term r is U[a,r]*V[b,r]*W[c,r].
    # 
    # Key insight: If we fix which (a,b,c) each term contributes to, and fix
    # the supports of U[:,r], V[:,r], W[:,r], then the system becomes linear
    # in the products. But it's still trilinear.
    #
    # Alternative: Fix U and V, then W is determined linearly (if the system
    # is consistent). So we can:
    # 1. Fix a sparsity pattern for U and V (which entries are nonzero)
    # 2. Solve for the values of U and V
    # 3. Check if W can be solved linearly
    #
    # This is still hard. Let's try a more direct approach:
    # Assume each term has exactly 2 nonzeros in U, 2 in V, 2 in W.
    # Then each term contributes to at most 8 entries of T.
    # With 16 terms, we have 128 "slots" for the 18 entries.
    # 
    # The key constraint is that for each nonzero entry T[a,b,c]=1,
    # the sum over r of U[a,r]*V[b,r]*W[c,r] must equal 1.
    # And for each zero entry, the sum must equal 0.
    #
    # This is a system of 324 equations (6*9*6) in 16*6 + 16*9 + 16*6 = 360 unknowns.
    # But it's nonlinear.
    #
    # Let's try a different approach: use the fact that the tensor is sparse.
    # Only 18 entries are nonzero. For the 306 zero entries, we have:
    # sum_r U[a,r]*V[b,r]*W[c,r] = 0
    #
    # If we fix U and V, this becomes linear in W. So:
    # 1. Randomly initialize U and V with the right sparsity pattern
    # 2. Solve for W linearly (least squares)
    # 3. Check if the result is exact
    # 4. If not, try again with different initialization
    
    # Sparsity pattern: each term has 2 nonzeros in each factor
    # Total nonzeros: 16*2 = 32 in U, 32 in V, 32 in W
    
    # For the 18 nonzero entries, we need coverage.
    # Let's try to assign each nonzero entry to be "primarily" covered by
    # one term, with other terms contributing 0 or small corrections.
    
    # Actually, let's try a simpler approach: start from rank-17 and try to
    # merge two terms into one.
    
    U17, V17, W17 = _rank17(n, m, p)
    R17 = U17.shape[1]
    
    # Try merging pairs of terms
    for i in range(R17):
        for j in range(i+1, R17):
            if time.time() > deadline:
                return None
            
            # Try to find alpha, beta such that:
            # alpha*U[:,i] + beta*U[:,j], alpha*V[:,i] + beta*V[:,j], alpha*W[:,i] + beta*W[:,j]
            # replaces both terms i and j, and the total sum remains T.
            #
            # This means: (alpha-1)*U[:,i]*V[:,i]*W[:,i] + (beta-1)*U[:,j]*V[:,j]*W[:,j] 
            #             + alpha*beta*(U[:,i]*V[:,j]*W[:,i] + U[:,i]*V[:,i]*W[:,j] + U[:,j]*V[:,i]*W[:,j] + U[:,j]*V[:,j]*W[:,i]) = 0
            #
            # This is complex. Let's just try specific alpha, beta values.
            
            for alpha in [-2, -1, -0.5, 0.5, 1, 2]:
                for beta in [-2, -1, -0.5, 0.5, 1, 2]:
                    if alpha == 1 and beta == 1:
                        continue  # No change
                    
                    # New term
                    U_new = alpha * U17[:, i] + beta * U17[:, j]
                    V_new = alpha * V17[:, i] + beta * V17[:, j]
                    W_new = alpha * W17[:, i] + beta * W17[:, j]
                    
                    # Build new decomposition: keep all terms except i and j, add new term
                    keep = [t for t in range(R17) if t != i and t != j]
                    Ux = np.column_stack([U17[:, keep], U_new.reshape(-1, 1)])
                    Vx = np.column_stack([V17[:, keep], V_new.reshape(-1, 1)])
                    Wx = np.column_stack([W17[:, keep], W_new.reshape(-1, 1)])
                    
                    if _verify_decomposition(Ux, Vx, Wx, T):
                        return (Ux, Vx, Wx)
    
    return None

def _try_rank16_alternating(n, m, p, deadline):
    """
    Alternating minimization approach for rank-16.
    Initialize from rank-17 by zeroing one column, then alternate solving
    for each factor while fixing the others.
    """
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    U17, V17, W17 = _rank17(n, m, p)
    R17 = U17.shape[1]
    
    # Try dropping each term and running alternating minimization
    for drop in range(R17):
        if time.time() > deadline:
            return None
        
        keep = [t for t in range(R17) if t != drop]
        U0 = U17[:, keep].copy()
        V0 = V17[:, keep].copy()
        W0 = W17[:, keep].copy()
        
        U_alt = U0.copy()
        V_alt = V0.copy()
        W_alt = W0.copy()
        
        for iteration in range(30):
            if time.time() > deadline:
                return None
            
            # Solve for W: for each c, T[:,:,c] = sum_r U[:,r] V[:,r] W[c,r]
            # T[:,:,c].flatten() = [outer(U[:,r], V[:,r]).flatten() for r] @ W[:,c]
            A_mat = np.zeros((n_a * n_b, U_alt.shape[1]))
            for r in range(U_alt.shape[1]):
                A_mat[:, r] = np.outer(U_alt[:, r], V_alt[:, r]).flatten()
            
            W_new = np.zeros((n_c, U_alt.shape[1]))
            for c in range(n_c):
                rhs_c = T[:, :, c].flatten()
                try:
                    W_new[c, :] = np.linalg.lstsq(A_mat, rhs_c, rcond=None)[0]
                except:
                    return None
            
            W_alt = W_new
            
            # Solve for U: for each a, T[a,:,:].flatten() = [outer(V[:,r], W[:,r]).flatten() for r] @ U[:,a]
            B_mat = np.zeros((n_b * n_c, U_alt.shape[1]))
            for r in range(U_alt.shape[1]):
                B_mat[:, r] = np.outer(V_alt[:, r], W_alt[:, r]).flatten()
            
            U_new = np.zeros((n_a, U_alt.shape[1]))
            for a in range(n_a):
                rhs_a = T[a, :, :].flatten()
                try:
                    U_new[a, :] = np.linalg.lstsq(B_mat, rhs_a, rcond=None)[0]
                except:
                    return None
            
            U_alt = U_new
            
            # Solve for V: for each b, T[:,b,:].flatten() = [outer(U[:,r], W[:,r]).flatten() for r] @ V[:,b]
            C_mat = np.zeros((n_a * n_c, U_alt.shape[1]))
            for r in range(U_alt.shape[1]):
                C_mat[:, r] = np.outer(U_alt[:, r], W_alt[:, r]).flatten()
            
            V_new = np.zeros((n_b, U_alt.shape[1]))
            for b in range(n_b):
                rhs_b = T[:, b, :].flatten()
                try:
                    V_new[b, :] = np.linalg.lstsq(C_mat, rhs_b, rcond=None)[0]
                except:
                    return None
            
            V_alt = V_new
        
        # Verify
        if _verify_decomposition(U_alt, V_alt, W_alt, T):
            return (U_alt, V_alt, W_alt)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # explicit safety margin (< 120s wall-clock)

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Baseline: verified rank-17 (Strassen + direct).
    U17, V17, W17 = _rank17(n, m, p)
    if _verify_decomposition(U17, V17, W17, T):
        base = (U17, V17, W17)
    else:
        base = _schoolbook(n, m, p)

    # Try rank-16 structured approach
    rank16_result = _try_rank16_structured(n, m, p, deadline)
    if rank16_result is not None:
        U, V, W = rank16_result
    else:
        # Try alternating minimization for rank-16
        rank16_alt = _try_rank16_alternating(n, m, p, deadline)
        if rank16_alt is not None:
            U, V, W = rank16_alt
        else:
            U, V, W = base

    # Final exact verification; fall back to schoolbook on any failure.
    if not _verify_decomposition(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
