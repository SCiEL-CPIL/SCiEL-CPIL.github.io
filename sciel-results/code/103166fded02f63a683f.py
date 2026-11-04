import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # First try to find rank-16 via structured algebraic search
    result_16 = _try_rank16_algebraic(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17
    return _rank17_decomposition(n, m, p)


def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition with half-integer rounding."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(np.int64)
    Vi = np.rint(V_s * 2).astype(np.int64)
    Wi = np.rint(W_s * 2).astype(np.int64)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None


def _try_rank16_algebraic(n, m, p, deadline):
    """
    Structured algebraic search for rank-16 decomposition.
    
    Strategy: The 2x3x3 multiplication tensor can be decomposed by viewing
    it as two 1x3x3 tensors. We use a structured ansatz where most entries
    are fixed to {0, ±1, ±1/2} and we search over a small number of free
    parameters using a combination of:
    1. Pattern enumeration over structured templates
    2. Newton's method on the free parameters
    3. Rounding to half-integers and verification
    """
    rank = 16
    target = np.ones((n, m, p))
    
    # Strategy 1: Try to build from known 3x3x3 rank-7 (Strassen-like) patterns
    # combined with sharing between the two rows
    
    # The 2x3x3 tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We can think of this as: for each (j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Approach: Use a "block" structure where we share rank terms between rows
    # Let's try a template where:
    # - 7 terms handle the "diagonal" contributions for row 0
    # - 7 terms handle the "diagonal" contributions for row 1  
    # - 2 terms are shared "cross" terms
    # Total: 16
    
    # Template: Fix most entries, optimize a few free parameters
    # We'll use multiple template families
    
    best_err = float('inf')
    best_result = None
    
    # Template family 1: Based on extending the rank-17 by merging two terms
    # In rank-17, terms 7-16 are the "outer" terms for column 2 of V and row 2 of U
    # Can we merge any pairs?
    
    # Let's try: fix 14 of the 16 rank terms to known patterns, optimize 2 free terms
    # Actually, let's try a more systematic approach:
    
    # For each pair of "candidate merges" from the rank-17, try to find if
    # two rank-1 terms can be replaced by a single rank-1 term
    
    # Alternative: Direct structured search
    # Fix U, V, W to have entries from {0, ±1, ±1/2} except for a few free entries
    
    # Let's try: enumerate sparse patterns with small support
    # Each rank term (r) has at most 3 non-zero entries in U, V, W respectively
    # (since the target tensor is all-ones, each term must contribute to specific (i,j,k))
    
    # Actually, for the all-ones tensor, each rank term U[i,j,r]*V[j,k,r]*W[k,i,r]
    # must be non-zero for exactly the (i,j,k) where all three factors are non-zero
    
    # Let's use a different approach: algebraic elimination
    # Fix V and W to simple patterns, solve for U
    
    # Pattern: V[j,k,r] is non-zero for at most 2 (j,k) pairs per r
    # W[k,i,r] is non-zero for at most 2 (k,i) pairs per r
    
    # Try: V has rank-1 structure per r: V[j,k,r] = v_j[r] * w_k[r] (outer product)
    # W[k,i,r] = w_k[r] * u_i[r] (outer product)
    # Then U[i,j,r] * V[j,k,r] * W[k,i,r] = U[i,j,r] * v_j[r] * w_k[r] * w_k[r] * u_i[r]
    # This is getting complex. Let's try direct numerical search with structure.
    
    # Strategy: Random initialization with structured constraints + Newton refinement
    np.random.seed(42)
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Initialize with structured random pattern
        # Most entries are 0, some are ±1 or ±0.5
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Each rank term should have sparse structure
        for r in range(rank):
            # Choose 1-2 non-zero entries in U for this term
            nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
            
            nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
            
            nz_w = np.random.choice(p * n, size=np.random.randint(1, 3), replace=False)
            for idx in nz_w:
                W[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
        
        # Newton refinement on all parameters
        U, V, W = _newton_refine(U, V, W, n, m, p, max_iter=50, tol=1e-12)
        
        # Check if we found exact solution
        ok, Us, Vs, Ws = _verify(U, V, W, n, m, p)
        if ok:
            return {
                "U": Us.tolist(),
                "V": Vs.tolist(),
                "W": Ws.tolist()
            }
        
        # Track best error
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        err = np.sum((prod - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
    
    # If we found a near-solution, try to round and verify
    if best_result is not None and best_err < 1.0:
        U_b, V_b, W_b = best_result
        ok, Us, Vs, Ws = _verify(U_b, V_b, W_b, n, m, p)
        if ok:
            return {
                "U": Us.tolist(),
                "V": Vs.tolist(),
                "W": Ws.tolist()
            }
    
    return None


def _newton_refine(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Newton's method refinement for tensor decomposition."""
    rank = U.shape[1]
    target = np.ones((n, m, p))
    
    for it in range(max_iter):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - target
        err = np.sum(res**2)
        
        if err < tol:
            break
        
        # Compute gradients (Jacobian of the residual)
        # dU[i,j,r] = sum_{k} res[i,j,k] * V[j,k,r] * W[k,i,r]
        dU = np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Approximate Hessian using Gauss-Newton (J^T J)
        # For speed, use a damped Newton step with adaptive step size
        dU_flat = dU.reshape(-1)
        dV_flat = dV.reshape(-1)
        dW_flat = dW.reshape(-1)
        
        # Gradient norm
        grad_norm = np.sqrt(np.sum(dU_flat**2) + np.sum(dV_flat**2) + np.sum(dW_flat**2))
        
        if grad_norm < 1e-15:
            break
        
        # Adaptive step size with backtracking
        step = 1.0
        for _ in range(20):
            U_new = U - step * dU_flat.reshape(U.shape)
            V_new = V - step * dV_flat.reshape(V.shape)
            W_new = W - step * dW_flat.reshape(W.shape)
            
            U_t_new = U_new.reshape(n, m, rank)
            V_t_new = V_new.reshape(m, p, rank)
            W_t_new = W_new.reshape(p, n, rank)
            prod_new = np.einsum('ijr,jkr,kir->ijk', U_t_new, V_t_new, W_t_new)
            err_new = np.sum((prod_new - target)**2)
            
            if err_new < err:
                U, V, W = U_new, V_new, W_new
                break
            step *= 0.5
        
        else:
            # No improvement, try smaller step or break
            break
    
    return U, V, W


def _try_rank16_direct(n, m, p, deadline):
    """
    Direct structured search: exploit the fact that the target tensor is all-ones.
    
    Key observation: T[i,j,k] = 1 for all (i,j,k). This means the decomposition
    must satisfy: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    We can fix W to have a specific structure and solve for U and V.
    """
    rank = 16
    target = np.ones((n, m, p))
    
    # Fix W to a structured pattern: W[k,i,r] = 1 for specific (k,i,r)
    # Try: W has at most one non-zero per (k,i) pair across all r
    # i.e., for each (k,i), there's exactly one r where W[k,i,r] != 0
    
    # This means: for each (k,i), the contribution to T[i,j,k] from all r is
    # sum_j' U[i,j',r_ki]*V[j',k,r_ki] where r_ki is the unique r for (k,i)
    # This must equal 1 for all j.
    
    # So we need: for each (k,i), there exists r such that
    # sum_j U[i,j,r]*V[j,k,r] = 1 (with W[k,i,r]=1)
    
    # This is a system of linear equations in U and V!
    
    # Let's enumerate possible assignments of r to (k,i) pairs
    # There are p*n = 6 (k,i) pairs and rank=16 terms
    # Each (k,i) can be assigned to any of the 16 terms
    
    # But this is a huge search space (16^6). Let's be smarter.
    
    # Alternative: Fix U and V to have simple structure, solve for W
    # If U[i,j,r] and V[j,k,r] are fixed, then for each (i,j,k):
    # W[k,i,r] must satisfy: sum_r W[k,i,r] * (U[i,j,r]*V[j,k,r]) = 1
    # This is a linear system in W!
    
    # Let's try: fix U and V to sparse patterns, solve linear system for W
    
    np.random.seed(123)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate sparse U and V
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for r in range(rank):
            # U[i,j,r]: pick 1-2 (i,j) pairs
            nz = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz:
                U[idx, r] = np.random.choice([-1, 1])
            
            nz = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Now solve for W: for each (i,j,k), sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
        # This is: for each (i,j,k): A_{(i,j,k), (k',i')} ... hmm, this is not linear in W alone
        # because the coefficient depends on (i,j,k)
        
        # Actually, let me reconsider. For fixed U and V:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Let's define B[i,j,k,r] = U[i,j,r] * V[j,k,r]
        # Then T[i,j,k] = sum_r B[i,j,k,r] * W[k,i,r]
        # 
        # For fixed (i,k), varying j: T[i,j,k] = sum_r B[i,j,k,r] * W[k,i,r] = 1
        # This gives m equations in the p*n variables W[k,i,r] for r=0..rank-1
        # But W[k,i,r] appears in multiple (i,j,k) equations
        # 
        # Total equations: n*m*p = 18
        # Total unknowns: p*n*rank = 6*16 = 96
        # Underdetermined! We can solve this as a least-squares problem.
        
        # Build the linear system: for each (i,j,k), equation:
        # sum_r B[i,j,k,r] * W[k,i,r] = 1
        # 
        # Let's index W as W_flat[(k,i,r)] = W[k*n+i, r]
        # For equation (i,j,k): coefficient of W_flat[(k,i,r)] is B[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r]
        
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (k * n + i) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
        
        # Solve least squares
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_flat.reshape(p * n, rank)
        except:
            continue
        
        # Check if solution is close to half-integers
        W_rounded = np.round(W_sol * 2) / 2.0
        W_int = np.rint(W_rounded * 2).astype(np.int64)
        
        # Verify
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W_int.reshape(p, n, rank)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        if np.all(prod == 8):
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = W_rounded
            return {
                "U": U_s.tolist(),
                "V": V_s.tolist(),
                "W": W_s.tolist()
            }
        
        # Also try without rounding W, just check if U and V are already half-integers
        # and if the residual is small
        err = np.sum((prod - 8)**2)
        if err < 1e-6:
            # Try rounding all
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W_sol * 2) / 2.0
            Ui = np.rint(U_r * 2).astype(np.int64).reshape(n, m, rank)
            Vi = np.rint(V_r * 2).astype(np.int64).reshape(m, p, rank)
            Wi = np.rint(W_r * 2).astype(np.int64).reshape(p, n, rank)
            prod2 = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
            if np.all(prod2 == 8):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    return None
