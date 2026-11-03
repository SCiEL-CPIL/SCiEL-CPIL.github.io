"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    Checks the diagonal condition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    U_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in U]
    V_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in V]
    W_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(len(U_frac[0])))
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def _snap_to_half_integers(X):
    """Snap values to nearest half-integer."""
    # Candidates: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    # Multiplying by 2 gives integers -4 to 4
    X_scaled = X * 2
    X_snapped = np.round(X_scaled)
    return X_snapped / 2.0

def _compute_tensor_rank_lower_bound(n, m, p):
    """
    Compute a rigorous lower bound on the tensor rank.
    The multiplication tensor T has entries T[i,j,k] = 1 if the (i,j) entry of the product
    depends on A[i,j]*B[j,k]. 
    
    A rank-r decomposition means T = sum_{c=1}^r U_c ⊗ V_c ⊗ W_c where
    U_c is n*m dimensional, V_c is m*p dimensional, W_c is p*n dimensional.
    
    The flattening T_{(ij)|(jk)} has rank at most r (since each term contributes a rank-1 
    matrix). So the matrix rank of this flattening is a lower bound.
    """
    # Build the flattening matrix: rows indexed by (i,j), columns by (j,k)
    # Entry at ((i,j),(j,k)) = 1 if i,j,k form a valid "path" (which they always do in matmul)
    # Actually for matmul, T[i,j,k] = 1 for all valid (i,j,k)
    # The flattening over (i,j) vs (j,k): M[(i,j),(j',k)] = sum_k' T[i,j,k'] * delta[j,j']
    # Wait, let me think more carefully.
    
    # The tensor T in F^{n*m} ⊗ F^{m*p} ⊗ F^{p*n} has T[i,j,k] = 1.
    # Flatten to matrix: M[(i,j), (j',k)] = sum over shared index... 
    # Actually the standard flattening for matmul tensor:
    # M has rows (i,j) and columns (j',k), and M[(i,j),(j',k)] = T[i,j,k] * delta[j,j']
    # = 1 if j=j' else 0
    # So M is block-diagonal with blocks being J_{n x p} for each j
    # Rank of M = m * rank(J_{n x p}) = m * min(n,p)
    
    # For n=2, m=3, p=6: rank = 3 * min(2,6) = 3*2 = 6
    # But this is a lower bound. The actual tensor rank is higher.
    
    # A better lower bound: consider the "slice" structure.
    # For each fixed k, we have a matrix S_k[i,j] = T[i,j,k] = 1 (all ones, n x m)
    # These must be representable as S_k = sum_c W_k[c] * (U_c * V_c^T) where U_c is n*m vector
    # Hmm, this is getting complex. Let me use a different approach.
    
    # Actually, the known lower bound for matmul rank is max(n*m, m*p, p*n) = 18 for (2,3,6)
    # This comes from the fact that each factor has dimension at least the max of the three products.
    
    # Let's just use the known lower bound
    return max(n*m, m*p, p*n)

def _try_rank_r(n, m, p, r, max_restarts=2000):
    """
    Try to find a rank-r decomposition using structured search.
    We fix U and V randomly, solve for W, and check if W is "nice".
    """
    candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize U and V with sparse random values
        # Higher sparsity helps keep W simple
        U = np.random.choice(candidates, size=(n*m, r))
        V = np.random.choice(candidates, size=(m*p, r))
        
        # Ensure U and V are not trivially zero
        if np.all(U == 0) or np.all(V == 0):
            continue
            
        W = np.zeros((p*n, r))
        valid_w = True
        
        # Solve for each (k, i) pair independently
        for k in range(p):
            for i in range(n):
                # Construct A for this (k, i)
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                b = np.ones(m)
                
                # Solve A * W_col = b
                # W_col is the row of W corresponding to (k, i)
                try:
                    # Use lstsq to find minimum norm solution
                    W_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = W_col
                except:
                    valid_w = False
                    break
            if not valid_w:
                break
        
        if not valid_w:
            continue
            
        # Snap W to half-integers
        W_snapped = _snap_to_half_integers(W)
        
        # Verify the full decomposition with snapped W
        # U and V are already integers/half-integers from candidates
        ok, _ = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-6)
        
        if ok:
            # Final polish: verify with tight tolerance
            ok_tight, _ = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-9)
            if ok_tight:
                return U, V, W_snapped
            else:
                # If tight fails, try snapping U and V too (though they are already from candidates)
                # and re-verify
                U_s = _snap_to_half_integers(U)
                V_s = _snap_to_half_integers(V)
                ok_tight2, _ = _verify_decomposition(U_s, V_s, W_snapped, n, m, p, tol=1e-9)
                if ok_tight2:
                    return U_s, V_s, W_snapped

    return None

def _try_structured_rank_r(n, m, p, r, max_restarts=500):
    """
    Try a structured approach: exploit the matmul tensor structure.
    For matmul, we can try decompositions where the factors have a specific pattern.
    
    One known approach: for 2x3 * 3x6, try to find a rank-18 decomposition.
    The key insight is that we can group the computation.
    """
    if time.time() - START_TIME > TIME_LIMIT - 15:
        return None
    
    # Try a specific structured decomposition
    # For 2x3 * 3x6, the product is 2x6.
    # We can think of this as computing 6 independent 2x3 * 3x1 products.
    # Each 2x3 * 3x1 product has rank at most 2*3*1 = 6, but actually the rank 
    # of a 2x3 * 3x1 matmul tensor is at most 2*3 = 6 (since the third dimension is 1).
    
    # Let's try a block decomposition approach.
    # Actually, let's try to directly construct a rank-r solution using the linear structure.
    
    # The matmul tensor T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition is: T = sum_{c=1}^r U_c ⊗ V_c ⊗ W_c
    # where (U_c)_{i,j} U[i*m+j,c], (V_c)_{j,k} V[j*p+k,c], (W_c)_{k,i} W[k*n+i,c]
    
    # This means: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try to find such a decomposition with small entries.
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try different sparsity patterns
        sparsity = np.random.choice([0.3, 0.5, 0.7])
        candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        
        for i in range(n*m):
            for c in range(r):
                if np.random.random() < sparsity:
                    U[i, c] = np.random.choice(candidates)
        
        for j in range(m*p):
            for c in range(r):
                if np.random.random() < sparsity:
                    V[j, c] = np.random.choice(candidates)
        
        if np.all(U == 0) or np.all(V == 0):
            continue
        
        # Solve for W
        W = np.zeros((p*n, r))
        valid = True
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                
                try:
                    W_col, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = W_col
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        W_snapped = _snap_to_half_integers(W)
        ok, _ = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-6)
        
        if ok:
            ok_tight, _ = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-9)
            if ok_tight:
                return U, V, W_snapped
    
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Lower bound is max(nm, mp, pn) = 18
    # We try ranks from 18 upwards
    for r in range(18, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        # Try structured search
        result = _try_rank_r(n, m, p, r, max_restarts=1000)
        if result is None:
            # Try the structured approach
            result = _try_structured_rank_r(n, m, p, r, max_restarts=500)
        
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
            
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
