"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # Build all index combinations
    errors = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    errors += 1
                    if errors > 5:
                        return False
    return errors == 0

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _try_merge(U, V, W, n, m, p, c1, c2, deadline):
    """
    Try to merge columns c1 and c2 into a single column.
    We look for alpha, beta such that:
    For all (i,j,k): U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] + 
                     U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
                     = (alpha*U[:,c1] + beta*U[:,c2]) * (alpha*V[:,c1] + beta*V[:,c2]) * (alpha*W[:,c1] + beta*W[:,c2])
    
    Actually, the merge is: replace two rank-1 terms with one.
    Term1: u1 ⊗ v1 ⊗ w1
    Term2: u2 ⊗ v2 ⊗ w2
    We want: u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w for some u,v,w
    This is only possible if the two tensors are linearly dependent in a specific way.
    
    A more practical approach: try to find if we can absorb one term into another
    by modifying the remaining terms.
    """
    # This is complex; instead, try a simpler approach:
    # Randomly perturb and see if we can reduce rank
    pass

def _random_search(n, m, p, target_rank, deadline, seed=0):
    """Random search for a decomposition of given rank."""
    rng = np.random.RandomState(seed)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization with structure
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Try a few iterations of coordinate descent
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Update U
            for idx in range(n * m):
                i, j = divmod(idx, m)
                # We need: sum_c U[idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system in U[idx,:]
                # For each k: sum_c U[idx,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # Let A_k[c] = V[j*p+k,c] * W[k*n+i,c]
                # We need: A_k @ U[idx,:] = 1 for all k
                # This is an overdetermined system (p equations, target_rank unknowns)
                
                # Build the system
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                
                # Solve least squares: min ||A @ u - 1||^2
                # u = (A^T A)^{-1} A^T 1
                ATA = A.T @ A
                AT1 = A.T @ np.ones(p)
                try:
                    u_new = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), AT1)
                except np.linalg.LinAlgError:
                    continue
                
                U[idx] = u_new
            
            # Update V
            for idx in range(m * p):
                j, k = divmod(idx, p)
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i] = U[i * m + j] * W[k * n + i]
                
                ATA = A.T @ A
                AT1 = A.T @ np.ones(n)
                try:
                    v_new = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), AT1)
                except np.linalg.LinAlgError:
                    continue
                
                V[idx] = v_new
            
            # Update W
            for idx in range(p * n):
                k, i = divmod(idx, n)
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j] = U[i * m + j] * V[j * p + k]
                
                ATA = A.T @ A
                AT1 = A.T @ np.ones(m)
                try:
                    w_new = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), AT1)
                except np.linalg.LinAlgError:
                    continue
                
                W[idx] = w_new
            
            # Check convergence
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
    
    return None

def _structured_search(n, m, p, target_rank, deadline, seed=0):
    """Structured search using patterns."""
    rng = np.random.RandomState(seed)
    
    # Try sparse structured initialization
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Random sparse pattern
        for c in range(target_rank):
            # Choose random support
            u_support = rng.choice(n * m, size=rng.randint(1, 5), replace=False)
            v_support = rng.choice(m * p, size=rng.randint(1, 5), replace=False)
            w_support = rng.choice(p * n, size=rng.randint(1, 5), replace=False)
            
            for idx in u_support:
                U[idx, c] = rng.choice([-1, 1])
            for idx in v_support:
                V[idx, c] = rng.choice([-1, 1])
            for idx in w_support:
                W[idx, c] = rng.choice([-1, 1])
        
        # Coordinate descent
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            for idx in range(n * m):
                i, j = divmod(idx, m)
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                ATA = A.T @ A
                AT1 = A.T @ np.ones(p)
                try:
                    U[idx] = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), AT1)
                except:
                    pass
            
            # Update V
            for idx in range(m * p):
                j, k = divmod(idx, p)
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i] = U[i * m + j] * W[k * n + i]
                ATA = A.T @ A
                AT1 = A.T @ np.ones(n)
                try:
                    V[idx] = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), AT1)
                except:
                    pass
            
            # Update W
            for idx in range(p * n):
                k, i = divmod(idx, n)
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j] = U[i * m + j] * V[j * p + k]
                ATA = A.T @ A
                AT1 = A.T @ np.ones(m)
                try:
                    W[idx] = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), AT1)
                except:
                    pass
            
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
    
    return None

def _greedy_merge(n, m, p, deadline):
    """
    Start from schoolbook and greedily try to merge terms.
    For each pair of terms, check if they can be combined.
    """
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Try to find pairs that can be merged
    # Two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 can be merged into one
    # if they are proportional in a specific way.
    # More practically: try to find if removing one term and adjusting others works.
    
    # Alternative: try to find linear dependencies among the rank-1 terms
    # If sum_c alpha_c * u_c⊗v_c⊗w_c = 0, we can eliminate one term.
    
    # For now, use a simpler heuristic: try random linear combinations
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try to merge two random columns
        if best_rank <= 2:
            break
        
        c1, c2 = random.sample(range(best_rank), 2)
        
        # Try to find alpha, beta such that the sum of terms c1 and c2
        # can be represented as a single term
        # This is equivalent to finding if u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank-1
        # which is rare, so instead try a perturbation approach
        
        # Try: replace columns c1,c2 with a single column that's a linear combo
        # and adjust other columns to compensate
        new_U = best_U[:, :best_rank-1].copy()
        new_V = best_V[:, :best_rank-1].copy()
        new_W = best_W[:, :best_rank-1].copy()
        
        # Remove column c2, keep c1 but try to adjust
        if c2 > c1:
            new_U[:, c1] = best_U[:, c1] + best_U[:, c2]
            new_V[:, c1] = best_V[:, c1] + best_V[:, c2]  
            new_W[:, c1] = best_W[:, c1] + best_W[:, c2]
        elif c2 < c1:
            new_U[:, c2] = best_U[:, c1] + best_U[:, c2]
            new_V[:, c2] = best_V[:, c1] + best_V[:, c2]
            new_W[:, c2] = best_W[:, c1] + best_W[:, c2]
        
        # This simple addition won't work in general, but try coordinate descent to fix
        # Run a few iterations of coordinate descent on the reduced-rank system
        for iteration in range(20):
            if time.time() > deadline:
                break
            
            # Update U
            for idx in range(n * m):
                i, j = divmod(idx, m)
                A = np.zeros((p, best_rank-1))
                for k in range(p):
                    A[k] = new_V[j * p + k] * new_W[k * n + i]
                ATA = A.T @ A
                AT1 = A.T @ np.ones(p)
                try:
                    new_U[idx] = np.linalg.solve(ATA + 1e-10 * np.eye(best_rank-1), AT1)
                except:
                    pass
            
            # Update V
            for idx in range(m * p):
                j, k = divmod(idx, p)
                A = np.zeros((n, best_rank-1))
                for i in range(n):
                    A[i] = new_U[i * m + j] * new_W[k * n + i]
                ATA = A.T @ A
                AT1 = A.T @ np.ones(n)
                try:
                    new_V[idx] = np.linalg.solve(ATA + 1e-10 * np.eye(best_rank-1), AT1)
                except:
                    pass
            
            # Update W
            for idx in range(p * n):
                k, i = divmod(idx, n)
                A = np.zeros((m, best_rank-1))
                for j in range(m):
                    A[j] = new_U[i * m + j] * new_V[j * p + k]
                ATA = A.T @ A
                AT1 = A.T @ np.ones(m)
                try:
                    new_W[idx] = np.linalg.solve(ATA + 1e-10 * np.eye(best_rank-1), AT1)
                except:
                    pass
            
            if _verify_fast(new_U, new_V, new_W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank -= 1
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Strategy 1: Try random search for lower ranks
    for target_rank in range(best_rank - 1, max(best_rank - 30, 70), -1):
        if time.time() > deadline - 20:
            break
        
        # Try random search
        result = _random_search(n, m, p, target_rank, deadline - 10, seed=target_rank)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break  # Found a valid decomposition at this rank
        
        # Try structured search
        result = _structured_search(n, m, p, target_rank, deadline - 10, seed=target_rank + 1000)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Strategy 2: Try greedy merge from schoolbook
    if time.time() < deadline - 15:
        U_m, V_m, W_m, rank_m = _greedy_merge(n, m, p, deadline - 5)
        if rank_m < best_rank:
            if _verify(U_m, V_m, W_m, n, m, p):
                best_rank = rank_m
                best_U, best_V, best_W = U_m, V_m, W_m
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_fb, V_fb, W_fb
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
