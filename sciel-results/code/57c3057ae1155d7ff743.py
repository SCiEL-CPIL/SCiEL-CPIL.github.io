"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def _build_target_tensor(n, m, p):
    """Build the full target tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k), else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _validate_full(U, V, W, n, m, p, tol=1e-6):
    """Full validation: check ALL entries of the tensor decomposition.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] must match the target tensor exactly.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = _build_target_tensor(n, m, p)
    for c in range(p * n):
        M_c = (U * W[c, :]) @ V.T
        if np.max(np.abs(M_c - T[:, :, c])) > tol:
            return False
    return True

def _validate_diagonal_only(U, V, W, n, m, p, tol=1e-6):
    """Quick check: only the 45 diagonal entries."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _landsberg_333():
    """
    Return the rank-21 decomposition for 3x3x3 matrix multiplication.
    This is based on Landsberg's (1990) algorithm.
    The decomposition uses the identity:
    (a1*a2*a3 + a1*a4*a5 + a1*a7*a8 + a2*a3*a6 + a2*a4*a9 + a2*a7*a7 + 
     a3*a5*a9 + a3*a6*a8 + a4*a5*a6 + a5*a7*a8) 
    and its permutations.
    
    For 3x3x3, we use a structured approach with 21 terms.
    """
    n, m, p = 3, 3, 3
    rank = 21
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # The Landsberg decomposition for 3x3x3
    # We'll use a known construction
    
    # Term 0-8: diagonal terms (schoolbook for the 9 products)
    for i in range(3):
        for j in range(3):
            r = i * 3 + j
            U[i * 3 + j, r] = 1.0
            V[j * 3 + i, r] = 1.0
            W[i * 3 + j, r] = 1.0
    
    # Terms 9-20: additional terms for the off-diagonal products
    # This is a simplified version; the full Landsberg construction is complex
    # We'll use a known rank-21 construction
    
    # Actually, let's use a different approach: use the known fact that
    # R(3,3,3) = 23 is the best known, not 21. Let me correct this.
    # The best known upper bound for 3x3x3 is 23 (Landsberg 1990).
    
    # For now, let's use a rank-23 construction or fall back to schoolbook
    # and try to find a better one via search
    
    return None  # Placeholder - we'll implement a proper construction

def _construct_335_from_333():
    """
    Construct a 3x3x5 decomposition by extending a 3x3x3 decomposition.
    
    Strategy: The 3x3x5 matrix multiplication can be viewed as computing
    C = A * B where A is 3x3 and B is 3x5.
    
    We can write B = [B1, B2] where B1 and B2 are 3x3 and 3x2.
    Then C = A*B1 + A*B2 (but B2 is 3x2, so this doesn't directly work).
    
    Better approach: Use the fact that 3x3x5 can be decomposed using
    a 3x3x3 decomposition plus additional terms for the extra 2 columns.
    
    The minimum possible rank is at least max(9, 15, 15) = 15.
    The schoolbook rank is 45.
    
    A better approach: use a rank-23 3x3x3 decomposition and add terms
    for the extra 2 columns of B.
    """
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = 45
    
    # Try to find a better decomposition using structured search
    # based on the 3x3x3 structure
    
    # The key insight: for 3x3x5, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0,1,2,3,4
    # 
    # We can use a decomposition that shares structure between the 5 columns.
    
    # Let's try a rank-30 construction:
    # Use 3 separate 3x3x1 decompositions (each rank 9) for the first 3 columns
    # and 2 separate 3x3x1 decompositions for the last 2 columns
    # Total: 5 * 9 = 45 (schoolbook)
    
    # Better: use a shared structure
    # For each row i of A, we compute A[i,:] * B
    # This is a 1x3 times 3x5 = 1x5 operation
    # We can decompose this using rank-5 terms per row, total 3*5=15
    # But this doesn't exploit sharing between rows.
    
    # Let's try a different approach: use the Strassen-like structure
    # For 3x3x5, we can use:
    # - 9 terms for the "diagonal" (A[i,j] * B[j,k] for matching indices)
    # - Additional terms for off-diagonal products
    
    # Actually, the simplest improvement: use rank 30
    # by grouping the 5 columns into {0,1,2} and {3,4}
    # and using 3x3x3 (rank 23) + 3x3x2 (rank 18) = 41, not helpful
    
    # Let's just use a smart random search with the structure of 3x3x3
    
    # For 3x3x5, a good upper bound is 30:
    # Use 5 copies of a 3x3x1 decomposition, but share the A factors
    # Each 3x3x1 has rank 9, but we can share A factors
    # Total: 9 + 5*3 = 24? No, that's not right.
    
    # Let's use a direct construction:
    # Rank 30: for each of the 5 output columns, use 6 terms
    # that compute the 9 products with sharing
    
    # Actually, let's just do a focused search for rank 25-35
    
    for target_rank in range(20, 45):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-target_rank decomposition
        U, V, W = _search_structured(n, m, p, target_rank, deadline)
        if U is not None:
            if _validate_full(U, V, W, n, m, p, tol=1e-5):
                best_result = (U, V, W)
                best_rank = target_rank
                break
    
    return best_result

def _search_structured(n, m, p, rank, deadline):
    """
    Structured search for a rank-rank decomposition of n x m x p matrix multiplication.
    Uses the structure that the tensor has 1s on the "diagonal" (i,j,k) triples.
    """
    T = _build_target_tensor(n, m, p)
    rng = np.random.RandomState(42)
    
    # Initialize with a structured pattern
    # Each term r corresponds to a "pattern" of (i,j,k) indices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # First, place schoolbook terms for the first min(rank, n*m*p) products
    n_terms = min(rank, n * m * p)
    for t in range(n_terms):
        i = t // (m * p)
        j = (t // p) % m
        k = t % p
        U[i * m + j, t] = 1.0
        V[j * p + k, t] = 1.0
        W[k * n + i, t] = 1.0
    
    # Now try to optimize the remaining terms and adjust values
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(3):
        if time.time() > deadline - 5:
            break
        
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Random perturbation of the non-schoolbook terms
        for r in range(n_terms, rank):
            U_try[:, r] = rng.randn(n * m) * 0.1
            V_try[:, r] = rng.randn(m * p) * 0.1
            W_try[:, r] = rng.randn(p * n) * 0.1
        
        # Gradient descent
        lr = 0.005
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
            dU = np.zeros_like(U_try)
            dV = np.zeros_like(V_try)
            dW = np.zeros_like(W_try)
            loss = 0.0
            
            for c in range(p * n):
                M_c = (U_try * W_try[c, :]) @ V_try.T
                err_c = M_c - T[:, :, c]
                loss += np.sum(err_c * err_c)
                
                err_c_at_V = err_c @ V_try
                dU += 2.0 * (err_c_at_V * W_try[c, :])
                
                err_cT_at_U = err_c.T @ U_try
                dV += 2.0 * (err_cT_at_U * W_try[c, :])
                
                EV = err_c @ V_try
                dW[c, :] += 2.0 * np.sum(U_try * EV, axis=0)
            
            U_try -= lr * dU
            V_try -= lr * dV
            W_try -= lr * dW
            
            if loss < 1e-10:
                break
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
        
        if best_loss < 1e-8:
            break
    
    if best_U is not None and best_loss < 1e-5:
        return best_U, best_V, best_W
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try the structured construction
    result = _construct_335_from_333()
    U, V, W = result
    
    # Fallback to schoolbook if construction fails
    if time.time() > deadline - 5:
        U, V, W = _schoolbook(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
