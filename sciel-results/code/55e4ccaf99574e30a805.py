"""Tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    The tensor T[i,j,k] for multiplication is 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    Actually the tensor for matrix mult: T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in C[i,k].
    
    The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait, let me re-read the convention.
    
    W[k*n+i, r] - so W is indexed by (k, i) with k in range p, i in range n.
    
    The matrix multiplication tensor: for multiplying A (n x m) by B (m x p) to get C (n x p),
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 (the structure tensor).
    A rank-1 decomposition: T[i,j,k] = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r]
    
    But wait - the standard formulation: the matrix multiplication tensor M_{n,m,p} is a tensor
    in R^{n x m} ⊗ R^{m x p} ⊗ R^{p x n} (or similar). Let me think about this more carefully.
    
    Actually, the standard way: the tensor for matrix multiplication of (n×m)·(m×p) → (n×p)
    is T ∈ R^{nm × mp × pn} where T[(i,j), (j',k), (k',i')] = δ_{j,j'} δ_{k,k'} δ_{i,i'}
    
    Hmm, let me just use the convention from the task:
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    The tensor entry T[i,j,k] (representing the coefficient of A[i,j]*B[j,k] in C[i,k]) should equal:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And T[i,j,k] = 1 for all valid (i,j,k) since each product A[i,j]*B[j,k] appears exactly once in C[i,k].
    """
    rank = len(U[0]) if U else 0
    # Build the tensor as a 3D array
    # T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Use einsum for efficiency
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is: np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
    T_computed = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
    
    # Expected: all ones
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T_computed, T_expected, atol=1e-10)

def _verify_exact(U, V, W, n=N, m=M, p=P):
    """Verify with exact rational arithmetic."""
    rank = len(U[0]) if U else 0
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = N, M, P
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = nm * mp  # 60
    
    # Try to find lower rank decompositions
    # The matrix multiplication tensor M_{3,4,5} has rank at most 60 (schoolbook).
    # We'll try various approaches to reduce rank.
    
    # Approach: Use the fact that we can potentially combine terms.
    # For small matrices, we can try random rank-r decompositions and verify.
    
    # First, let's try a few known techniques:
    # 1. Random search with rational rounding
    # 2. Greedy combination of schoolbook terms
    
    deadline = 140  # seconds, leave 10s margin
    
    # Try ranks from best_rank - 1 downward
    # But first, let's see if we can do better than 60.
    
    # Known lower bounds: rank(M_{n,m,p}) >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Upper bound: 60 (schoolbook)
    # For 3x4x5, the Strassen-like algorithms might help but those are for square matrices.
    
    # Let's try a systematic approach:
    # Generate random rank-r factorizations and check if they work.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try to find decompositions with rank < 60
    # We'll use a randomized approach:
    # For a given rank r, generate random U, V, W and check if they decompose the tensor.
    
    # Actually, for exact decomposition, we need to solve a system of equations.
    # The tensor has n*m*p = 60 entries, all equal to 1.
    # Unknowns: r * (nm + mp + pn) = r * (12 + 20 + 15) = r * 47
    # Equations: 60
    # For r < 60/47 ≈ 1.28, we can't even have enough unknowns.
    # Wait, that's not right. Each rank-1 term contributes to all 60 entries.
    # Actually the equations are: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # That's 60 equations in r*47 unknowns (but they're coupled).
    
    # Let me think differently. The tensor is a specific 3-way array.
    # A rank-r decomposition means we can write it as sum of r rank-1 tensors.
    
    # For the matrix multiplication tensor, there are known results:
    # rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{3,3,3}) <= 23 (known upper bound), lower bound 19
    
    # For 3x4x5, I don't know the exact rank. Let's try to find it computationally.
    
    # Strategy: Try ranks from 20 upward (since lower bound is 20) and see if we can find a valid decomposition.
    # But this is a non-convex problem, so random search may not work well.
    
    # Alternative: Try to combine schoolbook terms.
    # Each schoolbook term is a rank-1 tensor: e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is a basis vector in R^{nm}, etc.
    
    # Can we combine two such terms into a lower-rank representation?
    # In general, no - the rank of a sum is at most the sum of ranks, and can be less.
    
    # Let's try a different approach: use the structure of the problem.
    # The matrix multiplication tensor for n×m × m×p → n×p can be written as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ e_{j,k} ⊗ e_{k,i}
    
    # Hmm, let me just try random search with verification for various ranks.
    
    def try_random_rank(r, attempts=100):
        """Try to find a rank-r decomposition via random initialization."""
        for _ in range(attempts):
            if _elapsed() > deadline:
                return None
            
            # Initialize with small random values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Check if this happens to be a valid decomposition (very unlikely for random init)
            # Instead, let's try to solve the system directly.
            pass
        return None
    
    # Better approach: Use the fact that for the matrix multiplication tensor,
    # we can parameterize the decomposition more cleverly.
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms.
    
    # Actually, let's try a more principled approach.
    # The matrix multiplication tensor M_{n,m,p} can be decomposed using
    # the "partitioning" technique.
    
    # For 3x4x5:
    # We can partition the 4 intermediate dimension into parts.
    # E.g., write 4 = 2 + 2, then use block matrix multiplication.
    # But this doesn't directly reduce rank.
    
    # Let's try the following: since we need exact integer/half-integer entries,
    # let's search over small rational values.
    
    # For rank r, we have r*47 unknowns and 60 equations.
    # If r >= 60/47 ≈ 2, we have more unknowns than equations (in a sense).
    # But the equations are nonlinear (bilinear/trilinear).
    
    # Let's try rank 20, 21, ..., 60 and see if we can find valid decompositions.
    # We'll use a randomized local search.
    
    def search_rank(r, max_iter=50):
        """Try to find a rank-r decomposition."""
        if _elapsed() > deadline:
            return None
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for init in range(max_iter):
            if _elapsed() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Normalize to prevent divergence
            for rr in range(r):
                nU = np.linalg.norm(U[:, rr])
                nV = np.linalg.norm(V[:, rr])
                nW = np.linalg.norm(W[:, rr])
                if nU > 0 and nV > 0 and nW > 0:
                    scale = (nU * nV * nW) ** (1/3)
                    U[:, rr] /= scale
                    V[:, rr] /= scale
                    W[:, rr] /= scale
            
            # Gradient descent on the error
            # Error = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            lr = 0.01
            for it in range(100):
                if _elapsed() > deadline:
                    return None
                
                # Compute current tensor
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
                err = np.sum((T - 1)**2)
                
                if err < 1e-12:
                    # Found an exact solution (in floating point)
                    return U, V, W
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                # Compute gradients
                # d/dU[i,j,r] = 2 * (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r] summed over k
                # Actually: dT[i,j,k]/dU[i',j',r'] = δ_{i,i'} δ_{j,j'} V[j,k,r'] W[k,i']
                # So gradient wrt U[i,j,r] = 2 * sum_k (T[i,j,k]-1) * V[j,k,r] * W[k,i]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # grad_U[i,j,r] = 2 * sum_k (T[i,j,k]-1) * V[j,k,r] * W[k,i]
                # = 2 * np.einsum('ijk,jkr,kir->ijr', T-1, V_3d, W_3d)
                grad_U = 2 * np.einsum('ijk,jkr,kir->ijr', T - 1, V_3d, W_3d).reshape(nm, r)
                
                # grad_V[j,k,r] = 2 * sum_i (T[i,j,k]-1) * U[i,j,r] * W[k,i]
                # = 2 * np.einsum('ijk,ijr,kir->jkr', T-1, U_3d, W_3d)
                grad_V = 2 * np.einsum('ijk,ijr,kir->jkr', T - 1, U_3d, W_3d).reshape(mp, r)
                
                # grad_W[k,i,r] = 2 * sum_j (T[i,j,k]-1) * U[i,j,r] * V[j,k,r]
                # = 2 * np.einsum('ijk,ijr,jkr->kir', T-1, U_3d, V_3d)
                grad_W = 2 * np.einsum('ijk,ijr,jkr->kir', T - 1, U_3d, V_3d).reshape(pn, r)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Renormalize
                for rr in range(r):
                    nU = np.linalg.norm(U[:, rr])
                    nV = np.linalg.norm(V[:, rr])
                    nW = np.linalg.norm(W[:, rr])
                    if nU > 0 and nV > 0 and nW > 0:
                        scale = (nU * nV * nW) ** (1/3)
                        U[:, rr] /= scale
                        V[:, rr] /= scale
                        W[:, rr] /= scale
            
            # Check best solution
            if best_err < 1e-10:
                return best_U, best_V, best_W
        
        return None
    
    # Try ranks from 20 upward
    # But first, let's check if rank 60 (schoolbook) is indeed valid
    assert _verify(best["U"], best["V"], best["W"])
    
    # Try to find lower rank
    # Start from rank 25 and go up to 59
    for r in range(25, 60):
        if _elapsed() > deadline:
            break
        
        result = search_rank(r, max_iter=20)
        if result is not None:
            U, V, W = result
            # Convert to fractions and verify
            U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U.tolist()]
            V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V.tolist()]
            W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W.tolist()]
            
            if _verify_exact(U_frac, V_frac, W_frac):
                # Found a valid decomposition!
                return {"U": U_frac, "V": V_frac, "W": W_frac}
            else:
                # Try rounding to half-integers
                U_half = [[round(x*2)/2 for x in row] for row in U.tolist()]
                V_half = [[round(x*2)/2 for x in row] for row in V.tolist()]
                W_half = [[round(x*2)/2 for x in row] for row in W.tolist()]
                
                U_frac2 = [[Fraction(x) for x in row] for row in U_half]
                V_frac2 = [[Fraction(x) for x in row] for row in V_half]
                W_frac2 = [[Fraction(x) for x in row] for row in W_half]
                
                if _verify_exact(U_frac2, V_frac2, W_frac2):
                    return {"U": U_frac2, "V": V_frac2, "W": W_frac2}
    
    # If we couldn't find a better decomposition, return schoolbook
    return best

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
