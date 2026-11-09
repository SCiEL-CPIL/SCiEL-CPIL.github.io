"""Tensor decomposition search for 2x3 * 3x8 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
DEADLINE = 115.0  # safety margin before 120s limit

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple."""
    # T has shape (n, m, p) with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The tensor we need to decompose is the bilinear map: C[i,k] = sum_j A[i,j]*B[j,k]
    # As a 3-tensor: T[i,j,k] = delta (it's 1 for all valid indices)
    # Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, let me re-read the interface.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is the standard matrix multiplication tensor decomposition.
    return n, m, p

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m + j][r]) * Fraction(V[j*p + k][r]) * Fraction(W[k*n + i][r])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_bilinear_reduction(n, m, p):
    """
    For 2x3 * 3x8: The middle dimension is 3.
    Key insight: If we can find a decomposition where the rank is less than n*m*p,
    we exploit the fact that the sum over j only has m terms.
    
    One approach: Use the fact that for fixed i,k, the value is sum_j A[i,j]*B[j,k].
    This is a bilinear form in A and B for each (i,k) pair.
    
    Number of (i,k) pairs: n*p = 16.
    Each bilinear form has m=3 terms.
    
    A rank-1 term in the tensor decomposition corresponds to:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Let's think of it differently. The tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0.
    
    Actually T[a,b,c] = 1 iff there exist i,j,k such that a=im+j, b=jp+k, c=kn+i.
    Since the mapping (i,j,k) -> (im+j, jp+k, kn+i) is bijective (given dimensions),
    T[im+j, jp+k, kn+i] = 1 for all valid i,j,k, and 0 otherwise.
    
    This is the standard matrix multiplication tensor.
    
    Known lower bounds: For m x n * n x p, rank >= max(mn, np, pm) is not quite right.
    The trivial bound is m*n*p.
    
    For 2x3*3x8: m*n*p = 48.
    
    Let me try a random search approach with structured initialization.
    """
    pass

def _random_search_structured(n, m, p, max_rank, seed, time_limit):
    """
    Try to find a decomposition of rank < n*m*p.
    Use structured random search: initialize with schoolbook and try to merge terms.
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    
    best = None
    best_rank = n * m * p
    
    # Try ranks from best_rank-1 down
    for target_rank in range(best_rank - 1, max(15, -1), -1):
        if time.time() - start > time_limit:
            break
        
        # Try a few random initializations at this rank
        attempts = 3
        for attempt in range(attempts):
            if time.time() - start > time_limit:
                break
            
            # Initialize with random values
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Normalize
            for r in range(target_rank):
                norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
                if norm > 0:
                    U[:,r] /= norm**(1/3)
                    V[:,r] /= norm**(1/3)
                    W[:,r] /= norm**(1/3)
            
            # Gradient descent to minimize error
            lr = 0.01
            for step in range(50):
                if time.time() - start > time_limit:
                    break
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            # Target is 1
                            val = 0.0
                            grad_factor = 0.0
                            for r in range(target_rank):
                                val += U[a,r] * V[b,r] * W[c,r]
                                grad_factor = val - 1.0
                            
                            for r in range(target_rank):
                                dU[a,r] += grad_factor * V[b,r] * W[c,r]
                                dV[b,r] += grad_factor * U[a,r] * W[c,r]
                                dW[c,r] += grad_factor * U[a,r] * V[b,r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            val = 0.0
                            for r in range(target_rank):
                                val += U[a,r] * V[b,r] * W[c,r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Found a solution! Try to round to rationals
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank < n*m*p:
        # We found something, but need exact rational solution
        # For now, return schoolbook if we can't get exact
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try to improve with structured search
    start = time.time()
    time_budget = DEADLINE - start - 5  # leave 5s margin
    
    if time_budget > 10:
        # Try rank reduction search
        # For 2x3*3x8, the tensor has a special structure.
        # Let's try a more principled approach.
        
        # Key idea: The multiplication tensor for (m x n) * (n x p) can sometimes
        # be decomposed with rank less than m*n*p using the structure of the indices.
        
        # For n=3 (middle dim), we have:
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        # This is 16 independent bilinear forms, each of rank 3 (in the standard sense).
        # But they share variables, so the tensor rank might be less than 48.
        
        # Let's try a direct algebraic approach.
        # The tensor is sum_{i,j,k} e_{im+j} ⊗ e_{jp+k} ⊗ e_{kn+i}
        
        # One known technique: if one dimension is small, we can sometimes
        # use the fact that the tensor is "separable" in that dimension.
        
        # For m=3: T = sum_j (sum_i e_{im+j} ⊗ e_{jp+*}) ... this doesn't directly help.
        
        # Let me try a different approach: use the fact that we only need
        # integer/half-integer entries.
        
        # Try small rank values with a smarter initialization
        # based on the structure.
        
        # Actually, for this specific case (2,3,8), let's try to see if
        # rank 47 or lower is achievable.
        
        # The tensor rank of matrix multiplication is a hard problem.
        # For 2x2*2x2, rank is 7 (Strassen).
        # For 2x3*3x8, it's not well-studied.
        
        # Let's try a heuristic: start from schoolbook and try to "merge"
        # pairs of terms.
        
        # Schoolbook has 48 terms. Each term r has:
        # U: nonzero at (i_r, j_r) with value 1
        # V: nonzero at (j_r, k_r) with value 1  
        # W: nonzero at (k_r, i_r) with value 1
        
        # Can we replace two terms with one?
        # Term 1: (i1,j1,k1), Term 2: (i2,j2,k2)
        # We need: U1[a,b]*V1[b,c]*W1[c,a] + U2[a,b]*V2[b,c]*W2[c,a] = combined
        # This is generally not possible with rank 1.
        
        # But with the specific structure, maybe.
        
        # Let's just do a limited random search for now.
        found = _random_search_structured(n, m, p, n*m*p - 1, 42, min(time_budget, 60))
        if found is not None:
            result = found
    
    return result
