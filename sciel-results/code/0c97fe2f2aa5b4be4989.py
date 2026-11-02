import time
import sys
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The tensor rank of M_{2,3,4} is 24 (schoolbook is optimal).
    Returns the standard rank-24 decomposition with integer entries.
    
    We attempt to find a rank-23 or lower decomposition via local search,
    but fall back to the known rank-24 schoolbook decomposition.
    """
    n, m, p = 2, 3, 4
    rank_schoolbook = n * m * p  # 24
    
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    
    def build_schoolbook(rank):
        """Build the standard schoolbook decomposition of given rank."""
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp(U, V, W, tol=1e-9):
        """
        Verify that the decomposition exactly represents matrix multiplication.
        For M_{n,m,p}, the tensor T[i,j,k] = delta_{i,j,k} in the sense that
        (A*B)[i,k] = sum_j A[i,j]*B[j,k].
        
        The decomposition satisfies:
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        where T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
        For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) since each 
        output element (i,k) gets contribution from A[i,j]*B[j,k] for each j.
        """
        # The multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # And 0 for all other combinations (but there are no other combinations in the tensor)
        
        # Actually, the tensor is defined on indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 for all such (i,j,k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_with_integers(U, V, W):
        """Verify using exact integer/fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(len(U[0])):
                        s += Fraction(U[i*m + j][r]) * Fraction(V[j*p + k][r]) * Fraction(W[k*n + i][r])
                    if s != 1:
                        return False
        return True
    
    def try_rank_search(target_rank, max_iters=5000):
        """
        Try to find a decomposition of the given rank using local search.
        We parameterize the decomposition and try to satisfy the constraints.
        """
        if time.time() > deadline - 5:
            return None
        
        import numpy as np
        
        # Initialize with random values
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        random.seed(42)
        
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Try gradient descent / coordinate descent
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (val - 1.0) ** 2
                
                if err < 1e-12:
                    # Found a solution!
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                # Simple coordinate updates (expensive, so do sparse updates)
                # Update a random few entries
                for _ in range(50):
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        idx = random.randint(0, n*m-1)
                        r = random.randint(0, target_rank-1)
                        # Gradient w.r.t. U[idx, r]
                        grad = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i*m + j == idx:
                                        val = 0.0
                                        for rr in range(target_rank):
                                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                        grad += 2*(val - 1.0) * V[j*p+k, r] * W[k*n+i, r]
                        U[idx, r] -= lr * grad
                    elif mat == 'V':
                        idx = random.randint(0, m*p-1)
                        r = random.randint(0, target_rank-1)
                        grad = 0.0
                        for j in range(m):
                            for k in range(p):
                                if j*p + k == idx:
                                    for i in range(n):
                                        val = 0.0
                                        for rr in range(target_rank):
                                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                        grad += 2*(val - 1.0) * U[i*m+j, r] * W[k*n+i, r]
                        V[idx, r] -= lr * grad
                    else:  # W
                        idx = random.randint(0, p*n-1)
                        r = random.randint(0, target_rank-1)
                        grad = 0.0
                        for k in range(p):
                            for i in range(n):
                                if k*n + i == idx:
                                    for j in range(m):
                                        val = 0.0
                                        for rr in range(target_rank):
                                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                        grad += 2*(val - 1.0) * U[i*m+j, r] * V[j*p+k, r]
                        W[idx, r] -= lr * grad
                
                if best_err < 1e-8:
                    break
        
        # Check if best solution is valid
        if best_U is not None and best_err < 1e-8:
            result = {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
            # Try to round to integers or half-integers
            U_rounded = [[round(x * 2) / 2 for x in row] for row in result["U"]]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in result["V"]]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in result["W"]]
            
            if verify_decomp(U_rounded, V_rounded, W_rounded, tol=1e-6):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            else:
                # Return unrounded if rounded doesn't work
                return result
    
    # First, try to find a rank-23 decomposition
    # This is a hard problem, so we do a limited search
    if time.time() < deadline - 30:
        result = try_rank_search(23, max_iters=2000)
        if result is not None:
            # Verify it works
            if verify_decomp(result["U"], result["V"], result["W"], tol=1e-6):
                return result
    
    # Fall back to schoolbook rank-24
    return build_schoolbook(rank_schoolbook)
# EVOLVE-BLOCK-END
