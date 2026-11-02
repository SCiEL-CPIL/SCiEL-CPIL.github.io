"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # Target: T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,8)
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try to find rank-r decomposition with r < 64
    # Use iterative refinement: start from schoolbook, try to merge terms
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def check(U, V, W, tol=1e-8):
        """Check if U, V, W give the correct tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try rank 32, 36, 40, 44, 48, 52, 56, 60
    # For each, try to find a solution using a structured approach
    
    # Approach: use the fact that we can write 1 = sum of products
    # Try a "Hadamard-like" decomposition
    
    # For m=4, use the 4-point DFT/Hadamard structure
    # H = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    # H @ H.T = 4*I
    # So for vectors x, y in R^4: x·y = (1/4)(Hx)·(Hy)
    
    # This doesn't directly help with rank, but let's try a different structure
    
    # Key idea: for each (i,k) pair, we compute sum_j A[i,j]B[j,k]
    # We can group the j indices and use identities
    
    # For m=4: A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # = (A[i,0]+A[i,1])B[0,k] + (A[i,2]+A[i,3])B[1,k] + A[i,1](B[0,k]-B[1,k]) + A[i,3](B[2,k]-B[3,k])
    # That's still 4 terms.
    
    # Let's try: (A[i,0]+A[i,1]+A[i,2]+A[i,3]) * (B[0,k]+B[1,k]+B[2,k]+B[3,k]) / something
    # No, that gives cross terms.
    
    # I think for this specific problem, the best we can do with simple methods
    # is around rank 40-50. Let me try a numerical search.
    
    # Use scipy to solve the nonlinear system
    from scipy.optimize import fsolve
    
    def try_rank_numerical(r, max_iter=50):
        """Try to find rank-r decomposition using numerical optimization."""
        # Variables: U (nm*r), V (mp*r), W (pn*r)
        # Constraints: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0
        # That's n*m*p = 64 constraints
        # Variables: (nm+mp+pn)*r = 56*r
        # For r=32: 1792 vars, 64 constraints -> underdetermined, good!
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Scale to make products O(1)
        # Each product U*V*W should be O(1/r) so sum is O(1)
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        # Define residual function
        def residuals(params):
            U_flat = params[:nm*r].reshape(nm, r)
            V_flat = params[nm*r:nm*r+mp*r].reshape(mp, r)
            W_flat = params[nm*r+mp*r:].reshape(pn, r)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        res.append(s - 1.0)
            return res
        
        # This is too slow for fsolve with 1792 variables
        # Let's use a different approach: iterative refinement
        
        # Start from a valid high-rank solution and try to reduce rank
        # by setting some columns to zero and re-solving
        
        # Actually, let's try a greedy approach:
        # 1. Start with schoolbook (rank 64)
        # 2. Try to merge pairs of terms: replace two rank-1 terms with one
        #    by finding u', v', w' such that u1*v1*w1 + u2*v2*w2 = u'*v'*w'
        #    This is generally not possible, but for structured tensors it might be
        
        # Alternative: use the fact that the tensor is "separable" in a certain way
        
        # For now, let's just try random half-integer solutions for small ranks
        for seed in range(100):
            np.random.seed(seed)
            # Try half-integer entries
            vals = np.array([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
            U = np.random.choice(vals, size=(nm, r))
            V = np.random.choice(vals, size=(mp, r))
            W = np.random.choice(vals, size=(pn, r))
            
            if check(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 32 to 63
    for r in range(32, 64):
        result = try_rank_numerical(r)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
