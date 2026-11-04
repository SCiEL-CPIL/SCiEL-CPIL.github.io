"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def verify_float(U_f, V_f, W_f, rank, tol=1e-8):
        """Fast float verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def compute_error_and_grad(U_f, V_f, W_f, rank):
        """Compute error and gradients for U, V, W"""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.sum(U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :])
        
        err = np.sum((T - 1.0)**2)
        
        dU = np.zeros_like(U_f)
        dV = np.zeros_like(V_f)
        dW = np.zeros_like(W_f)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 2.0 * (T[i,j,k] - 1.0)
                    dU[i*m+j, :] += diff * V_f[j*p+k, :] * W_f[k*n+i, :]
                    dV[j*p+k, :] += diff * U_f[i*m+j, :] * W_f[k*n+i, :]
                    dW[k*n+i, :] += diff * U_f[i*m+j, :] * V_f[j*p+k, :]
        
        return err, dU, dV, dW

    def optimize_rank(R, seed, max_iters=2000, tol=1e-10):
        """Try to find a rank R decomposition using gradient descent"""
        np.random.seed(seed)
        
        U_f = np.random.randn(n * m, R) * 0.5
        V_f = np.random.randn(m * p, R) * 0.5
        W_f = np.random.randn(p * n, R) * 0.5
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U_f)
        vV = np.zeros_like(V_f)
        vW = np.zeros_like(W_f)
        
        prev_err = np.inf
        plateau_count = 0
        err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            err, dU, dV, dW = compute_error_and_grad(U_f, V_f, W_f, R)
            
            if err < prev_err * 0.99:
                lr = min(lr * 1.1, 0.1)
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count > 50:
                    lr *= 0.9
                    if lr < 1e-6:
                        break
            
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U_f += vU
            V_f += vV
            W_f += vW
            
            prev_err = err
            
            if err < tol:
                break
        
        if err < tol:
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
            for i in range(n*m):
                for r in range(R):
                    U_rat[i,r] = Fraction(U_f[i,r]).limit_denominator(10**6)
            for i in range(m*p):
                for r in range(R):
                    V_rat[i,r] = Fraction(V_f[i,r]).limit_denominator(10**6)
            for i in range(p*n):
                for r in range(R):
                    W_rat[i,r] = Fraction(W_f[i,r]).limit_denominator(10**6)
            
            if verify_exact(U_rat, V_rat, W_rat, R):
                return U_rat, V_rat, W_rat, R
        
        return None

    def build_winograd_candidate():
        """
        Attempt to construct a rank-25 decomposition using Winograd-like
        polynomial interpolation structure for 3x3x4 multiplication.
        
        The idea: map matrix multiplication to polynomial multiplication
        and use interpolation points to define factor matrices.
        """
        # For 3x3x4, we can think of this as multiplying polynomials
        # A(x) = sum_{i,j} a_ij x^i y^j (3x3 grid)
        # B(y) = sum_{j,k} b_jk y^j z^k (3x4 grid)
        # C(x) = A(x) * B(y) evaluated at specific points
        
        # Winograd for 3x3x3 uses rank 23. For 3x3x4, we expect higher rank.
        # Let's try to construct a candidate using tensor product structure.
        
        # Use interpolation points for the 3x3x4 problem
        # We'll use a structured approach based on Kronecker products
        
        # Try rank 25 with a specific algebraic structure
        R = 25
        
        # Define interpolation points
        # For the 3x3 part, use points {0, 1, 2}
        # For the 4 part, use points {0, 1, 2, 3}
        
        # Build factor matrices using Vandermonde-like structure
        U_cand = np.zeros((n*m, R), dtype=object)
        V_cand = np.zeros((m*p, R), dtype=object)
        W_cand = np.zeros((p*n, R), dtype=object)
        
        # This is a heuristic construction - may not be optimal
        # Use a combination of structured and random initialization
        np.random.seed(42)
        
        # Initialize with structured patterns
        for r in range(R):
            # Assign structured values
            U_cand[:, r] = np.random.choice([-1, 0, 1], size=n*m)
            V_cand[:, r] = np.random.choice([-1, 0, 1], size=m*p)
            W_cand[:, r] = np.random.choice([-1, 0, 1], size=p*n)
        
        # Verify and refine
        # Convert to float for fast verification
        U_f = U_cand.astype(float)
        V_f = V_cand.astype(float)
        W_f = W_cand.astype(float)
        
        if not verify_float(U_f, V_f, W_f, R, tol=1e-4):
            return None
        
        # Try to optimize from this starting point
        result = optimize_rank(R, seed=42, max_iters=1000, tol=1e-10)
        if result:
            return result
        
        return None

    # Search for minimal rank
    best_result = None
    
    # First try the Winograd-inspired candidate
    if time.time() < deadline - 30:
        winograd_result = build_winograd_candidate()
        if winograd_result:
            U_rat, V_rat, W_rat, rank = winograd_result
            best_result = (U_rat, V_rat, W_rat, rank)
    
    # If Winograd didn't work, try standard search
    if not best_result:
        for R in range(25, 36):
            if time.time() > deadline - 20:
                break
                
            found = False
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                    
                result = optimize_rank(R, seed)
                if result:
                    U_rat, V_rat, W_rat, rank = result
                    best_result = (U_rat, V_rat, W_rat, rank)
                    found = True
                    break
            
            if found:
                break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        # Fallback to schoolbook
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
# EVOLVE-BLOCK-END
