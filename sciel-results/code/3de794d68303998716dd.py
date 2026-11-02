"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U * V * W (tensor product) equals the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    T = np.zeros((n, m, p), dtype=float)
    for l in range(rank):
        u = np.array([U[i * m + j][l] for i in range(n) for j in range(m)], dtype=float).reshape(n, m)
        v = np.array([V[j * p + k][l] for j in range(m) for k in range(p)], dtype=float).reshape(m, p)
        w = np.array([W[k * n + i][l] for k in range(p) for i in range(n)], dtype=float).reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                if u[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Expected tensor: T[i,j,k] = 1 if the entry is the identity-like product, else 0
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T_{i,j,k} is 1 if this term appears in the product, i.e., always 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k because
    # C[i,k] = sum_j A[i,j] B[j,k], so each term (i,j,k) has coefficient 1.
    expected = np.ones((n, m, p), dtype=float)
    if not np.allclose(T, expected, atol=1e-10):
        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for l in range(rank):
        pass
    # Check that for every (i,j,k), sum_l U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for l in range(rank):
                    total += Fraction(U[i*m+j][l]) * Fraction(V[j*p+k][l]) * Fraction(W[k*n+i][l])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 60
    
    # Try to find lower rank decompositions using randomized approach
    # The tensor is T[i,j,k] = 1 for all i,j,k (since every term appears)
    # We need sum_l U[i,j]_l * V[j,k]_l * W[k,i]_l = 1 for all i,j,k
    
    # Known: Strassen-like algorithms can reduce rank for some sizes
    # For 3x4x5, let's try a structured approach
    
    # First, try to verify if rank 60 is achievable with half-integers
    # (schoolbook uses integers, so it's valid)
    
    # Attempt 1: Try random rank-59 decomposition
    # For rank r, we have r*(n+m+p) = r*12 free parameters and n*m*p = 60 constraints
    # For r=59: 708 params, 60 constraints - very underdetermined
    # But the constraints are nonlinear (tensor rank-1)
    
    # Approach: Use the fact that we can modify the schoolbook decomposition
    # by combining some rank-1 terms
    
    # Actually, let's try a direct algebraic approach
    # For the all-ones tensor, we can use:
    # 1 = (a+b+c)(d+e+f) ... no, that's not right
    
    # Let's try: if we can express the tensor as sum of fewer rank-1 tensors
    # The all-ones tensor of size n x m x p has rank min(n,m,p) if we allow
    # arbitrary entries? No, that's for the sum tensor.
    
    # Actually the matrix multiplication tensor is NOT all ones in the sense of
    # having a simple structure. Each entry T[i,j,k]=1 means every triple contributes.
    
    # Let me reconsider. The standard matrix multiplication tensor T_{n,m,p} has
    # T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p.
    # This is because C[i,k] = sum_j A[i,j]*B[j,k], and each (i,j,k) term has coeff 1.
    
    # So we need: sum_l U_l[i,j] * V_l[j,k] * W_l[k,i] = 1 for all i,j,k
    
    # One known trick: use the identity 1 = x*x for appropriate x, but that doesn't help directly.
    
    # Let's try a computational approach: random initialization + iterative refinement
    np.random.seed(42)
    
    for target_rank in range(59, 0, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
        
        # Try several random initializations
        max_attempts = max(5, int((deadline - time.time()) / 5)) if time.time() < deadline else 1
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random small values
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative optimization: minimize ||T - sum_l U_l V_l W_l||_F^2
            # Using gradient descent or alternating least squares
            best_loss = float('inf')
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                T = np.zeros((n, m, p))
                for l in range(target_rank):
                    u = U_init[:, l].reshape(n, m)
                    v = V_init[:, l].reshape(m, p)
                    w = W_init[:, l].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            if u[i, j] != 0:
                                for k in range(p):
                                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                
                loss = np.sum((T - 1.0) ** 2)
                if loss < best_loss:
                    best_loss = loss
                
                if loss < 1e-12:
                    break
                
                # Alternating least squares
                # Update U: for each l, solve for U_l given V_l, W_l
                # This is complex, so use simple gradient step
                lr = 0.01
                for l in range(target_rank):
                    u = U_init[:, l].reshape(n, m).copy()
                    v = V_init[:, l].reshape(m, p).copy()
                    w = W_init[:, l].reshape(p, n).copy()
                    
                    # Gradient of loss w.r.t. u[i,j] = 2 * sum_k (T[i,j,k]-1) * v[j,k] * w[k,i]
                    for i in range(n):
                        for j in range(m):
                            grad = 0.0
                            for k in range(p):
                                grad += (T[i, j, k] - 1.0) * v[j, k] * w[k, i]
                            U_init[i * m + j, l] -= lr * 2 * grad
                    
                    # Gradient w.r.t. v[j,k] = 2 * sum_i (T[i,j,k]-1) * u[i,j] * w[k,i]
                    for j in range(m):
                        for k in range(p):
                            grad = 0.0
                            for i in range(n):
                                grad += (T[i, j, k] - 1.0) * u[i, j] * w[k, i]
                            V_init[j * p + k, l] -= lr * 2 * grad
                    
                    # Gradient w.r.t. w[k,i] = 2 * sum_j (T[i,j,k]-1) * u[i,j] * v[j,k]
                    for k in range(p):
                        for i in range(n):
                            grad = 0.0
                            for j in range(m):
                                grad += (T[i, j, k] - 1.0) * u[i, j] * v[j, k]
                            W_init[k * n + i, l] -= lr * 2 * grad
            
            # Check if we found an exact solution
            if best_loss < 1e-10:
                # Convert to fractions and verify
                U_frac = []
                for row in range(n * m):
                    U_frac.append([])
                    for col in range(target_rank):
                        val = U_init[row, col]
                        # Round to nearest half-integer
                        rounded = round(val * 2) / 2
                        U_frac[-1].append(rounded)
                
                V_frac = []
                for row in range(m * p):
                    V_frac.append([])
                    for col in range(target_rank):
                        val = V_init[row, col]
                        rounded = round(val * 2) / 2
                        V_frac[-1].append(rounded)
                
                W_frac = []
                for row in range(p * n):
                    W_frac.append([])
                    for col in range(target_rank):
                        val = W_init[row, col]
                        rounded = round(val * 2) / 2
                        W_frac[-1].append(rounded)
                
                if _verify(U_frac, V_frac, W_frac, n, m, p):
                    best = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = target_rank
                    break
        
        if target_rank < best_rank:
            break
    
    return best
# EVOLVE-BLOCK-END
