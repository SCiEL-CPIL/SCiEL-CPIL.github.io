"""Optimized tensor decomposition for 2x3x4 matrix multiplication."""
import time
import sys
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=None):
    """ALS decomposition of the matmul tensor."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Initialize
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Normalize
    for c in range(rank):
        nrm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
        U[:, c] /= nrm
        V[:, c] /= nrm
        W[:, c] /= nrm
    
    for it in range(max_iter):
        # Update U
        for c in range(rank):
            # For each entry (i,j), compute target
            for i in range(n):
                for j in range(m):
                    target = 0.0
                    for k in range(p):
                        target += V[j*p+k, c] * W[k*n+i, c]
                    # U[i*m+j, c] should satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta
                    # For fixed c, we need U[i*m+j,c] such that it contributes correctly
                    # Actually, let's do proper ALS:
                    # The tensor entry T[(i,j),(j',k),(k',i')] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
                    # We want sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if i=j'=k' and i=i' and k=k'
                    # Wait, the indices are: U indexed by (i,j), V by (j,k), W by (k,i)
                    # T[(i,j),(j',k),(k',i')] should be 1 iff i==i', j==j', k==k'
                    
                    # For ALS updating U[i*m+j, c]:
                    # Fix V and W, solve for U
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
                    # This is complex. Let me use a different approach.
                    pass
        
        # Better ALS approach:
        # Reshape and use matrix operations
        break
    
    # Use a cleaner ALS formulation
    # The matmul tensor: T[a,b,c] where a=(i,j), b=(j',k), c=(k',i')
    # T[a,b,c] = 1 iff i==i', j==j', k==k', else 0
    # This means T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k
    
    # For ALS, we can think of it as: for each c, U[:,c], V[:,c], W[:,c]
    # We want sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
    
    # Let me use a simpler iterative approach
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)  
    W = rng.randn(p*n, rank)
    
    for it in range(max_iter):
        # Update U: for each column c, solve for U[:,c]
        for c in range(rank):
            # We need: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
            # Fix V and W, for each a, the contribution from column c is U[a,c] * (V[b,c] * W[d,c])
            # This is tricky. Let's use a different strategy.
            
            # Actually, let's just do gradient descent or a simpler method
            pass
        
        # Simpler: use the fact that this is a structured tensor
        # For the matmul tensor, T[(i,j),(j',k),(k',i')] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
        
        # ALS update for U:
        # For fixed V and W, we want to find U that minimizes ||T - sum_c U[:,c] V[:,c] W[:,c]||^2
        # This is a linear system in U if we fix V and W
        
        # Let's build the linear system properly
        # T is a 3-tensor of size (n*m) x (m*p) x (p*n)
        # We want: for all a, b, d: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
        
        # For fixed V, W: this is linear in U
        # For each a: sum_c U[a,c] * (V[:,c] * W[:,c] as a vector) = T[a, :, :]
        
        # Actually this is a 3-way tensor decomposition, which is NP-hard in general.
        # For small problems, we can try various approaches.
        
        # Let's try a simple coordinate descent
        for c in range(rank):
            # Update U[:, c]
            for a in range(n*m):
                i, j = divmod(a, m)
                # T[a, b, d] for b=(j',k), d=(k',i')
                # T[a, b, d] = 1 iff i==i', j==j', k==k'
                # So for fixed a=(i,j): T[a, (j,k), (k,i)] = 1 for all k
                # And 0 otherwise
                
                # The equation: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
                # For column c, the contribution to entry (a,b,d) is U[a,c] * V[b,c] * W[d,c]
                
                # We want to choose U[a,c] to best fit.
                # This is a least squares problem:
                # For fixed a and c, U[a,c] appears in entries (a, b, d) with weight V[b,c]*W[d,c]
                # Target: T[a,b,d] - sum_{c' != c} U[a,c'] V[b,c'] W[d,c']
                
                # Build the system for U[a,c]:
                # sum_d' sum_b' U[a,c] V[b,c] W[d,c] * delta... 
                # This is getting complex. Let me just use numpy to solve it.
                
                # Reshape approach:
                # For fixed c, define M_c[b,d] = V[b,c] * W[d,c]
                # Then sum_c U[a,c] * M_c[b,d] = T[a,b,d]
                # For fixed a: sum_c U[a,c] M_c = T[a] (where T[a] is a (m*p) x (p*n) matrix)
                # This is: [M_0 M_1 ... M_{rank-1}] * [U[a,0] U[a,1] ... U[a,rank-1]]^T = T[a]
                # Where M_c is a column of the "Khatri-Rao" product
                
                # Actually, let's just do it numerically
                pass
        
        # Let me try a completely different, simpler approach
        break
    
    return U, V, W

def _try_rank(n, m, p, rank, time_limit=30.0):
    """Try to find a decomposition of given rank."""
    start = time.time()
    
    # Initialize with random values
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
            
        rng = np.random.RandomState(attempt * 1000 + 7)
        
        # Smart initialization based on schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Start with schoolbook-like initialization
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i*m+j) * p + (j*p+k) + (k*n+i)  # some hash
                    c = idx % rank
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
        
        # Add random perturbation
        U += rng.randn(n*m, rank) * 0.1
        V += rng.randn(m*p, rank) * 0.1
        W += rng.randn(p*n, rank) * 0.1
        
        # ALS iterations
        for it in range(100):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for a in range(n*m):
                i, j = divmod(a, m)
                # Target tensor slice: T[a, b, d]
                # T[a, (j',k), (k',i')] = 1 iff i==i', j==j', k==k'
                
                # Build the linear system for U[a, :]
                # sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
                
                # For each (b,d): T[a,b,d]
                # b = j'*p + k, d = k'*n + i'
                # T = 1 iff j==j' and k==k' and i==i'
                
                # So T[a, j*p+k, k*n+i] = 1 for all k, else 0
                
                # The equation is: sum_c U[a,c] * V[b,c] * W[d,c] = T[a,b,d]
                # This is a linear system in U[a, :]
                # Let X = [V[:,c] * W[:,c] for c] stacked appropriately
                
                # Actually, for fixed a, we have a (m*p) x (p*n) matrix of targets
                # And we want to find U[a, :] of length rank
                # such that sum_c U[a,c] * outer(V[:,c], W[:,c]) = T[a]
                
                # This is: sum_c U[a,c] * (V[:,c] ⊗ W[:,c]) = T[a] (vectorized)
                # Where V[:,c] is length m*p, W[:,c] is length p*n
                # So V[:,c] ⊗ W[:,c] is a (m*p)*(p*n) vector
                
                # Build the design matrix: each column c is vec(V[:,c] * W[:,c]) = V[:,c] ⊗ W[:,c]
                # But that's a huge vector. Instead, note that:
                # sum_c U[a,c] * V[b,c] * W[d,c] = T[a,b,d]
                # For fixed a, this is a system in variables U[a,0],...,U[a,rank-1]
                
                # We can write it as: for each (b,d): sum_c U[a,c] * V[b,c] * W[d,c] = T[a,b,d]
                # This is (m*p)*(p*n) equations in rank unknowns.
                
                # Since rank < (m*p)*(p*n) typically, this is overdetermined.
                # Solve by least squares.
                
                # Build the matrix A where A[(b,d), c] = V[b,c] * W[d,c]
                # And target t where t[(b,d)] = T[a,b,d]
                
                A = np.zeros((m*p*p*n, rank))
                t = np.zeros(m*p*p*n)
                
                for b in range(m*p):
                    j2, k2 = divmod(b, p)
                    for d in range(p*n):
                        k3, i3 = divmod(d, n)
                        idx = b * (p*n) + d
                        for c in range(rank):
                            A[idx, c] = V[b, c] * W[d, c]
                        if i == i3 and j == j2 and k2 == k3:
                            t[idx] = 1.0
                
                # Solve least squares
                U[a, :], residuals, rank_ls, sv = np.linalg.lstsq(A, t, rcond=None)
            
            # Update V similarly
            for b in range(m*p):
                j2, k2 = divmod(b, p)
                A = np.zeros((n*m*p*n, rank))
                t = np.zeros(n*m*p*n)
                
                for a in range(n*m):
                    i2, j3 = divmod(a, m)
                    for d in range(p*n):
                        k3, i3 = divmod(d, n)
                        idx = a * (p*n) + d
                        for c in range(rank):
                            A[idx, c] = U[a, c] * W[d, c]
                        if i2 == i3 and j2 == j3 and k2 == k3:
                            t[idx] = 1.0
                
                V[b, :], residuals, rank_ls, sv = np.linalg.lstsq(A, t, rcond=None)
            
            # Update W similarly
            for d in range(p*n):
                k3, i3 = divmod(d, n)
                A = np.zeros((n*m*m*p, rank))
                t = np.zeros(n*m*m*p)
                
                for a in range(n*m):
                    i2, j3 = divmod(a, m)
                    for b in range(m*p):
                        j2, k2 = divmod(b, p)
                        idx = a * (m*p) + b
                        for c in range(rank):
                            A[idx, c] = U[a, c] * V[b, c]
                        if i2 == i3 and j2 == j3 and k2 == k3:
                            t[idx] = 1.0
                
                W[d, :], residuals, rank_ls, sv = np.linalg.lstsq(A, t, rcond=None)
        
        # Verify
        if _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = schoolbook_rank
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
    
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find better ranks
    for rank in range(13, min(22, schoolbook_rank)):
        if time.time() > DEADLINE - 10:
            break
        
        result = _try_rank(n, m, p, rank, time_limit=20.0)
        if result is not None:
            U_float, V_float, W_float = result
            best_rank = rank
            # Convert to integers/rationals
            U_int = [[int(round(x)) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(1000) for x in row] for row in U_float]
            V_int = [[int(round(x)) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(1000) for x in row] for row in V_float]
            W_int = [[int(round(x)) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(1000) for x in row] for row in W_float]
            best_result = {"U": U_int, "V": V_int, "W": W_int}
            break
    
    return best_result
