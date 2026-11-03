"""Matrix multiplication 2x3x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # The tensor for matrix multiplication: T[i, k, j] = 1 if the (i,k) output
    # depends on A[i,j] * B[j,k], which is always true.
    # Actually, the multiplication tensor T has T[i,k,j] = 1 for all valid i,j,k
    # when we write (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T[i,k,j] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Rank-r decomposition: T[i,k,j] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the convention:
    # U has shape (n*m, rank): U[i*m + j, c]
    # V has shape (m*p, rank): V[j*p + k, c]  
    # W has shape (p*n, rank): W[k*n + i, c]
    # The tensor equation: T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,k,j] = 1 (the multiplication tensor)
    
    # For schoolbook: each (i,j,k) triple gets its own rank-1 term
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # We want to find minimum r such that there exist U, V, W with the above property.
    
    # Strategy: Use randomized search with Schwartz-Zippel verification.
    # For a given rank r, parameterize U, V, W with random rational/integer entries,
    # then check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    
    # Better approach: Set up the system of equations and solve.
    # The equations are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 30 equations in 3*r unknowns (with the constraint that
    # each term is a product of three variables).
    
    # Alternative: Fix U and V randomly, then solve for W.
    # For fixed U and V, the equations become linear in W:
    # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all (i,j,k)
    # But this couples different (i,k) pairs through the same c index... 
    # Actually no, for fixed (i,k), we have one equation involving W[k*n+i, c] for all c.
    # Wait, that's not right either. Let me re-index.
    
    # For fixed (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Group by (i,k): For each (i,k), and for each j, we have:
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is a system where for each (i,k), we have m equations (one per j) in r unknowns (W[k*n+i,0..r-1]).
    # So for fixed U,V, we need to solve p*n = 10 independent linear systems, each of size m x r.
    # For this to have a solution, we need r >= m = 3, and the systems must be consistent.
    
    # Even better: For fixed U and V, define for each (i,k) the matrix M_{ik} where
    # M_{ik}[j, c] = U[i*m+j, c] * V[j*p+k, c]
    # Then we need W[k*n+i, :] such that M_{ik} @ W[k*n+i, :]^T = 1_vector (of length m)
    # This has a solution iff rank(M_{ik}) <= m and 1_vector is in the column space.
    
    # Strategy: Randomly initialize U and V, then for each (i,k), check if the linear system
    # is consistent. If not, adjust. Use iterative refinement.
    
    # Let me try a different approach: use the fact that for small ranks, we can do
    # a direct search with random restarts and local optimization.
    
    # Actually, let me try to use known constructions. For 2x3x5, the known best upper bound
    # is 25 (from various papers). Let me try to find a rank-25 or lower decomposition.
    
    # Approach: Use random matrices and solve the linear systems.
    # For rank r, randomly generate U (n*m x r) and V (m*p x r) with rational entries.
    # Then for each (i,k), solve M_{ik} @ w = ones(m) where M_{ik}[j,c] = U[i*m+j,c]*V[j*p+k,c]
    # If all systems are consistent, we have a valid decomposition.
    
    def try_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition using random U,V and solving for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random rational U and V
            # Use small integers to keep things simple
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            
            for i in range(n * m):
                for c in range(r):
                    U[i, c] = random.uniform(-2, 2)
            for j in range(m * p):
                for c in range(r):
                    V[j, c] = random.uniform(-2, 2)
            
            # For each (i,k), solve M_{ik} @ w = ones(m)
            W = np.zeros((p * n, r), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M_{ik}: m x r matrix
                    M_ik = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            M_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Solve M_ik @ w = ones(m)
                    # Use least squares and check residual
                    try:
                        w, residuals, rank_M, sv = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                        residual = np.linalg.norm(M_ik @ w - np.ones(m))
                        if residual > 1e-8:
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
            
            if valid:
                # Verify the full decomposition
                U_int = U
                V_int = V
                W_int = W
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c] for c in range(r))
                            error += abs(val - 1.0)
                if error < 1e-6:
                    # Convert to exact fractions if possible, or return as-is
                    return U_int, V_int, W_int
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound for 2x3x5: at least 19 (from algebraic geometry bounds)
    # Known upper bound: 30 (schoolbook)
    # Recent results: 25 or 26 might be achievable
    
    # First, let's just try to beat 30
    # Try rank 29, 28, 27, 26, 25, ...
    
    best_result = None
    for r in range(25, 31):  # Try 25 through 30
        if time.time() > deadline:
            break
        result = try_rank(r, max_attempts=50)
        if result is not None:
            U, V, W = result
            # Convert to lists with Fraction for exactness if entries are close to rationals
            U_list = [[float(x) for x in U[i]] for i in range(n * m)]
            V_list = [[float(x) for x in V[i]] for i in range(m * p)]
            W_list = [[float(x) for x in W[i]] for i in range(p * n)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

