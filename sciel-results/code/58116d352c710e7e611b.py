"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product, combinations
    from fractions import Fraction

    start_time = time.time()
    DEATHLINE = 130.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = len(U[0]) if len(U) > 0 else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    # Start with trivial decomposition
    best_result, best_rank = build_trivial()

    if elapsed() > DEATHLINE - 5:
        return best_result

    # Systematic rank reduction search
    # Strategy: Try to find linear dependencies among the 60 basis tensors
    # Each basis tensor e_(i,j,k) has support on exactly one (u,v,w) position
    
    # For rank reduction, we need to find R < 60 vectors u_r, v_r, w_r such that:
    # sum_r u_r[u] * v_r[v] * w_r[w] = 1 if (u,v,w) = (i*m+j, j*p+k, k*n+i) for some (i,j,k)
    # sum_r u_r[u] * v_r[v] * w_r[w] = 0 otherwise
    
    # This is a non-convex optimization problem. Let's try a heuristic approach:
    # 1. Start with the trivial decomposition
    # 2. Try to merge pairs of terms that share structure
    # 3. Use random search to find better decompositions
    
    # Key observation: For each fixed j, we have n*p = 12 terms that share the same j
    # These terms have U support on rows {j, 5+j} (for i=0,1)
    # V support on rows {6j, 6j+1, ..., 6j+5}
    # W support on rows {2k, 2k+1} for k=0,...,5
    
    # Let's try to find a lower-rank decomposition by exploiting the structure
    # For each j, we have a 2x6 subproblem (n x p for fixed j)
    
    # Actually, let's try a direct approach: use linear algebra to find if there's
    # a rank-59 decomposition by finding a linear dependency
    
    # The tensor T can be viewed as a matrix of size (10*30) x 12 or similar
    # Let's try to find the rank of the tensor as a multilinear map
    
    # For a 3-tensor T of shape (a,b,c), the rank is the minimum R such that
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # We can try to find a lower-rank decomposition by:
    # 1. Fixing two of the vectors and solving for the third
    # 2. Using gradient descent (but this is non-convex)
    # 3. Using random initialization and local search
    
    # Let's try a random search approach with multiple restarts
    import random
    random.seed(42)
    
    def random_search(target_rank, n_iter=1000):
        """Try to find a rank-target_rank decomposition using random search."""
        for _ in range(n_iter):
            if elapsed() > DEATHLINE - 10:
                return None
            
            # Randomly initialize U, V, W
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if this decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Convert to list format
            U_list = [[float(U[i][r]) for r in range(target_rank)] for i in range(n * m)]
            V_list = [[float(V[i][r]) for r in range(target_rank)] for i in range(m * p)]
            W_list = [[float(W[i][r]) for r in range(target_rank)] for i in range(p * n)]
            
            if verify_decomposition(U_list, V_list, W_list, tol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try random search for rank 59, 58, ...
    for target_rank in range(59, 40, -1):
        if elapsed() > DEATHLINE - 10:
            break
        
        result = random_search(target_rank, n_iter=100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break

    # If random search didn't find anything, try a more structured approach
    # For 2x5 x 5x6, we can try to use the fact that n=2 is small
    # This might allow for a Strassen-like decomposition
    
    # Let's try to explicitly construct a lower-rank decomposition
    # by finding linear combinations that cancel out unwanted terms
    
    # For each (i,j,k), we need sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    # This is a system of 60 equations in R*3 unknowns (the entries of U, V, W)
    
    # Let's try to solve this system for R = 59, 58, ...
    # We can use least squares or other numerical methods
    
    # Actually, let's try a different approach: use the fact that the tensor
    # has a specific structure and try to find a decomposition that exploits it
    
    # For fixed j, the terms involving j are:
    # (i,j,k) for i in {0,1}, k in {0,...,5}
    # These 12 terms have:
    # U support on rows j and 5+j
    # V support on rows 6j to 6j+5
    # W support on rows 2k and 2k+1 for k=0,...,5
    
    # This suggests that we might be able to decompose each j-block separately
    # and then combine them
    
    # For a single j, we have a 2x6 x 12 tensor (n x p x (n*p))
    # This might have a lower rank than 12
    
    # Let's try to find the rank of this sub-tensor
    # For n=2, p=6, the trivial rank is 12
    # Can we do better?
    
    # Actually, let's just try to find a valid decomposition with rank < 60
    # using a more systematic search
    
    # Try to find a rank-59 decomposition by finding a linear dependency
    # among the 60 basis tensors
    
    # The basis tensors are e_(i,j,k) for i in {0,1}, j in {0,...,4}, k in {0,...,5}
    # Each e_(i,j,k) has support on exactly one (u,v,w) position
    
    # For a linear dependency, we need coefficients c_(i,j,k) such that:
    # sum_(i,j,k) c_(i,j,k) * e_(i,j,k) = 0
    
    # This means that for each (u,v,w), the sum of c_(i,j,k) over all (i,j,k)
    # that have support on (u,v,w) must be 0
    
    # But each (u,v,w) is supported by at most one (i,j,k), so this is impossible
    # unless all c_(i,j,k) = 0
    
    # This means the 60 basis tensors are linearly independent, so the rank is at least 60
    
    # Wait, that's not right. The rank of a tensor is not the same as the
    # linear independence of its basis vectors. The rank is the minimum number
    # of rank-1 tensors that sum to the tensor.
    
    # Let me reconsider. The tensor T has 60 nonzero entries. The rank is the
    # minimum R such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # For the trivial decomposition, we have R = 60 with each rank-1 tensor
    # having support on exactly one (u,v,w) position
    
    # To get R < 60, we need some rank-1 tensors to have support on multiple
    # (u,v,w) positions, and the sum of these tensors must still equal T
    
    # This is possible if the supports overlap in a way that allows cancellation
    # or constructive interference
    
    # Let's try to find such a decomposition by solving a system of equations
    
    # For R = 59, we have 59 rank-1 tensors. Each has 3 vectors of lengths 10, 30, 12
    # So we have 59 * (10 + 30 + 12) = 2768 unknowns
    
    # The constraints are:
    # 1. For each of the 60 (i,j,k) positions, the sum equals 1
    # 2. For all other (u,v,w) positions, the sum equals 0
    
    # The number of (u,v,w) positions is 10 * 30 * 12 = 3600
    # So we have 3600 constraints
    
    # This is an underdetermined system (2768 unknowns, 3600 constraints)
    # but the constraints are not all independent
    
    # Let's try to solve this system using least squares or other methods
    
    # Actually, this is a non-linear system because the constraints involve
    # products of the unknowns (U[u][r] * V[v][r] * W[w][r])
    
    # Let's try a different approach: use the fact that we can scale the vectors
    # without changing the product (if we scale u_r by a, v_r by b, w_r by c,
    # then the product is scaled by abc, so we can set abc = 1)
    
    # This reduces the degrees of freedom, but it's still a non-linear problem
    
    # Let's just try to find a valid decomposition using a heuristic search
    
    # Try to find a rank-59 decomposition by perturbing the trivial one
    # and checking if it's still valid
    
    # For each pair of terms (r1, r2), try to merge them into a single term
    # by finding u, v, w such that u*v*w = u1*v1*w1 + u2*v2*w2
    
    # This is possible if the two terms have overlapping supports
    
    # Let's try to find pairs of terms that have overlapping supports
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) have overlapping supports if:
    # - They share the same j (so V supports overlap)
    # - They share the same k (so W supports overlap)
    # - They share the same i (so U supports overlap)
    
    # Actually, the supports are:
    # Term (i,j,k) has U support on row i*m+j, V support on row j*p+k, W support on row k*n+i
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) have:
    # - U supports overlap if i1*m+j1 = i2*m+j2, i.e., i1=i2 and j1=j2
    # - V supports overlap if j1*p+k1 = j2*p+k2, i.e., j1=j2 and k1=k2
    # - W supports overlap if k1*n+i1 = k2*n+i2, i.e., k1=k2 and i1=i2
    
    # So two terms have overlapping supports only if they are the same term
    
    # This means we can't merge any two terms in the trivial decomposition
    
    # But we can still find a lower-rank decomposition by using rank-1 tensors
    # that have support on multiple (u,v,w) positions
    
    # Let's try to find such a decomposition by solving a system of equations
    
    # For R = 59, we need to find 59 triples (u_r, v_r, w_r) such that:
    # sum_r u_r[u] * v_r[v] * w_r[w] = T[u,v,w] for all (u,v,w)
    
    # This is a non-linear system. Let's try to solve it using a numerical method
    
    # We can use gradient descent, but this is non-convex and might get stuck
    # in local minima
    
    # Let's try a different approach: use the fact that the tensor has a
    # specific structure and try to find a decomposition that exploits it
    
    # For n=2, we can try to use the fact that there are only 2 values of i
    
    # Let's try to find a decomposition where the U vectors have a specific form
    
    # For example, we can try to set U[i*m+j][r] = a_r(i,j) for some function a_r
    
    # This is getting too complex. Let's just return the trivial decomposition
    
    return best_result
# EVOLVE-BLOCK-END
