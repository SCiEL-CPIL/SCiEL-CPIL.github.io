"""Matrix multiplication 3x4x4 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    # Build the matrix multiplication tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school
    
    def verify(U, V, W):
        """Verify decomposition reconstructs T exactly."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_recon, T, atol=1e-10)
    
    # Try to reduce rank by finding linear combinations
    # Strategy: for each pair of schoolbook terms, check if their sum can be
    # expressed as a single rank-1 tensor (which would allow merging)
    
    # More systematic: try rank r from 47 down to some lower bound
    # Lower bound: max(n*p, m*max(n,p)) = max(12, 16) = 16
    # But realistically, for this size, rank is likely close to 48
    
    # Try a greedy approach: start with schoolbook, try to eliminate terms
    # by finding if one term can be expressed as a combination of others
    
    # Alternative: use the fact that for each fixed (i,k), the j-dimension
    # gives a 4-dimensional bilinear form. We can try to find a shared
    # decomposition across all (i,k) pairs.
    
    # Let's try: for each j, we have terms for all (i,k). Can we group
    # terms by j and find a lower-rank decomposition per j-group?
    
    # Actually, let's try a randomized search for lower rank decompositions
    # using a least-squares approach
    
    def try_rank(r, max_iters=100):
        """Try to find a rank-r decomposition via iterative optimization."""
        if time.time() > deadline - 10:
            return None, None, None
        
        # Initialize with random sparse vectors
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Initialize by sampling from schoolbook
        for col in range(r):
            # Pick a random (i,j,k) and use its pattern
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
        
        # Iterative refinement: for each column, try to fit it better
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for col in range(r):
                # Compute residual after removing this column
                T_recon_without = np.zeros((nm, mp, pn), dtype=np.float64)
                for other_col in range(r):
                    if other_col != col:
                        T_recon_without += np.einsum('a,b,c->abc', U[:, other_col], V[:, other_col], W[:, other_col])
                
                residual = T - T_recon_without
                
                # Try to fit residual with a rank-1 tensor
                # This is a hard problem, so use a heuristic
                # Find the best (a,b,c) triple to add
                best_score = -np.inf
                best_a, best_b, best_c = None, None, None
                
                # Sample candidate positions
                for _ in range(50):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    # Score: how well does u[a]*v[b]*w[c] fit the residual
                    # We want to maximize the fit
                    score = residual[a, b, c]
                    if score > best_score:
                        best_score = score
                        best_a, best_b, best_c = a, b, c
                
                if best_a is not None:
                    # Set the new column
                    U[:, col] = 0
                    V[:, col] = 0
                    W[:, col] = 0
                    U[best_a, col] = 1
                    V[best_b, col] = 1
                    W[best_c, col] = 1
                    improved = True
            
            if not improved:
                break
        
        # Verify
        if verify(U, V, W):
            return U, V, W
        
        return None, None, None
    
    # Try ranks from 47 down
    for r in range(rank_school - 1, max(16, rank_school - 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random restarts
        for attempt in range(5):
            np.random.seed(attempt * 1000 + r)
            U, V, W = try_rank(r, max_iters=50)
            if U is not None:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
