"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]*B[j,k].
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c] should equal sum_j U[i*m+j,c]*V[j*p+k,c].
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,k): sum_j U3[i,j,:] * V3[j,k,:] should equal W3[k,i,:]
    for i in range(n):
        for k in range(p):
            lhs = np.zeros(rank)
            for j in range(m):
                lhs += U3[i, j, :] * V3[j, k, :]
            if not np.allclose(lhs, W3[k, i, :], atol=1e-9):
                return False
    return True

def _try_rank(R, n, m, p, seed=42, max_iter=5000, timeout=100):
    """Try to find a rank-R decomposition using structured search.
    
    Ansatz: Each rank-1 term c corresponds to a triple (i_c, j_c, k_c) with coefficients.
    We allow multiple triples to share the same rank index if their contributions are compatible.
    Start from schoolbook and try to merge terms.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initialize with schoolbook-like structure but with random coefficients
    # Each rank index c has a "type" (i, j, k) indicating which U,V,W rows it touches
    # U[i*m+j, c] = a_{i,j,c}, V[j*p+k, c] = b_{j,k,c}, W[k*n+i, c] = a_{i,j,c} * b_{j,k,c}
    
    # For a valid decomposition, for each (i,k):
    #   W[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # So W[k*n+i, c] must equal sum_j a_{i,j,c} * b_{j,k,c}
    
    # Strategy: assign each rank index a subset of (i,j) for U and (j,k) for V
    # Simplest: each rank index c is associated with one (i,j,k) triple
    # Then U[i*m+j,c]=1 if i_c=i and j_c=j else 0
    #          V[j*p+k,c]=1 if j_c=j and k_c=k else 0
    #          W[k*n+i,c]=1 if k_c=k and i_c=i else 0
    # This gives the schoolbook rank = n*m*p.
    
    # To reduce rank, we need to allow a single rank index to serve multiple (i,j,k) combos.
    # Key insight: if two triples (i1,j1,k1) and (i2,j2,k2) share the same i and k but different j,
    # they can potentially be merged if the j-index allows it.
    
    # Better approach: use a bipartite/graph coloring approach
    # Think of it as: we need to cover all (i,j,k) triples with rank indices
    # where each rank index c can handle a set of triples that are "compatible"
    
    # Compatibility: rank index c can handle triple (i,j,k) if we set
    #   U[i*m+j, c] = u_{i,j,c}, V[j*p+k, c] = v_{j,k,c}, W[k*n+i, c] = u_{i,j,c}*v_{j,k,c}
    # For a rank index to handle multiple triples (i_a, j_a, k_a), (i_b, j_b, k_b), ...
    # We need for each (i,k) pair: sum_j u_{i,j,c} * v_{j,k,c} = w_{i,k,c}
    
    # Simplest non-trivial structure: each rank index handles triples that form a "grid"
    # e.g., all (i, j, k) for fixed i, varying j and k, with u_{i,j,c}=alpha_j, v_{j,k,c}=beta_k
    # Then w_{i,k,c} = sum_j alpha_j * beta_k = beta_k * sum_j alpha_j
    # This only works if the pattern is separable.
    
    # Let's try a different approach: random search over structured decompositions
    # where each rank index is assigned a "pattern" of which (i,j) pairs and (j,k) pairs it activates.
    
    best_rank = R
    for trial in range(100):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization: assign each rank index a random subset of U-rows and V-rows
        U_mat = np.zeros((n*m, R))
        V_mat = np.zeros((m*p, R))
        W_mat = np.zeros((p*n, R))
        
        # Each rank index c: pick random i_c, then pick subset of j's, for each j pick subset of k's
        for c in range(R):
            # Pick a random i for this rank index
            i_c = rng.randint(n)
            # Pick random j's (subset of {0,...,m-1})
            js = set()
            for j in range(m):
                if rng.random() < 0.7:
                    js.add(j)
            if not js:
                js = {rng.randint(m)}
            # For each j in js, pick random k's
            for j in js:
                U_mat[i_c*m + j, c] = rng.choice([-1, 1])
                ks = set()
                for k in range(p):
                    if rng.random() < 0.5:
                        ks.add(k)
                if not ks:
                    ks = {rng.randint(p)}
                for k in ks:
                    V_mat[j*p + k, c] = rng.choice([-1, 1])
        
        # Compute W from U and V
        for c in range(R):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c]
                    W_mat[k*n+i, c] = s
        
        # Check if this is a valid decomposition
        # Actually, by construction W[k*n+i,c] = sum_j U[i*m+j,c]*V[j*p+k,c], so it's always valid!
        # The question is whether the overall tensor equation is satisfied.
        # The tensor equation is: for all (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
        # In the decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k' V[j'*p+k',c]*B[j',k'])
        # Wait, I need to be more careful about the tensor decomposition.
        
        # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k).
        # A decomposition of rank R means: T[i,j,k] = sum_{c=1}^R U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # So we need: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Let me check this condition:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list of lists with exact integer/half-integer values
            U_list = [[round(float(U_mat[r, c]), 6) for c in range(R)] for r in range(n*m)]
            V_list = [[round(float(V_mat[r, c]), 6) for c in range(R)] for r in range(m*p)]
            W_list = [[round(float(W_mat[r, c]), 6) for c in range(R)] for r in range(p*n)]
            
            # Verify with exact arithmetic
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 110s for safety
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also need at least max(n, m, p) = 8
    # Known: for 2x3x8, the rank is at most 48 (schoolbook)
    # Let's try from 24 upward
    
    for R in range(24, 49):
        if time.time() > deadline - 5:
            break
        result = _try_rank(R, n, m, p, seed=R, max_iter=5000, timeout=deadline - time.time() - 2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
