"""Matrix multiplication 2,5,6 tensor decomposition with Strassen on all 2x2x2 sub-blocks."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Construct a low-rank exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    Strategy: Use Strassen's algorithm (rank 7) on all six disjoint 2x2x2 sub-blocks
    formed by j-pairs (0,1),(2,3) crossed with k-pairs (0,1),(2,3),(4,5).
    Remaining elements use schoolbook (rank 1 each).
    
    Base rank: 6*7 (Strassen) + (60 - 6*8) schoolbook = 42 + 12 = 54
    
    Then apply greedy set-cover to find linear combinations of terms that
    cancel multiple uncovered (i,j,k) entries simultaneously, reducing rank.
    """
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Standard Strassen rank-7 decomposition of the 2x2x2 matmul tensor.
    # Each entry is (U_flat, V_flat, W_flat) in flattened 2x2 order [00,01,10,11].
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # m1 = (a+d)(e+h)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # m2 = (c+d)e
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # m3 = a(f-h)
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]), # m4 = d(g-e)
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),   # m5 = (a+b)h
        ([1,0,-1,0], [1,1,0,0], [0,0,0,-1]), # m6 = (c-a)(e+f)
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]), # m7 = (b-d)(g+h)
    ]
    
    # Verify Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for U, V, W in strassen_terms:
                        total += U[i] * V[j] * W[k]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify_strassen()
    if not valid:
        # Fallback: schoolbook
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
    
    # Define the six disjoint 2x2x2 sub-blocks:
    # j-pairs: (0,1), (2,3)  [j=4 is left for schoolbook]
    # k-pairs: (0,1), (2,3), (4,5)
    # For each combination, we get a 2x2x2 sub-block over i in {0,1}, j in {j0,j0+1}, k in {k0,k0+1}
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Track which (i,j,k) triples are covered by Strassen
    covered = set()
    all_terms = []
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Mark all (i, j, k) in this block as covered
            for i in range(n):
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        covered.add((i, j, k))
            
            # Add 7 Strassen terms for this block
            for U_flat, V_flat, W_flat in strassen_terms:
                U_sparse = {}
                for idx, val in enumerate(U_flat):
                    if val != 0:
                        # idx: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
                        i_local = idx // 2
                        j_local = idx % 2
                        j_global = [j0, j1][j_local]
                        U_sparse[i_local * m + j_global] = val
                V_sparse = {}
                for idx, val in enumerate(V_flat):
                    if val != 0:
                        # idx: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
                        j_local = idx // 2
                        k_local = idx % 2
                        j_global = [j0, j1][j_local]
                        k_global = [k0, k1][k_local]
                        V_sparse[j_global * p + k_global] = val
                W_sparse = {}
                for idx, val in enumerate(W_flat):
                    if val != 0:
                        # idx: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
                        k_local = idx // 2
                        i_local = idx % 2
                        k_global = [k0, k1][k_local]
                        W_sparse[k_global * n + i_local] = val
                all_terms.append((U_sparse, V_sparse, W_sparse))
    
    # Add schoolbook terms for all (i,j,k) NOT covered by Strassen
    uncovered_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered_triples.append((i, j, k))
                    U_sparse = {i * m + j: 1}
                    V_sparse = {j * p + k: 1}
                    W_sparse = {k * n + i: 1}
                    all_terms.append((U_sparse, V_sparse, W_sparse))
    
    # Now apply greedy set-cover to try to reduce the number of schoolbook terms
    # by finding linear combinations that cover multiple uncovered triples.
    # 
    # Idea: Each schoolbook term for triple (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # A linear combination of schoolbook terms with coefficients c_{ijk} gives:
    #   U[i*m+j] += sum of c_{ijk} over all k' for fixed (i,j)
    #   V[j*p+k] += sum of c_{ijk} over all i' for fixed (j,k)  
    #   W[k*n+i] += sum of c_{ijk} over all j' for fixed (k,i)
    # 
    # For exact decomposition, we need the tensor to equal the matmul tensor.
    # The matmul tensor T[i,j,k] = delta (it's 1 if this is a valid matmul entry).
    # Actually the tensor is T[i,j,k] = 1 for all (i,j,k) in the matmul.
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
    #
    # The Strassen terms already handle the covered triples. For uncovered triples,
    # we need to add terms that contribute exactly 1 to each uncovered (i,j,k) and
    # 0 to all covered (i,j,k).
    #
    # Let's build the residual: for each (i,j,k), what's the current contribution?
    # Covered triples should have contribution 1 from Strassen.
    # Uncovered triples have contribution 0 currently.
    #
    # We need to find terms (U,V,W) such that for all (i,j,k):
    #   sum of new terms' contributions = 1 if (i,j,k) is uncovered, else 0.
    #
    # This is a tensor completion problem. Let's try a greedy approach:
    # Build candidate terms from pairs/triples of uncovered entries and check
    # if they can be expressed as rank-1 tensors that don't disturb covered entries.
    
    # Build a 3D array of the residual needed
    residual = np.zeros((n, m, p), dtype=float)
    for i, j, k in uncovered_triples:
        residual[i, j, k] = 1.0
    
    # Remove schoolbook terms from all_terms (they're at the end)
    num_strassen_terms = len(j_pairs) * len(k_pairs) * 7
    num_schoolbook = len(uncovered_triples)
    
    # Try to find a low-rank decomposition of the residual tensor
    # that vanishes on all covered (i,j,k) positions.
    
    # Approach: iterative rank-1 extraction with constraints
    # At each step, find a rank-1 tensor U*V*W that best matches the residual
    # subject to U*V*W = 0 on all covered positions.
    
    def find_rank1_with_zeros(residual, covered_mask, max_iter=200):
        """Find rank-1 tensor approximating residual, zero on covered positions."""
        # Initialize with SVD of the full residual (ignoring zeros)
        # Then iteratively project out the covered positions
        
        # Flatten and use alternating least squares
        # U: (n*m,), V: (m*p,), W: (p*n,)
        # Tensor T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
        
        # Initialize
        U = np.ones(n * m)
        V = np.ones(m * p)
        W = np.ones(p * n)
        
        # Normalize initial guess
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iter):
            # Update U: for each (i,j), U[i*m+j] = sum_{k} residual[i,j,k] * V[j*p+k] * W[k*n+i] / (V[j*p+k]*W[k*n+i])
            # But we need to handle the constraint that T=0 on covered positions.
            # Use only uncovered positions for the update.
            
            for i in range(n):
                for j in range(m):
                    num = 0.0
                    den = 0.0
                    for k in range(p):
                        if (i, j, k) not in covered:
                            num += residual[i, j, k] * V[j*p+k] * W[k*n+i]
                            den += V[j*p+k] * W[k*n+i]
                    if den != 0:
                        U[i*m+j] = num / den
                    else:
                        U[i*m+j] = 0.0
            
            # Update V
            for j in range(m):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        if (i, j, k) not in covered:
                            num += residual[i, j, k] * U[i*m+j] * W[k*n+i]
                            den += U[i*m+j] * W[k*n+i]
                    if den != 0:
                        V[j*p+k] = num / den
                    else:
                        V[j*p+k] = 0.0
            
            # Update W
            for k in range(p):
                for i in range(n):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        if (i, j, k) not in covered:
                            num += residual[i, j, k] * U[i*m+j] * V[j*p+k]
                            den += U[i*m+j] * V[j*p+k]
                    if den != 0:
                        W[k*n+i] = num / den
                    else:
                        W[k*n+i] = 0.0
            
            # Compute error on uncovered positions
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            pred = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            err += (residual[i, j, k] - pred) ** 2
            
            # Check constraint: T should be ~0 on covered positions
            constraint_violation = 0.0
            for (i, j, k) in covered:
                pred = U[i*m+j] * V[j*p+k] * W[k*n+i]
                constraint_violation += pred ** 2
            
            if err < best_err and constraint_violation < 1e-6:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
            
            if time.time() - start_time > DEADLINE:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try to decompose the residual using greedy rank-1 extraction
    # Remove the schoolbook terms and try to replace them with fewer rank-1 terms
    all_terms = all_terms[:num_strassen_terms]
    
    residual_current = residual.copy()
    additional_terms = []
    
    # Try up to 10 rank-1 extractions
    max_extractions = 10
    for extr in range(max_extractions):
        if time.time() - start_time > DEADLINE:
            break
        
        U, V, W, err = find_rank1_with_zeros(residual_current, covered, max_iter=100)
        
        if err < 1e-8:
            # Good fit, subtract it
            U_sparse = {idx: val for idx, val in enumerate(U) if abs(val) > 1e-10}
            V_sparse = {idx: val for idx, val in enumerate(V) if abs(val) > 1e-10}
            W_sparse = {idx: val for idx, val in enumerate(W) if abs(val) > 1e-10}
            additional_terms.append((U_sparse, V_sparse, W_sparse))
            
            # Subtract from residual
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            residual_current[i, j, k] -= U[i*m+j] * V[j*p+k] * W[k*n+i]
            
            # Check if residual is essentially zero
            if np.max(np.abs(residual_current)) < 1e-8:
                break
        else:
            # Couldn't find a good rank-1 term, use schoolbook for remaining
            break
    
    # If we still have residual, add schoolbook terms for remaining entries
    if np.max(np.abs(residual_current)) > 1e-8:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered and abs(residual_current[i, j, k]) > 1e-8:
                        # Add a schoolbook term scaled to match the residual
                        scale = residual_current[i, j, k]
                        U_sparse = {i * m + j: scale}
                        V_sparse = {j * p + k: 1.0}
                        W_sparse = {k * n + i: 1.0}
                        additional_terms.append((U_sparse, V_sparse, W_sparse))
                        residual_current[i, j, k] = 0.0
    
    all_terms.extend(additional_terms)
    
    rank = len(all_terms)
    
    # Check time
    if time.time() - start_time > DEADLINE:
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
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_sparse, V_sparse, W_sparse) in enumerate(all_terms):
        for idx, val in U_sparse.items():
            U[idx][col] = val
        for idx, val in V_sparse.items():
            V[idx][col] = val
        for idx, val in W_sparse.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
