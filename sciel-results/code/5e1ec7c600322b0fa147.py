"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import itertools

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Build the target tensor set
    tensor_set = set()
    tensor_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_set.add((a, b, c))
                tensor_list.append((a, b, c))
    
    # Lower bound on rank
    lower_bound = max(nm, mp, pn)
    
    best_rank = n * m * p  # schoolbook = 63
    best_decomp = None
    
    # Try to find rank-r decompositions from lower_bound upward
    # Focus on a few promising ranks with more thorough search
    ranks_to_try = list(range(lower_bound, min(best_rank, lower_bound + 25)))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        decomp = _find_decomposition(tensor_set, tensor_list, nm, mp, pn, r, n, m, p, deadline - 10)
        if decomp is not None:
            best_rank = r
            best_decomp = decomp
            if r == lower_bound:
                break
    
    if best_decomp is not None:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(tensor_set, tensor_list, nm, mp, pn, r, n, m, p, deadline):
    """
    Try to find a rank-r CP decomposition using structured random search + polish.
    Uses block-structured initialization exploiting the 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1 structure.
    """
    num_restarts = 80
    max_iters = 500
    
    # Precompute support lists for fast ALS
    support_a = [[] for _ in range(nm)]
    support_b = [[] for _ in range(mp)]
    support_c = [[] for _ in range(pn)]
    for (a, b, c) in tensor_list:
        support_a[a].append((b, c))
        support_b[b].append((a, c))
        support_c[c].append((a, b))
    
    # Precompute the tensor as a sparse structure for fast error computation
    # tensor[a, b, c] = 1 if (a,b,c) in tensor_set else 0
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            return None
        
        rng = np.random.default_rng(seed=1000 + restart)
        
        # Structured initialization based on block decomposition
        # The 3x3x7 tensor can be seen as blocks over the p dimension
        # Block 0: k=0,1,2 -> 3x3x3
        # Block 1: k=3,4,5 -> 3x3x3  
        # Block 2: k=6 -> 3x3x1
        #
        # For each block, we can use structured patterns
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Strategy: Initialize using a mix of structured and random components
        # First, try to allocate some rank to each block structure
        
        if r >= 27:
            # Use schoolbook-like initialization for first 27 columns
            # This gives a valid starting point that we can then optimize
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m + j * p + k  # This isn't quite right for rank allocation
                        # Let's use a different approach
                        pass
        
        # Better: Use random initialization with some structure
        U = rng.standard_normal((nm, r))
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((pn, r))
        
        # Add some structured components to help convergence
        # For the first few columns, use patterns that respect the block structure
        num_structured = min(9, r)  # One per (i,j) pair
        for idx, (i, j) in enumerate(itertools.product(range(n), range(m))):
            if idx >= num_structured:
                break
            col = idx
            # U component
            U[i*m+j, col] += 2.0
            # V component - spread across k values
            for k in range(p):
                V[j*p+k, col] += 1.0 / p
            # W component
            for k in range(p):
                W[k*n+i, col] += 1.0 / p
        
        # Normalize columns
        for c in range(r):
            nu = np.linalg.norm(U[:, c])
            nv = np.linalg.norm(V[:, c])
            nw = np.linalg.norm(W[:, c])
            if nu > 1e-15:
                U[:, c] /= nu
            if nv > 1e-15:
                V[:, c] /= nv
            if nw > 1e-15:
                W[:, c] /= nw
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # ALS update U
            for a in range(nm):
                if not support_a[a]:
                    U[a, :] = 0
                    continue
                sa = support_a[a]
                if len(sa) < r:
                    # Underdetermined, use minimum norm solution
                    A = np.array([V[b, :] * W[c, :] for b, c in sa])
                    rhs = np.ones(len(sa))
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    except:
                        U[a, :] = 0
                else:
                    A = np.array([V[b, :] * W[c, :] for b, c in sa])
                    rhs = np.ones(len(sa))
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    except:
                        U[a, :] = 0
            
            # ALS update V
            for b in range(mp):
                if not support_b[b]:
                    V[b, :] = 0
                    continue
                sb = support_b[b]
                A = np.array([U[a, :] * W[c, :] for a, c in sb])
                rhs = np.ones(len(sb))
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                except:
                    V[b, :] = 0
            
            # ALS update W
            for c in range(pn):
                if not support_c[c]:
                    W[c, :] = 0
                    continue
                sc = support_c[c]
                A = np.array([U[a, :] * V[b, :] for a, b in sc])
                rhs = np.ones(len(sc))
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                except:
                    W[c, :] = 0
            
            # Compute error efficiently using vectorized operations
            err = 0.0
            for (a, b, c) in tensor_list:
                approx = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                err += (approx - 1.0) ** 2
            err = np.sqrt(err / len(tensor_list))
            
            if abs(prev_err - err) < 1e-13:
                break
            prev_err = err
        
        # Verify exactness: check all tensor entries
        max_err = 0.0
        for (a, b, c) in tensor_list:
            approx = float(np.dot(U[a, :], V[b, :] * W[c, :]))
            max_err = max(max_err, abs(approx - 1.0))
        
        if max_err > 1e-8:
            continue
        
        # Check non-tensor entries (sample more thoroughly)
        non_tensor_err = 0.0
        num_samples = 1000
        for _ in range(num_samples):
            a = int(rng.integers(0, nm))
            b = int(rng.integers(0, mp))
            c = int(rng.integers(0, pn))
            if (a, b, c) not in tensor_set:
                approx = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                non_tensor_err = max(non_tensor_err, abs(approx))
        
        if non_tensor_err > 1e-8:
            continue
        
        # Success! Convert to lists
        U_list = [[float(U[a, c]) for c in range(r)] for a in range(nm)]
        V_list = [[float(V[b, c]) for c in range(r)] for b in range(mp)]
        W_list = [[float(W[c, c2]) for c2 in range(r)] for c in range(pn)]
        return (U_list, V_list, W_list)
    
    return None
# EVOLVE-BLOCK-END
