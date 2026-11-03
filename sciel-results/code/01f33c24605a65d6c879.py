"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _laplacian_eigvecs(mat, k):
    """Compute the k smallest non-trivial eigenvectors of the graph Laplacian of mat."""
    n = mat.shape[0]
    # Use absolute values to ensure non-negative weights for Laplacian
    w = np.abs(mat).astype(np.float64)
    # Degree matrix
    deg = w.sum(axis=1)
    # Laplacian L = D - W
    # We want eigenvectors of L, but for large n we use a few smallest eigenvectors
    # Use scipy if available, else numpy
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L = csr_matrix(np.diag(deg) - w)
        # Find k smallest eigenvalues (excluding the trivial 0 eigenvalue)
        # Use shift-invert for smallest eigenvalues
        if k < n - 1:
            vals, vecs = eigsh(L, k=k + 1, sigma=0, which='LM')
        else:
            vals, vecs = np.linalg.eigh(np.diag(deg) - w)
            vecs = vecs[:, :k + 1]
        # Drop the first (constant) eigenvector
        vecs = vecs[:, 1:]
        return vecs
    except Exception:
        # Fallback: use numpy
        L = np.diag(deg) - w
        vals, vecs = np.linalg.eigh(L)
        vecs = vecs[:, 1:k + 1]
        return vecs


def _spectral_init(a, b, n, k=10):
    """Initialize assignment using spectral embedding of both graphs."""
    try:
        # Get spectral embeddings
        Va = _laplacian_eigvecs(a, k)
        Vb = _laplacian_eigvecs(b, k)
        
        # Normalize rows to unit length for meaningful distance comparison
        norm_a = np.linalg.norm(Va, axis=1, keepdims=True)
        norm_b = np.linalg.norm(Vb, axis=1, keepdims=True)
        norm_a[norm_a == 0] = 1
        norm_b[norm_b == 0] = 1
        Va = Va / norm_a
        Vb = Vb / norm_b
        
        # For each facility, find the best matching location by minimizing
        # the sum of weighted distances in the spectral space
        # Use a simple greedy assignment: sort facilities by "centrality" in spectral space
        # and assign to nearest unassigned location
        
        # Compute pairwise distances in spectral space
        # Va: (n, k), Vb: (n, k)
        # dist[i, j] = ||Va[i] - Vb[j]||^2
        # = ||Va[i]||^2 + ||Vb[j]||^2 - 2 Va[i] . Vb[j]
        # Since normalized, = 2 - 2 Va[i].Vb[j]
        
        # Compute similarity matrix
        sim = Va @ Vb.T  # (n, n)
        
        # Greedy assignment: for each facility, pick the most similar unassigned location
        p = np.full(n, -1, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        
        # Process facilities in order of decreasing "centrality" (norm in original space)
        fac_order = np.argsort(-np.abs(a).sum(axis=1), kind='stable')
        
        for i in fac_order:
            # Find best unassigned location
            scores = sim[i].copy()
            scores[used] = -np.inf
            j = np.argmax(scores)
            p[i] = j
            used[j] = True
        
        return p
    except Exception:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def _rank_init(a, b, n):
    """Rank-matching initialization."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _calc_cost(a, b, p):
    """Calculate QAP cost."""
    return np.sum(a * b[np.ix_(p, p)])


def _swap_delta(a, b, p, i, j):
    """Compute delta for swapping facilities i and j."""
    # Current: cost includes a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] + 
    #   sum_{k!=i,j} a[i,k]*b[p[i],p[k]] + a[j,k]*b[p[j],p[k]]
    # After swap: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]] +
    #   sum_{k!=i,j} a[i,k]*b[p[j],p[k]] + a[j,k]*b[p[i],p[k]]
    
    # delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    #        + a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
    # Since b is symmetric: b[p[j],p[i]] = b[p[i],p[j]], so the cross terms cancel
    # delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    
    dA = a[i, :] - a[j, :]
    dB = b[p[j], :] - b[p[i], :]
    dB[i] = 0
    dB[j] = 0
    return np.dot(dA, dB)


def _local_search(a, b, p, max_time, deadline):
    """First-improvement 2-opt local search."""
    n = len(p)
    p = p.copy()
    cost = _calc_cost(a, b, p)
    
    improved = True
    while improved and (time.time() - max_time) < deadline:
        improved = False
        for i in range(n):
            if (time.time() - max_time) > deadline:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    if (time.time() - max_time) > deadline:
                        return p, cost
    return p, cost


def _anneal(a, b, p, T_start, T_end, alpha, max_time, deadline):
    """Simulated annealing."""
    n = len(p)
    p = p.copy()
    cost = _calc_cost(a, b, p)
    T = T_start
    
    while T > T_end and (time.time() - max_time) < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            T *= alpha
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            accept = True
        else:
            exp_arg = -delta / T
            if exp_arg > -50:
                accept = np.random.rand() < np.exp(exp_arg)
            else:
                accept = False
        
        if accept:
            p[i], p[j] = p[j], p[i]
            cost += delta
        
        T *= alpha
    
    return p, cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 140.0  # Safety margin: evaluator kills at 150s
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Rank matching
    inits.append(_rank_init(a, b, n))
    
    # 2. Spectral with different k values
    for k in [5, 10, 20]:
        try:
            if (time.time() - start_time) < deadline - 60:
                p_spec = _spectral_init(a, b, n, k=k)
                inits.append(p_spec)
        except Exception:
            pass
    
    # 3. Random initialization
    inits.append(np.random.permutation(n).astype(np.int64))
    
    best_p = None
    best_cost = np.inf
    
    for init_p in inits:
        if (time.time() - start_time) > deadline - 10:
            break
        
        # Local search
        p_ls, cost_ls = _local_search(a, b, init_p, start_time, deadline - 20)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Simulated annealing from the local search result
        if (time.time() - start_time) < deadline - 15:
            p_sa, cost_sa = _anneal(a, b, p_ls, 5000.0, 1.0, 0.995, start_time, deadline - 10)
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
            
            # Final polish
            if (time.time() - start_time) < deadline - 5:
                p_final, cost_final = _local_search(a, b, p_sa, start_time, deadline - 3)
                if cost_final < best_cost:
                    best_cost = cost_final
                    best_p = p_final.copy()
    
    if best_p is None:
        best_p = inits[0]
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
