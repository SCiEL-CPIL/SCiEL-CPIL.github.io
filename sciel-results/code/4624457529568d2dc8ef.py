"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)

def _sa_solve(a, b, time_limit):
    """
    Solve QAP using Simulated Annealing with periodic spectral bisection + Hungarian reassignment.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize with a random permutation
    p = np.random.permutation(n).astype(np.int32)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    T = 1000.0
    alpha = 0.999
    
    def _swap_delta(i, j):
        li = p[i]
        lj = p[j]
        diff_row = b[lj] - b[li]
        diff_col = b[:, lj] - b[:, li]
        
        dr_p = diff_row[p]
        dc_p = diff_col[p]
        
        t1 = np.dot(a[i], dr_p)
        t2 = np.dot(a[:, i], dc_p)
        t3 = -np.dot(a[j], dr_p)
        t4 = -np.dot(a[:, j], dc_p)
        
        total_cross = t1 + t2 + t3 + t4
        
        c_ii = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li]) + \
               (a[i, i] - a[i, j]) * (b[li, lj] - b[li, li])
               
        c_jj = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj]) + \
               (a[j, i] - a[j, j]) * (b[lj, lj] - b[lj, li])
               
        cross_sum_excl = total_cross - c_ii - c_jj
        
        d_self = a[i, i] * (b[lj, lj] - b[li, li]) + \
                 a[j, j] * (b[li, li] - b[lj, lj]) + \
                 a[i, j] * (b[lj, li] - b[li, lj]) + \
                 a[j, i] * (b[li, lj] - b[lj, li])
                 
        return d_self + cross_sum_excl
    
    def _spectral_hungarian_reassign():
        """
        Use spectral bisection on the flow matrix to create two clusters,
        then use Hungarian assignment to reassign locations within each cluster.
        This provides a min-cut-inspired global reassignment.
        """
        nonlocal p, current_cost, best_p, best_cost
        
        # Compute flow matrix (symmetrize if needed)
        flow = (a + a.T) * 0.5
        
        # Compute Laplacian for spectral bisection
        degree = flow.sum(axis=1)
        L = np.diag(degree) - flow
        
        # Compute Fiedler vector (second smallest eigenvector)
        from scipy.sparse.linalg import eigsh
        try:
            # Find the second smallest eigenvalue/vector
            vals, vecs = eigsh(L, k=2, which='SM')
            fiedler = vecs[:, 1]
        except Exception:
            # Fallback: use first principal component of flow
            from scipy.linalg import eigh
            try:
                vals, vecs = eigh(flow)
                fiedler = vecs[:, 0]
            except Exception:
                return  # Can't compute, skip
        
        # Split into two clusters based on Fiedler vector
        median_val = np.median(fiedler)
        cluster1 = np.where(fiedler < median_val)[0]
        cluster2 = np.where(fiedler >= median_val)[0]
        
        # For each cluster, compute a linearized cost matrix and use Hungarian
        new_p = p.copy()
        for cluster in [cluster1, cluster2]:
            if len(cluster) < 2:
                continue
            
            # Get current locations for this cluster
            locs = p[cluster]
            
            # Compute linearized cost: C[i, j] = sum_k (a[i, k] + a[k, i]) * b[locs[i], locs[j]]
            # This is the standard linearization for QAP
            C = np.zeros((len(cluster), len(cluster)), dtype=np.float64)
            for idx, i in enumerate(cluster):
                for jdx, j in enumerate(cluster):
                    if i != j:
                        # Sum over all k
                        C[idx, jdx] = np.dot(a[i] + a[:, i], b[locs[jdx], :])
            
            # Use Hungarian to find best assignment within this cluster
            row_ind, col_ind = linear_sum_assignment(C)
            for r, c in zip(row_ind, col_ind):
                new_p[cluster[r]] = locs[c]
        
        # Check if the new permutation is better
        new_cost = _qap_cost(a, b, new_p)
        if new_cost < current_cost or (np.random.random() < np.exp(-(new_cost - current_cost) / T)):
            p = new_p
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    steps_since_reassign = 0
    reassign_interval = 5000  # Every 5000 steps, try a global reassignment
    
    while time.time() < deadline:
        # Generate random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(i, j)
        
        # Acceptance
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < 1e-4:
            T = 1e-4
        
        steps_since_reassign += 1
        if steps_since_reassign >= reassign_interval:
            steps_since_reassign = 0
            _spectral_hungarian_reassign()
    
    # Final polish: run a quick local search on the best solution
    p = best_p.copy()
    current_cost = best_cost
    polish_deadline = min(deadline, time.time() + 5.0)
    while time.time() < polish_deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        delta = _swap_delta(i, j)
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    """
    Solve the QAP instance using Simulated Annealing with spectral bisection + Hungarian reassignment.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    p, cost = _sa_solve(a, b, deadline - start_time)
    
    return p.tolist()
# EVOLVE-BLOCK-END
