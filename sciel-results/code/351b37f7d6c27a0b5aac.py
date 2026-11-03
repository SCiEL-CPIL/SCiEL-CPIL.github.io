"""QAP solver for tho150 using local search with efficient swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 140  # 140s safety margin
    
    # Precompute row sums of a for initial heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Initial solution: rank matching
    heavy = np.argsort(-a_row_sums, kind='stable')
    central = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    # Compute initial cost
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i],p[j]]
        # Using matrix multiplication trick: 
        # For each i, sum_j a[i,j] * b[p[i],p[j]]
        # = sum_i sum_j a[i,j] * b[p[i],p[j]]
        # We can compute this as: for each location l, the facilities assigned to l
        # Actually, let's just do it directly with numpy
        # cost = sum over all i,j of a[i,j] * b[p[i],p[j]]
        # This can be written as: sum_i (a[i,:] @ b[p[i],:]).sum()
        # But more efficiently: 
        # Let M = b[p[:,None], p[None,:]]  # n x n matrix where M[i,j] = b[p[i],p[j]]
        # Then cost = sum(a * M)
        M = b[np.ix_(p, p)]
        return np.sum(a * M)
    
    best_cost = compute_cost(p)
    best_p = p.copy()
    
    # Precompute for efficient swap delta
    # When swapping p[i] and p[j], the cost change is:
    # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + 
    #         sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) +
    #         sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) +
    #         sum_k a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    # But since a is symmetric (flow matrix), a[i,k]=a[k,i], so:
    # delta = 2 * sum_k (a[i,k] + a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]]) for k != i,j
    # Plus the i,j pair terms:
    # The terms involving both i and j: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] 
    # After swap: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    # Since b is symmetric, these are equal, so no change from i,j pair.
    # Wait, let me be more careful.
    
    # Let me define: cost = sum_{i,j} a[i,j] * b[p[i],p[j]]
    # After swapping p[i] and p[j], call new permutation p'
    # p'[i] = p[j], p'[j] = p[i], p'[k] = p[k] for k != i,j
    # 
    # delta = cost(p') - cost(p)
    # = sum_{i,j} a[i,j]*(b[p'[i],p'[j]] - b[p[i],p[j]])
    # Only terms where i or j is in {i,j} change.
    # 
    # Terms with first index i: sum_j a[i,j]*(b[p'[i],p'[j]] - b[p[i],p[j]])
    # = sum_j a[i,j]*(b[p[j],p'[j]] - b[p[i],p[j]])
    # For j != i: p'[j] = p[j] if j != i, j != j... wait j is the second index
    # p'[j] = p[j] for j not in {i,j}, p'[i]=p[j], p'[j]=p[i]
    # So for first index i:
    #   j=i: a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]])  [since p'[i]=p[j], p'[i]=p[j]]
    #        = a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]])
    #   j=j: a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
    #   j=k (k!=i,j): a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    # 
    # Similarly for first index j:
    #   j=i: a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    #   j=j: a[j,j]*(b[p[i],p[i]] - b[p[j],p[j]])
    #   j=k: a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    
    # This is getting complex. Let me just compute it directly for simplicity,
    # but use numpy vectorization.
    
    def swap_delta(p, i, j):
        """Compute cost change when swapping positions i and j in permutation p."""
        pi = p[i]
        pj = p[j]
        
        # Vectorized computation
        # For all k != i, j:
        # contribution from a[i,k] and a[k,i]: (a[i,k] + a[k,i]) * (b[pj,p[k]] - b[pi,p[k]])
        # contribution from a[j,k] and a[k,j]: (a[j,k] + a[k,j]) * (b[pi,p[k]] - b[pj,p[k]])
        
        # Create mask excluding i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) == 0:
            # Only two elements, handle separately
            return (a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi]) - \
                   (a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj])
        
        pk = p[ks]
        
        # Terms involving i (as first or second index)
        # a[i,k]*(b[pj,pk] - b[pi,pk]) + a[k,i]*(b[pk,pj] - b[pk,pi])
        term_i = a[i, ks] * (b[pj, pk] - b[pi, pk]) + a[ks, i] * (b[pk, pj] - b[pk, pi])
        
        # Terms involving j
        term_j = a[j, ks] * (b[pi, pk] - b[pj, pk]) + a[ks, j] * (b[pk, pi] - b[pk, pj])
        
        # Terms where both i and j are involved (i,j pair and j,i pair)
        # a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
        # Since b is symmetric: b[pj,pi] = b[pi,pj], so this is 0
        # But let's include for correctness:
        term_ij = a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
        
        # Diagonal terms: a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
        term_diag = a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
        
        return np.sum(term_i) + np.sum(term_j) + term_ij + term_diag
    
    # Local search: try all swaps, accept best improving move
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Try all pairs
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                d = swap_delta(p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_i >= 0:
            # Apply swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += best_delta
            improved = True
            best_p = p.copy()
    
    # Final check
    final_cost = compute_cost(p)
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
