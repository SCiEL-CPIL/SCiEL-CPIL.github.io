"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # Efficient: sum_i (a[i,:] @ b[p[i],:]) @ ones ... but let's do it properly
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (row_i_of_a @ b[p[i], :]) @ ones
    # Actually: for each i, sum_j a[i,j] * b[p[i], p[j]] = (a[i,:] * b[p[i], p]).sum()
    # Total = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Vectorized: sum over i of (a[i,:] @ b[p[i], p])
    # b[p[i], p] is row p[i] of b, indexed by p
    # So for each i: a[i,:] @ b[p[i], p]
    # This is a matrix-vector product per row... let's use einsum or direct
    # cost = np.einsum('ij,ik,jk->', a, b[p], b[p])  # wrong
    # Let me think: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # For fixed i: sum_j a[i,j] * b[p[i], p[j]] = dot(a[i,:], b[p[i], p])
    # So total = sum_i dot(a[i,:], b[p[i], p])
    # b[p, p] gives a matrix where (i,j) entry is b[p[i], p[j]]
    # So total = sum_{i,j} a[i,j] * b[p[i], p[j]] = sum(a * b[np.ix_(p, p)])
    return int(np.sum(a * b[np.ix_(p, p)]))


def _delta_swap(a, b, p, i, j):
    """
    Compute change in cost when swapping positions i and j in permutation p.
    p[i] and p[j] are the facilities at locations i and j.
    After swap: p[i] <-> p[j].
    
    Cost = sum_{m,n} a[m,n] * b[p[m], p[n]]
    
    The change involves terms where m or n is i or j.
    Let's compute delta = new_cost - old_cost.
    
    Terms affected: those with m in {i,j} or n in {i,j}.
    Specifically:
    - (i,i): a[i,i]*b[p[i],p[i]] -> a[i,i]*b[p[j],p[j]]
    - (i,j): a[i,j]*b[p[i],p[j]] -> a[i,j]*b[p[j],p[i]]
    - (j,i): a[j,i]*b[p[j],p[i]] -> a[j,i]*b[p[i],p[j]]
    - (j,j): a[j,j]*b[p[j],p[j]] -> a[j,j]*b[p[i],p[i]]
    - (m,i) for m not in {i,j}: a[m,i]*b[p[m],p[i]] -> a[m,i]*b[p[m],p[j]]
    - (m,j) for m not in {i,j}: a[m,j]*b[p[m],p[j]] -> a[m,j]*b[p[m],p[i]]
    - (i,m) for m not in {i,j}: a[i,m]*b[p[i],p[m]] -> a[i,m]*b[p[j],p[m]]
    - (j,m) for m not in {i,j}: a[j,m]*b[p[j],p[m]] -> a[j,m]*b[p[i],p[m]]
    
    Let pi = p[i], pj = p[j].
    
    delta = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,j]*(b[pj,pi]-b[pi,pj])
          + a[j,i]*(b[pi,pj]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
          + sum_{m not in {i,j}} a[m,i]*(b[p[m],pj]-b[p[m],pi])
          + sum_{m not in {i,j}} a[m,j]*(b[p[m],pi]-b[p[m],pj])
          + sum_{m not in {i,j}} a[i,m]*(b[pj,p[m]]-b[pi,p[m]])
          + sum_{m not in {i,j}} a[j,m]*(b[pi,p[m]]-b[pj,p[m]])
    
    Let's simplify. Note that b[pj,pi]-b[pi,pj] appears with coefficient a[i,j] - a[j,i]... no wait.
    
    Let me group differently. Define:
    For the "column i" terms (n=i): sum_{m != i} a[m,i]*b[p[m], p[i]]
    After swap, p[i] becomes pj, so this becomes sum_{m != i} a[m,i]*b[p[m], pj]
    But m=j is now at position i... wait no, we're swapping positions, so after swap:
    position i has facility pj, position j has facility pi.
    
    Let me just compute it numerically but efficiently.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct computation of delta using precomputed vectors would be ideal
    # For now, compute the affected terms
    # Old contributions involving position i or j:
    # We need: sum_{m,n} a[m,n]*b[p[m],p[n]] where m==i or n==i or m==j or n==j
    
    # Let's compute old and new for these terms
    # This is O(n) per swap if done carefully
    
    # Terms with m=i: sum_n a[i,n]*b[pi, p[n]]
    # Terms with n=i: sum_m a[m,i]*b[p[m], pi]
    # Terms with m=j: sum_n a[j,n]*b[pj, p[n]]
    # Terms with n=j: sum_m a[m,j]*b[p[m], pj]
    # But (i,i), (i,j), (j,i), (j,j) are counted twice in the above if we're not careful.
    # Actually each term (m,n) is counted once per condition it satisfies.
    # Let me just compute the full delta by summing over all m,n where m or n is in {i,j}
    
    # More efficient: precompute for each position k, the vector of contributions
    # But for simplicity in the delta, let's do O(n) computation
    
    # delta from position i changing from pi to pj:
    # For all n: a[i,n] changes from b[pi, p[n]] to b[pj, p[n]]
    # For all m: a[m,i] changes from b[p[m], pi] to b[p[m], pj]
    # Similarly for j.
    # But we double-count (i,j) and (j,i) and (i,i) and (j,j).
    
    # Let's use inclusion-exclusion or just compute carefully.
    
    # Contribution of (i,n) for all n: sum_n a[i,n]*b[pi,p[n]] -> sum_n a[i,n]*b[pj,p[n]]
    # Contribution of (m,i) for all m: sum_m a[m,i]*b[p[m],pi] -> sum_m a[m,i]*b[p[m],pj]
    # Contribution of (j,n) for all n: sum_n a[j,n]*b[pj,p[n]] -> sum_n a[j,n]*b[pi,p[n]]
    # Contribution of (m,j) for all m: sum_m a[m,j]*b[p[m],pj] -> sum_m a[m,j]*b[p[m],pi]
    
    # But (i,i) appears in both first and second, (i,j) in first and fourth, etc.
    # Let me just compute the delta as:
    
    # new - old for the four "corner" terms plus the cross terms.
    
    # I'll use a precomputed approach: maintain for each position k:
    # col_k[m] = a[m,k] (column k of a)
    # row_k[n] = a[k,n] (row k of a)
    # And b is fixed.
    
    # For swap i<->j:
    # delta = sum_n a[i,n]*(b[pj,p[n]]-b[pi,p[n]]) + sum_m a[m,i]*(b[p[m],pj]-b[p[m],pi])
    #       + sum_n a[j,n]*(b[pi,p[n]]-b[pj,p[n]]) + sum_m a[m,j]*(b[p[m],pi]-b[p[m],pj])
    #       - (corrections for double-counted terms)
    
    # Actually this double counts. Let me think again.
    
    # The total cost is sum_{m,n} a[m,n]*b[p[m],p[n]].
    # When we swap p[i] and p[j], only terms where m==i, n==i, m==j, or n==j change.
    # Let S = {i,j}. The affected terms are {(m,n) : m in S or n in S}.
    
    # Let me just compute old and new for these terms directly. O(n) each.
    
    # Old:
    # sum_{n} a[i,n]*b[pi,p[n]] + sum_{m != i} a[m,i]*b[p[m],pi]  # row i + col i excluding (i,i)
    # + sum_{n} a[j,n]*b[pj,p[n]] + sum_{m != j} a[m,j]*b[p[m],pj]  # row j + col j excluding (j,j)
    # But (i,j) is in row i and col j, (j,i) is in row j and col i.
    # This is getting messy. Let me just do it with numpy slices.
    
    # Simpler: compute the cost contribution of positions in {i,j} directly.
    # pos_i_old = sum_n a[i,n]*b[pi,p[n]] + sum_{m!=i} a[m,i]*b[p[m],pi]
    # This equals: row_i_contrib + col_i_contrib - a[i,i]*b[pi,pi] (since (i,i) counted twice)
    
    # Actually, let me just compute delta numerically with O(n) work:
    
    # For position i (facility pi -> pj):
    # delta_i = sum_n a[i,n]*(b[pj,p[n]] - b[pi,p[n]]) + sum_{m!=i} a[m,i]*(b[p[m],pj] - b[p[m],pi])
    # For position j (facility pj -> pi):
    # delta_j = sum_n a[j,n]*(b[pi,p[n]] - b[pj,p[n]]) + sum_{m!=j} a[m,j]*(b[p[m],pi] - b[p[m],pj])
    
    # But this double-counts (i,j), (j,i), (i,i), (j,j).
    # Specifically:
    # delta_i includes (i,j) term: a[i,j]*(b[pj,p[j]] - b[pi,p[j]]) = a[i,j]*(b[pj,pj]-b[pi,pj])
    # delta_j includes (j,i) term: a[j,i]*(b[pi,p[i]] - b[pj,p[i]]) = a[j,i]*(b[pi,pi]-b[pj,pi])
    # delta_i includes (i,i) term: a[i,i]*(b[pj,pi]-b[pi,pi])  [wait, p[n] when n=i is p[i]=pi]
    #   = a[i,i]*(b[pj,pi]-b[pi,pi])
    # delta_j includes (j,j) term: a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # Also delta_i includes (m,i) for m=j: a[j,i]*(b[p[j],pj]-b[p[j],pi]) = a[j,i]*(b[pj,pj]-b[pj,pi])
    # And delta_j includes (m,j) for m=i: a[i,j]*(b[p[i],pi]-b[p[i],pj]) = a[i,j]*(b[pi,pi]-b[pi,pj])
    
    # This is getting complicated. Let me just compute the delta by direct O(n) summation
    # of the affected terms, being careful about double counting.
    
    # Affected terms: all (m,n) where m in {i,j} or n in {i,j}
    # = (row i) union (col i) union (row j) union (col j)
    
    # Let me compute:
    # old_affected = sum_{n} a[i,n]*b[pi,p[n]] + sum_{n} a[j,n]*b[pj,p[n]] 
    #               + sum_{m not in {i,j}} a[m,i]*b[p[m],pi] + sum_{m not in {i,j}} a[m,j]*b[p[m],pj]
    # This counts (i,i) once in row i, (i,j) in row i and col j... no.
    # Row i: (i,n) for all n. Includes (i,i) and (i,j).
    # Row j: (j,n) for all n. Includes (j,i) and (j,j).
    # Col i (m not in {i,j}): (m,i) for m not i,j.
    # Col j (m not in {i,j}): (m,j) for m not i,j.
    # So (i,i) counted once, (i,j) counted once (in row i), (j,i) counted once (in row j), (j,j) counted once.
    # And (m,i) for m not in {i,j} counted once, (m,j) for m not in {i,j} counted once.
    # Total: all affected terms exactly once. 
    
    # New:
    # new_affected = sum_{n} a[i,n]*b[pj,p[n]] + sum_{n} a[j,n]*b[pi,p[n]]
    #               + sum_{m not in {i,j}} a[m,i]*b[p[m],pj] + sum_{m not in {i,j}} a[m,j]*b[p[m],pi]
    
    # delta = new_affected - old_affected
    
    # This is O(n) per swap. Let's implement it.
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    old_affected = (
        np.dot(a[i, :], b[pi, p]) +
        np.dot(a[j, :], b[pj, p]) +
        np.dot(a[mask, i], b[p[mask], pi]) +
        np.dot(a[mask, j], b[p[mask], pj])
    )
    
    new_affected = (
        np.dot(a[i, :], b[pj, p]) +
        np.dot(a[j, :], b[pi, p]) +
        np.dot(a[mask, i], b[p[mask], pj]) +
        np.dot(a[mask, j], b[p[mask], pi])
    )
    
    return new_affected - old_affected


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Phase 1: Hungarian initialization
    # The QAP can be decomposed into two assignment problems as an approximation.
    # We use the row sums to create a proxy cost matrix for the Hungarian algorithm.
    # Actually, a better approach: use the fact that QAP ~ min_p sum_{i,j} a[i,j]*b[p[i],p[j]]
    # This is not a standard assignment, but we can use a linearization.
    # For now, use rank-matching as initial, then improve with SA.
    
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Simulated Annealing with O(n) delta swaps
    if time.time() >= deadline:
        return best_p.tolist()
    
    # Precompute row and column sums of a for faster delta computation
    # Actually, let's just use the delta_swap function above
    
    # SA parameters
    T = 1000.0
    T_min = 1.0
    cooling = 0.9999  # will adjust based on time
    rng = np.random.default_rng(42)
    
    # Estimate how many iterations we can do
    # Each iteration is O(n) = O(150) operations, very fast
    # Let's just run until deadline
    
    while time.time() < deadline:
        # Pick two random positions to swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        delta = _delta_swap(a, b, p, i, j)
        
        # Accept or reject
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
