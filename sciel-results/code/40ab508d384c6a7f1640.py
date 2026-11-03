"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _qap_cost(a, b, p):
    """Compute the QAP cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _rank_matching_initial(a, b):
    """Generate initial permutation via rank matching."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _linearized_cost_matrix(a, b, p):
    """
    Compute the linearized cost matrix C where C[i, j] approximates
    the marginal cost of assigning facility i to location j.
    C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    This is the gradient of the QAP objective with respect to the assignment.
    """
    n = len(a)
    # For each facility i and location j:
    # cost[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    # = (a @ b[:, p])[i, j] + (a.T @ b[p, :])[i, j]
    # Note: b[:, p] means column j of b[:, p] is b[:, p[j]]... wait.
    # b[j, p[k]] means we take column p[k] of b and look at row j.
    # So sum_k a[i, k] * b[j, p[k]] = sum_k a[i, k] * (b[:, p[k]])[j]
    # Let B_perm = b[:, p] (n x n matrix where column k is b[:, p[k]])
    # Then (a @ B_perm)[i, j] = sum_k a[i, k] * B_perm[k, j] = sum_k a[i, k] * b[j, p[k]]
    # Wait, that's not right. B_perm[k, j] = b[:, p[k]][j]... no.
    # b[:, p] creates a matrix where the k-th column is b[:, p[k]].
    # So B_perm[k, j] doesn't make sense in that indexing. Let me be more careful.
    
    # b[j, p[k]]: for fixed j and k, this is element (j, p[k]) of b.
    # sum_k a[i, k] * b[j, p[k]]
    # Let's define M1[i, j] = sum_k a[i, k] * b[j, p[k]]
    # = sum_k a[i, k] * (b.T[p[k], :])[j] ... no
    # b[j, p[k]] = (b.T)[p[k], j]
    # So M1[i, j] = sum_k a[i, k] * b.T[p[k], j]
    # Let BT_p = b.T[p, :] (n x n matrix where row k is b.T[p[k], :] = b[:, p[k]].T)
    # Then M1 = a @ BT_p
    
    # Similarly, M2[i, j] = sum_k a[k, i] * b[p[k], j]
    # a[k, i] = a.T[i, k]
    # b[p[k], j] = (b[p, :])[k, j]
    # So M2[i, j] = sum_k a.T[i, k] * b[p, :][k, j] = (a.T @ b[p, :])[i, j]
    
    BT_p = b.T[np.ix_(p, range(n))]  # This is b.T[p, :] which is b[:, p].T
    # Actually, b.T[p, :] selects rows p from b.T, giving an n x n matrix where row k is b.T[p[k], :]
    # b[p, :] selects rows p from b, giving an n x n matrix where row k is b[p[k], :]
    
    M1 = a @ BT_p  # M1[i, j] = sum_k a[i, k] * b[j, p[k]]
    M2 = a.T @ b[np.ix_(p, range(n))]  # M2[i, j] = sum_k a[k, i] * b[p[k], j]
    
    return M1 + M2


def _hungarian_solve(cost_matrix):
    """Solve the assignment problem using the Hungarian algorithm."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(len(cost_matrix), dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _two_opt_delta(a, b, p, i, j):
    """
    Compute the cost change from swapping positions i and j in permutation p.
    p[i] and p[j] are the locations currently assigned to facilities i and j.
    After swap: facility i goes to p[j], facility j goes to p[i].
    
    Delta = new_cost - old_cost
    """
    # Let u = p[i], v = p[j]
    # Old contributions involving i or j:
    # sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u] + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,j]*b[p[k],v]
    # New contributions:
    # sum_k a[i,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],v] + sum_k a[j,k]*b[u,p[k]] + sum_k a[k,j]*b[p[k],u]
    # Note: the term a[i,j]*b[u,v] and a[j,i]*b[v,u] are included in both sums.
    # Let's compute the delta directly.
    
    u = p[i]
    v = p[j]
    
    # Delta for terms involving facility i:
    # Old: sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u]
    # New: sum_k a[i,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],v]
    # Delta_i = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[k,i]*(b[p[k],v] - b[p[k],u])
    
    # Delta for terms involving facility j:
    # Old: sum_k a[j,k]*b[v,p[k]] + sum_k a[k,j]*b[p[k],v]
    # New: sum_k a[j,k]*b[u,p[k]] + sum_k a[k,j]*b[p[k],u]
    # Delta_j = sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]]) + sum_k a[k,j]*(b[p[k],u] - b[p[k],v])
    
    # Note that Delta_i + Delta_j:
    # = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[k,i]*(b[p[k],v] - b[p[k],u])
    #   + sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]]) + sum_k a[k,j]*(b[p[k],u] - b[p[k],v])
    # = sum_k (a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]]) + sum_k (a[k,i] - a[k,j])*(b[p,k,v] - b[p[k],u])
    
    # But we must be careful: when k=i or k=j, the terms need special handling because
    # p[k] changes for k=i and k=j.
    
    # Let's do it more carefully by considering all pairs.
    # The total cost is sum_{i',j'} a[i',j']*b[p[i'],p[j']]
    # Only terms where i'=i or j'=i or i'=j or j'=j change.
    
    # Terms with i'=i: sum_{j'} a[i,j']*b[u,p[j']] -> sum_{j'} a[i,j']*b[v,p[j']]
    # But p[j'] changes if j'=i or j'=j.
    # If j' != i and j' != j: p[j'] stays the same.
    # If j' = i: b[u,p[i]] = b[u,u] -> b[v,p[i]] = b[v,v] (since p[i] becomes v)
    # If j' = j: b[u,p[j]] = b[u,v] -> b[v,p[j]] = b[v,u] (since p[j] becomes u)
    
    # This is getting complex. Let's use a simpler approach:
    # Compute the delta by considering the affected terms explicitly.
    
    # Affected terms are those where at least one of the facility indices is i or j,
    # OR at least one of the location indices is u or v (but locations are determined by facilities).
    
    # Actually, the cleanest way:
    # Old cost contribution from pairs involving facility i or j:
    # C_old = sum_{k} a[i,k]*b[u,p[k]] + sum_{k} a[k,i]*b[p[k],u] + sum_{k} a[j,k]*b[v,p[k]] + sum_{k} a[k,j]*b[p[k],v]
    # But this double-counts a[i,j]*b[u,v] and a[j,i]*b[v,u].
    
    # Let's just compute the full delta by iterating over all k for the relevant terms.
    
    # For facility i:
    # Old: sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u]
    # New: sum_k a[i,k]*b[v,p'[k]] + sum_k a[k,i]*b[p'[k],v] where p' is p with i,j swapped
    
    # For k not in {i,j}: p'[k] = p[k]
    # For k = i: p'[i] = v
    # For k = j: p'[j] = u
    
    # This is still complex. Let me use a different approach: compute the cost before and after
    # for the affected sub-matrix.
    
    # The change only affects terms where at least one facility is i or j.
    # Let's compute:
    # delta = (new terms with i or j) - (old terms with i or j)
    
    # Old terms with facility i:
    # sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u]
    # = a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
    #   + a[i,i]*b[u,u] + a[j,i]*b[v,u] + sum_{k!=i,j} a[k,i]*b[p[k],u]
    # Wait, I'm confusing myself. Let me just do it numerically.
    
    # Since n=256, and we're doing this for candidate swaps, let's compute efficiently.
    
    # Precompute: for the current p, let's compute the contribution of facility i to the total cost.
    # contrib_i = sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u]
    # Similarly for j.
    
    # But the cross terms a[i,j]*b[u,v] and a[j,i]*b[v,u] appear in both contrib_i and contrib_j.
    
    # Let's define:
    # S_i = sum_k a[i,k]*b[u,p[k]]  (row i of a dotted with row u of b permuted by p)
    # T_i = sum_k a[k,i]*b[p[k],u]  (col i of a dotted with col u of b permuted by p)
    # contrib_i = S_i + T_i
    
    # Similarly S_j, T_j.
    
    # Total old contribution from i and j (avoiding double count):
    # old_ij = S_i + T_i + S_j + T_j - a[i,j]*b[u,v] - a[j,i]*b[v,u]
    # (subtract the cross terms that appear in both)
    
    # After swap, u and v are exchanged:
    # S_i_new = sum_k a[i,k]*b[v,p'[k]] where p'[i]=v, p'[j]=u, p'[k]=p[k] otherwise
    # = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
    
    # This is getting too complex for an analytical delta. Let me just compute it by
    # temporarily swapping and computing the difference in a vectorized way.
    
    # Actually, for a 2-opt swap, the delta can be computed as:
    # delta = sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
    # with corrections for k=i and k=j.
    
    # Let me just do it the brute force way for correctness, since n=256 is small enough
    # for the inner loop to be fast if we vectorize.
    
    # Actually, let me just compute the cost difference directly.
    # We'll compute the change by considering all pairs (i', j') where i' in {i,j} or j' in {i,j}.
    
    # There are at most 4*256 = 1024 such pairs, which is fine.
    
    # But actually, a simpler approach: just compute the cost before and after for the
    # affected facilities.
    
    # Let's use the fact that:
    # cost = sum_{i',j'} a[i',j'] * b[p[i'],p[j']]
    
    # The change from swapping p[i] and p[j] affects all terms where i'=i or i'=j or j'=i or j'=j.
    # That's at most 2*n - 1 terms (with overlap).
    
    # For a given swap (i, j) with u=p[i], v=p[j]:
    # Old contribution: sum_{i' in {i,j}} sum_{j'} a[i',j']*b[p[i'],p[j']] + sum_{j' in {i,j}} sum_{i' not in {i,j}} a[i',j']*b[p[i'],p[j']]
    # This is getting messy. Let me just do a direct computation.
    
    # For n=256, a full cost computation is O(n^2) = 65536 operations.
    # If we do this for each candidate swap in a 2-opt pass, that's too slow.
    
    # Let me use the analytical formula. After careful derivation:
    # delta = (a[i,:]-a[j,:]) @ (b[v,p]-b[u,p]) + (a[:,i]-a[:,j]) @ (b[p,v]-b[p,u])
    # where b[v,p] means the vector with components b[v,p[k]] for k=0..n-1
    # and b[p,v] means the vector with components b[p[k],v] for k=0..n-1
    
    # But we need to correct for the fact that p[i] and p[j] change.
    # When k=i: b[v,p[i]] becomes b[v,v] and b[u,p[i]] becomes b[u,u]
    # When k=j: b[v,p[j]] becomes b[v,u] and b[u,p[j]] becomes b[u,v]
    
    # So the formula above uses the OLD p for all k, but for k=i and k=j, the new p is different.
    # Let's handle this by computing with old p and then correcting.
    
    # Let diff_row = a[i,:] - a[j,:]
    # Let diff_col = a[:,i] - a[:,j]
    # Let diff_loc_row = np.array([b[v,p[k]] - b[u,p[k]] for k in range(n)])
    # Let diff_loc_col = np.array([b[p[k],v] - b[p[k],u] for k in range(n)])
    
    # delta_approx = diff_row @ diff_loc_row + diff_col @ diff_loc_col
    
    # Corrections for k=i and k=j:
    # For k=i in diff_loc_row: we used b[v,p[i]] - b[u,p[i]] = b[v,u] - b[u,u]
    # But in the new assignment, p'[i] = v, so the term should use b[v,v] - b[u,v]... 
    # Hmm, this is getting complicated. Let me just do the direct computation.
    
    # For performance, let me precompute some things outside the loop.
    # For now, let's just compute the delta directly by summing over the affected terms.
    
    # The affected terms are those where i' in {i,j} or j' in {i,j}.
    # Let's compute the old and new contributions for these terms.
    
    # This will be O(n) per swap, which is acceptable.
    
    # Old contribution:
    old = 0
    for k in range(len(p)):
        # Term (i, k): a[i,k] * b[u, p[k]]
        old += a[i, k] * b[u, p[k]]
        # Term (k, i): a[k,i] * b[p[k], u]
        old += a[k, i] * b[p[k], u]
        # Term (j, k): a[j,k] * b[v, p[k]]
        old += a[j, k] * b[v, p[k]]
        # Term (k, j): a[k,j] * b[p[k], v]
        old += a[k, j] * b[p[k], v]
    
    # Subtract double-counted terms:
    # (i,j) and (j,i) appear in both i and j sums
    old -= a[i, j] * b[u, v]
    old -= a[j, i] * b[v, u]
    
    # New contribution (after swapping p[i] and p[j]):
    # p'[i] = v, p'[j] = u, p'[k] = p[k] for k != i,j
    new = 0
    for k in range(len(p)):
        pk_new = p[k] if k != i and k != j else (v if k == i else u)
        # Term (i, k): a[i,k] * b[v, pk_new]
        new += a[i, k] * b[v, pk_new]
        # Term (k, i): a[k,i] * b[pk_new, v]
        new += a[k, i] * b[pk_new, v]
        # Term (j, k): a[j,k] * b[u, pk_new]
        new += a[j, k] * b[u, pk_new]
        # Term (k, j): a[k,j] * b[pk_new, u]
        new += a[k, j] * b[pk_new, u]
    
    # Subtract double-counted terms:
    new -= a[i, j] * b[v, u]
    new -= a[j, i] * b[u, v]
    
    return new - old


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Initial solution via rank matching
    p = _rank_matching_initial(a, b)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # ADMM-inspired iterative improvement
    # Alternate between:
    # 1. Local search (2-opt) to refine the current permutation
    # 2. Hungarian-based reassignment using linearized cost matrix
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Step 1: 2-opt local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    delta = _two_opt_delta(a, b, p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        best_p = p.copy()
                        improved = True
        
        # Step 2: Hungarian reassignment using linearized cost
        if time.time() > deadline - 5:
            break
        
        cost_matrix = _linearized_cost_matrix(a, b, p)
        p_new = _hungarian_solve(cost_matrix)
        
        cost_new = _qap_cost(a, b, p_new)
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p = p_new
        else:
            # Keep current p for next iteration's local search
            pass
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
