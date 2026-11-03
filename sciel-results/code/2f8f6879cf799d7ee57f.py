"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Hungarian Algorithm on Linear Relaxation
    # We model the QAP as a maximum weight bipartite matching problem by constructing
    # a cost matrix C where C_{ij} represents the "affinity" between facility i and location j.
    # The linear relaxation of the QAP cost sum_{i,j} a[i,j] b[p[i], p[j]] can be approximated
    # by sum_i (sum_k a[i,k]) * (sum_k b[j,k]) or similar.
    # A stronger linearization: C[i, j] = - sum_k a[i, k] * b[j, k] is not directly linear in p.
    # However, the "linear sum assignment" heuristic often uses:
    # cost[i, j] = sum_k a[i, k] * b[j, k] is not standard.
    # Standard linearization: minimize sum_i,j a[i,j] b[p[i], p[j]].
    # If we fix the "average" flow, we can use:
    # Let's use the standard spectral or sum-based initialization as it is robust.
    # But the directive suggests: C_{ij} = -\sum_{k} a_{ik} b_{jk}.
    # Note: This C is an n x n matrix. Solving LAP on C gives a permutation.
    # This effectively minimizes sum_i C[i, p[i]] = - sum_i sum_k a[i,k] b[p[i], k].
    # This is a valid linear relaxation that captures pairwise affinities.
    
    try:
        # Compute affinity matrix C
        # C[i, j] = - sum_k a[i, k] * b[j, k]
        # This is - (A @ B.T)
        c_mat = -(a @ b.T)
        row_ind, col_ind = linear_sum_assignment(c_mat)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
    except Exception:
        # Fallback to simple sorting if LAP fails
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        p = np.argsort(np.argsort(a_sum, kind="stable"), kind="stable")

    # 2. Vectorized Local Search (Best-Improve)
    # We will perform a best-improve local search using 2-opt swaps.
    # To make this fast, we precompute the "delta" for all possible swaps (i, j) simultaneously.
    
    # Cost function: Cost(p) = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let P be the permutation matrix. Cost = Tr(P^T A P B) = sum_i,j a[i,j] b[p[i], p[j]].
    
    # Delta for swapping p[i] and p[j]:
    # Let u = p[i], v = p[j].
    # Old contribution involving i or j:
    # sum_{k} a[i,k] b[u, p[k]] + sum_{k} a[j,k] b[v, p[k]] + sum_{k} a[k,i] b[p[k], u] + sum_{k} a[k,j] b[p[k], v]
    # Note: The term a[i,j] b[u, v] and a[j,i] b[v, u] are counted twice in the full sum if we are not careful,
    # but since A and B are symmetric (usually), and the sum is over all i,j,
    # it is easier to compute the change directly.
    
    # Change in cost when swapping locations u and v for facilities i and j:
    # Delta = (a[i,:] - a[j,:]) @ (b[:, v] - b[:, u]) + (a[:, i] - a[:, j]) @ (b[u, :] - b[v, :])
    # Wait, let's derive carefully.
    # Cost = sum_{r,s} a[r,s] b[p[r], p[s]]
    # Terms affected by swapping p[i] and p[j]:
    # 1. Terms where r=i or s=i or r=j or s=j.
    # Let's use the property:
    # Delta = sum_{k} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) 
    #       + sum_{k} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    #       - 2 * (a[i,j] + a[j,i]) * (b[u,v] - b[v,u]) ? No, the cross terms are handled.
    
    # Actually, a simpler vectorized formula for delta(i, j) given current permutation p:
    # Let u = p[i], v = p[j].
    # Delta = (a[i] - a[j]) @ (B[:, v] - B[:, u]) + (A[:, i] - A[:, j]) @ (B[u, :] - B[v, :])
    # where B is the location matrix and A is the facility matrix.
    # Note: B[:, v] is the column v of B.
    # The term (a[i] - a[j]) @ (B[:, v] - B[:, u]) accounts for the "outgoing" flows from i and j to all other locations.
    # The term (A[:, i] - A[:, j]) @ (B[u, :] - B[v, :]) accounts for the "incoming" flows from all other facilities to i and j.
    # Since A and B are symmetric, these two terms are actually identical if we consider the full sum?
    # Let's check:
    # Sum_{k} a[i,k] b[v, p[k]] - sum_{k} a[i,k] b[u, p[k]]
    # Sum_{k} a[j,k] b[v, p[k]] - sum_{k} a[j,k] b[u, p[k]]
    # This is sum_{k} (a[i,k] - a[j,k]) b[v, p[k]] - sum_{k} (a[i,k] - a[j,k]) b[u, p[k]]
    # = (a[i] - a[j]) @ (B[:, v] - B[:, u])
    
    # The second part:
    # Sum_{k} a[k,i] b[p[k], v] - Sum_{k} a[k,i] b[p[k], u]
    # Sum_{k} a[k,j] b[p[k], v] - Sum_{k} a[k,j] b[p[k], u]
    # = Sum_{k} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
    # = (A[:, i] - A[:, j]) @ (B[u, :] - B[v, :])  -- Wait, B[u,:] is row u.
    # b[p[k], v] is element (p[k], v) of B.
    # So we are dotting vector (A[:,i]-A[:,j]) with vector (B[p[:], v] - B[p[:], u]).
    # B[p[:], v] is the column v of B, but permuted by p? No.
    # B[p[k], v] is just B at row p[k], col v.
    # So the vector is B[p, v] - B[p, u].
    
    # So Delta = (a[i] - a[j]) @ (B[:, v] - B[:, u]) + (a[:, i] - a[:, j]) @ (B[p, v] - B[p, u])
    
    # Precomputation:
    # We can precompute M = A @ B.T ? No.
    # Let's compute the delta matrix for all pairs (i, j) at once.
    # This is O(n^3) memory if we store all deltas, which is 150^3 = 3.3M floats = 26MB. This is fine.
    # But computing it is the cost.
    
    # Efficient calculation:
    # Delta(i, j) = a[i] @ B[:, v] - a[i] @ B[:, u] - a[j] @ B[:, v] + a[j] @ B[:, u]
    #              + a[:, i] @ B[p, v] - a[:, i] @ B[p, u] - a[:, j] @ B[p, v] + a[:, j] @ B[p, u]
    
    # Let's define:
    # X = A @ B  (n x n). X[i, v] = sum_k a[i,k] b[k,v].
    # Wait, b[k,v] is B[k, v].
    # So X[i, v] = sum_k a[i,k] B[k, v].
    # Then a[i] @ B[:, v] = X[i, v].
    # So first part: X[i, v] - X[i, u] - X[j, v] + X[j, u].
    
    # Second part:
    # a[:, i] @ B[p, v].
    # B[p, v] is the column v of B, indexed by p.
    # Let Y = A.T @ B ? No.
    # a[:, i] is column i of A.
    # We need sum_k a[k, i] B[p[k], v].
    # This is (A[:, i])^T @ B[p, :][:, v].
    # Let Z = A.T @ B.T ?
    # (A.T @ B.T)[i, v] = sum_k A.T[i, k] B.T[k, v] = sum_k A[k, i] B[v, k].
    # This is sum_k a[k, i] b[v, k].
    # We want sum_k a[k, i] b[p[k], v].
    # This is not directly a matrix product with standard A, B unless we permute.
    # However, note that B is symmetric. So b[p[k], v] = b[v, p[k]].
    # So sum_k a[k, i] b[v, p[k]].
    # This is (A[:, i])^T @ B[v, p].
    # This is hard to precompute for all v, p[k] without looping or heavy ops.
    
    # Alternative:
    # The second term is actually the same as the first term if A and B are symmetric?
    # Term 1: sum_k a[i,k] (b[v, p[k]] - b[u, p[k]])
    # Term 2: sum_k a[k,i] (b[p[k], v] - b[p[k], u])
    # Since A is symmetric, a[k,i] = a[i,k].
    # Since B is symmetric, b[p[k], v] = b[v, p[k]].
    # So Term 1 = Term 2.
    # Therefore, Delta(i, j) = 2 * ( (a[i] - a[j]) @ (B[:, v] - B[:, u]) )
    # Wait, is this correct?
    # Let's re-verify the "2x" factor.
    # Cost = sum_{r,s} a[r,s] b[p[r], p[s]].
    # Change in cost = Sum_{r,s} a[r,s] (b[p'[r], p'[s]] - b[p[r], p[s]]).
    # Only terms where r or s is i or j change.
    # r=i, s=j: a[i,j] (b[u,v] - b[v,u]) = 0 (since b is symmetric and u!=v? No, b[u,v] is same as b[v,u]).
    # Actually, if we swap p[i] and p[j], then p'[i]=v, p'[j]=u.
    # Term (i,j): a[i,j] b[v,u] - a[i,j] b[u,v] = 0.
    # Term (j,i): a[j,i] b[u,v] - a[j,i] b[v,u] = 0.
    # Term (i,k) for k!=i,j: a[i,k] (b[v, p[k]] - b[u, p[k]]).
    # Term (k,i) for k!=i,j: a[k,i] (b[p[k], v] - b[p[k], u]).
    # Since a[k,i]=a[i,k] and b[p[k],v]=b[v,p[k]], these two are equal.
    # So sum is 2 * sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]]).
    # Note that the sum in my previous derivation included k=i and k=j.
    # For k=i: a[i,i] (b[v, v] - b[u, u]).
    # For k=j: a[i,j] (b[v, u] - b[u, v]) = 0.
    # So the formula 2 * ( (a[i] - a[j]) @ (B[:, v] - B[:, u]) ) includes the k=i and k=j terms.
    # Let's check k=i term in the formula:
    # 2 * a[i,i] (B[i, v] - B[i, u])? No.
    # The formula is (a[i] - a[j]) @ (B[:, v] - B[:, u]).
    # Component i: a[i,i] (B[i, v] - B[i, u]) - a[j,i] (B[i, v] - B[i, u]).
    # This is getting messy.
    
    # Let's stick to the safe, slightly slower vectorized loop for the outer i, and vectorize the inner j.
    # Or, just compute the full delta matrix for all i,j in one go using matrix operations.
    
    # Delta[i, j] = 2 * ( A[i, :] @ (B[:, p[j]] - B[:, p[i]]) - A[j, :] @ (B[:, p[j]] - B[:, p[i]]) )
    # This is still O(n^3) if done naively.
    
    # Let's use the following optimization:
    # Precompute D = A @ B (n x n).
    # D[i, v] = sum_k a[i,k] b[k,v].
    # Then a[i] @ B[:, v] = D[i, v].
    # So (a[i] - a[j]) @ B[:, v] = D[i, v] - D[j, v].
    # And (a[i] - a[j]) @ B[:, u] = D[i, u] - D[j, u].
    # So the term is 2 * ( (D[i, v] - D[j, v]) - (D[i, u] - D[j, u]) ).
    # Wait, this assumes the "2x" and symmetry hold perfectly.
    # Let's verify:
    # Delta = 2 * sum_{k} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
    # = 2 * ( sum_k a[i,k] b[v, p[k]] - sum_k a[i,k] b[u, p[k]] - sum_k a[j,k] b[v, p[k]] + sum_k a[j,k] b[u, p[k]] )
    # = 2 * ( (a[i] @ B[:, v]) - (a[i] @ B[:, u]) - (a[j] @ B[:, v]) + (a[j] @ B[:, u]) )
    # Note: a[i] @ B[:, v] is sum_k a[i,k] b[k, v]? No!
    # B[:, v] is the vector b[0,v], b[1,v], ..., b[n-1,v].
    # a[i] @ B[:, v] = sum_k a[i,k] b[k, v].
    # But we need sum_k a[i,k] b[v, p[k]].
    # These are different! b[k,v] is not b[v, p[k]].
    # So the precomputation D = A @ B does NOT work directly for the permuted indices.
    
    # We must use the permuted B.
    # Let Bp = B[:, p] ? No, B[p, :]?
    # We need b[v, p[k]].
    # Let M = B.T. Then M[v, k] = b[k, v].
    # We want sum_k a[i,k] b[v, p[k]].
    # This is a[i] @ (B.T[v, p])? No.
    # B.T[v, :] is b[:, v].
    # We need to index this by p.
    # Let Bp = B.T[:, p] ? No.
    # Let's define B_perm = B[p, :].
    # Then B_perm[k, v] = b[p[k], v].
    # We want b[v, p[k]]. Since B is symmetric, b[v, p[k]] = b[p[k], v] = B_perm[k, v].
    # So sum_k a[i,k] b[v, p[k]] = sum_k a[i,k] B_perm[k, v] = (a[i] @ B_perm.T)[v]?
    # B_perm.T[v, k] = B_perm[k, v] = b[p[k], v].
    # So yes, a[i] @ B_perm.T gives a vector of length n, where element v is sum_k a[i,k] b[p[k], v].
    # This is exactly what we need for the "v" part.
    
    # So, for a fixed permutation p:
    # Let AP = A @ B_perm.T (n x n).
    # AP[i, v] = sum_k a[i,k] b[p[k], v].
    # Then Delta(i, j) = 2 * ( AP[i, p[j]] - AP[i, p[i]] - AP[j, p[j]] + AP[j, p[i]] ).
    # Wait, let's check the indices.
    # Term 1: sum_k a[i,k] b[v, p[k]] where v=p[j]. This is AP[i, p[j]].
    # Term 2: sum_k a[i,k] b[u, p[k]] where u=p[i]. This is AP[i, p[i]].
    # Term 3: sum_k a[j,k] b[v, p[k]] where v=p[j]. This is AP[j, p[j]].
    # Term 4: sum_k a[j,k] b[u, p[k]] where u=p[i]. This is AP[j, p[i]].
    
    # So Delta(i, j) = 2 * ( AP[i, p[j]] - AP[i, p[i]] - AP[j, p[j]] + AP[j, p[i]] ).
    
    # This allows us to compute the delta for a specific (i, j) in O(1) time after O(n^2) precomputation of AP.
    # However, AP changes when p changes.
    # If we swap p[i] and p[j], how does AP change?
    # B_perm changes.
    # Specifically, B_perm[k, v] changes for v in {p[i], p[j]}?
    # No, B_perm is B[p, :].
    # If we swap p[i] and p[j], the rows i and j of B_perm are swapped.
    # So B_perm_new[i, :] = B_perm_old[j, :]
    # B_perm_new[j, :] = B_perm_old[i, :]
    # Other rows are unchanged.
    
    # AP = A @ B_perm.T.
    # AP_new = A @ B_perm_new.T.
    # B_perm_new.T has columns i and j swapped compared to B_perm.T.
    # So AP_new[:, i] = AP_old[:, j]
    # AP_new[:, j] = AP_old[:, i]
    # AP_new[:, k] = AP_old[:, k] for k != i, j.
    
    # This is a very efficient update!
    # We can maintain AP and update it in O(n) time per swap.
    
    # Algorithm:
    # 1. Init p.
    # 2. Compute B_perm = B[p, :].
    # 3. Compute AP = A @ B_perm.T.
    # 4. Loop:
    #    a. Find best (i, j) with i < j such that Delta(i, j) < 0.
    #       Delta(i, j) = 2 * ( AP[i, p[j]] - AP[i, p[i]] - AP[j, p[j]] + AP[j, p[i]] ).
    #       We can vectorize this search.
    #       Let's construct a matrix Delta_Mat of size n x n.
    #       Delta_Mat[i, j] = 2 * ( AP[i, p[j]] - AP[i, p[i]] - AP[j, p[j]] + AP[j, p[i]] ).
    #       This can be computed as:
    #       Let U = AP[:, p] (n x n). U[i, k] = AP[i, p[k]].
    #       Then AP[i, p[j]] = U[i, j].
    #       AP[i, p[i]] = U[i, i] (diagonal).
    #       AP[j, p[j]] = U[j, j].
    #       AP[j, p[i]] = U[j, i].
    #       So Delta_Mat[i, j] = 2 * ( U[i, j] - U[i, i] - U[j, j] + U[j, i] ).
    #       This is a simple matrix operation!
    #       U is n x n.
    #       Diag_U = np.diag(U).
    #       Delta_Mat = 2 * ( U - np.diag(Diag_U) - Diag_U[:, None] + U.T )?
    #       Let's check:
    #       U[i, j] - U[i, i] - U[j, j] + U[j, i]
    #       = U[i, j] + U[j, i] - U[i, i] - U[j, j]
    #       Let M = U + U.T.
    #       Let D = np.diag(U).
    #       Delta_Mat = 2 * ( M - D[:, None] - D[None, :] ).
    #    b. Find min in Delta_Mat for i < j.
    #    c. If min < 0, swap p[i], p[j].
    #       Update AP:
    #       u = p[i], v = p[j].
    #       AP[:, i] and AP[:, j] are swapped?
    #       Recall: AP_new[:, i] = AP_old[:, j] and AP_new[:, j] = AP_old[:, i].
    #       So we just swap columns i and j of AP.
    #       Wait, is that correct?
    #       AP = A @ B_perm.T.
    #       B_perm.T has columns corresponding to locations.
    #       B_perm[k, v] = b[p[k], v].
    #       B_perm.T[v, k] = b[p[k], v].
    #       If we swap p[i] and p[j], then for row i of B_perm, the values change.
    #       B_perm_new[i, v] = b[p[j], v] = B_perm_old[j, v].
    #       B_perm_new[j, v] = b[p[i], v] = B_perm_old[i, v].
    #       So B_perm_new.T[v, i] = B_perm_old.T[v, j].
    #       So column i of B_perm_new.T is column j of B_perm_old.T.
    #       So column i of AP_new is column j of AP_old.
    #       Yes! We just swap columns i and j of AP.
    #    d. Repeat until no improvement or time limit.
    
    # This is O(n^2) per iteration (to compute Delta_Mat and find min).
    # n=150, n^2=22500.
    # 115 seconds / (time for 22500 ops) -> potentially many iterations.
    # NumPy ops on 150x150 are very fast (~microseconds).
    # We can likely do 10,000+ iterations.
    
    current_cost = int(np.sum(a * b[p][:, p]))
    best_cost = current_cost
    best_p = p.copy()
    
    B_perm = b[p, :]
    AP = a @ B_perm.T
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Compute Delta Matrix
        U = AP[:, p] # U[i, j] = AP[i, p[j]]
        D = np.diag(U)
        M = U + U.T
        Delta_Mat = 2 * (M - D[:, None] - D[None, :])
        
        # Mask out i=j and i>j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        Delta_Mat_masked = np.where(mask, Delta_Mat, np.inf)
        
        min_val = np.min(Delta_Mat_masked)
        
        if min_val >= 0:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturb
                # Random swap
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
                AP[:, i], AP[:, j] = AP[:, j].copy(), AP[:, i].copy()
                current_cost = int(np.sum(a * b[p][:, p]))
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve = 0
                continue
            else:
                # Try a few random swaps
                for _ in range(10):
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
                    AP[:, i], AP[:, j] = AP[:, j].copy(), AP[:, i].copy()
                    current_cost = int(np.sum(a * b[p][:, p]))
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                no_improve = 0
                continue
        else:
            no_improve = 0
        
        # Find indices of min
        min_idx = np.unravel_index(np.argmin(Delta_Mat_masked), Delta_Mat.shape)
        i, j = min_idx
        
        # Perform swap
        p[i], p[j] = p[j], p[i]
        AP[:, i], AP[:, j] = AP[:, j].copy(), AP[:, i].copy()
        
        current_cost += int(min_val)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
