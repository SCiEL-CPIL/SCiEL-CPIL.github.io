"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_cost(a, b, p):
    """Compute exact cost sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _calc_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute row sums of a for quick reference
    a_row_sums = a.sum(axis=1)
    
    # Local search with 2-opt swaps
    # For each pair (i,j), compute delta cost of swapping p[i] and p[j]
    # delta = sum_{k!=i,j} [a[i,k]*(b[pj,pk] - b[pi,pk]) + a[j,k]*(b[pi,pk] - b[pj,pk])]
    #         + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # We'll use a first-improvement or best-improvement strategy
    # Best-improvement is slower per iteration but may converge faster
    
    no_improve_count = 0
    max_no_improve = 50  # after 50 full passes without improvement, stop
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            pi = p[i]
            a_i = a[i]
            for j in range(i + 1, n):
                pj = p[j]
                a_j = a[j]
                
                # Compute delta cost for swapping i and j
                # Terms from k != i, j:
                # a[i,k] * (b[pj,pk] - b[pi,pk]) + a[j,k] * (b[pi,pk] - b[pj,pk])
                # = (a[i,k] - a[j,k]) * (b[pj,pk] - b[pi,pk])
                
                # Vectorized computation for all k at once
                # But we need to exclude i and j
                diff_a = a_i - a_j  # shape (n,)
                diff_b = b[pj] - b[pi]  # shape (n,)
                
                # Sum over all k, then subtract k=i and k=j contributions
                total = np.dot(diff_a, diff_b)
                
                # Subtract contribution from k=i:
                # a[i,i]*(b[pj,pi]-b[pi,pi]) + a[j,i]*(b[pi,pi]-b[pj,pi])
                # = (a[i,i] - a[j,i]) * (b[pj,pi] - b[pi,pi])
                # But wait, in the sum over k, when k=i:
                # a[i,i]*(b[pj,p[i]] - b[pi,p[i]]) + a[j,i]*(b[pi,p[i]] - b[pj,p[i]])
                # = a[i,i]*(b[pj,pi] - b[pi,pi]) + a[j,i]*(b[pi,pi] - b[pj,pi])
                # = (a[i,i] - a[j,i]) * (b[pj,pi] - b[pi,pi])
                
                # When k=j:
                # a[i,j]*(b[pj,pj] - b[pi,pj]) + a[j,j]*(b[pi,pj] - b[pj,pj])
                # = (a[i,j] - a[j,j]) * (b[pj,pj] - b[pi,pj])
                
                # The diagonal terms a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
                # = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
                
                # So total delta = total - [k=i term] - [k=j term] + diagonal correction
                # Actually, the formula I derived includes all k, and the diagonal terms
                # a[i,j] and a[j,i] are handled separately. Let me redo.
                
                # Original cost terms involving i or j:
                # For k != i,j: a[i,k]*b[pi,pk] + a[j,k]*b[pj,pk]
                # a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
                # Plus self terms: a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj]
                
                # After swap:
                # For k != i,j: a[i,k]*b[pj,pk] + a[j,k]*b[pi,pk]
                # a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
                # Plus self terms: a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi]
                
                # Delta for k != i,j part:
                # sum_{k!=i,j} [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[j,k]*(b[pi,pk]-b[pj,pk])]
                # = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
                
                # Delta for i,j cross terms:
                # a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
                # = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                
                # Delta for self terms:
                # a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
                # = (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
                
                # So total delta = sum_{k!=i,j} (a_i[k]-a_j[k])*(b[pj,k]-b[pi,k])
                #                  + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                #                  + (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
                
                # Compute sum over all k first:
                sum_all_k = np.dot(diff_a, diff_b)
                
                # Subtract k=i term: diff_a[i] * diff_b[i] = (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
                # Subtract k=j term: diff_a[j] * diff_b[j] = (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                
                sum_excl_ij = sum_all_k - diff_a[i] * diff_b[i] - diff_a[j] * diff_b[j]
                
                # Add cross and self corrections
                cross_term = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
                self_term = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
                
                delta = sum_excl_ij + cross_term + self_term
                
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
