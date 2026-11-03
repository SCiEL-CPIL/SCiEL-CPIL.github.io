"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Compute initial cost
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    best_cost = compute_cost(p)
    best_p = p.copy()

    # Precompute row sums of a for incremental updates
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For a swap of p[i] and p[j], the change is:
    #   delta = sum_k a[i,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #         + sum_k a[j,k]*(b[p[j],p[k]] - b[p[i],p[k]])
    #         + sum_k a[k,i]*(b[p[k],p[i]] - b[p[k],p[j]])
    #         + sum_k a[k,j]*(b[p[k],p[j]] - b[p[k],p[i]])
    #         - (diagonal corrections if i==j, but i!=j here)
    # Simplified: let u = p[i], v = p[j]
    #   delta = sum_k (a[i,k] + a[k,i]) * (b[u,p[k]] - b[v,p[k]])
    #         + sum_k (a[j,k] + a[k,j]) * (b[v,p[k]] - b[u,p[k]])
    #         = sum_k (a[i,k]+a[k,i] - a[j,k]-a[k,j]) * (b[u,p[k]] - b[v,p[k]])

    ai = a + a.T  # symmetric sum of a (for the swap delta formula)
    # Actually let's be more careful:
    # cost = sum_{i,j} a[i,j] * b[p[i],p[j]]
    # After swapping p[i] <-> p[j]:
    # Terms involving i or j change:
    # Old: sum_k a[i,k]*b[u,p[k]] + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],u] + sum_k a[k,j]*b[p[k],v]
    # New: sum_k a[i,k]*b[v,p[k]] + sum_k a[j,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],v] + sum_k a[k,j]*b[p[k],u]
    # Delta = sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
    #       = sum_k [(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])]
    # Note b[p[k],v] - b[p[k],u] = -(b[u,p[k]] - b[v,p[k]]) if b is symmetric, but it may not be.
    # Let's just compute directly for correctness.

    def swap_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        # Get the relevant rows/columns
        # sum_k a[i,k]*b[v,p[k]] - a[i,k]*b[u,p[k]] = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]])
        # sum_k a[j,k]*b[u,p[k]] - a[j,k]*b[v,p[k]] = sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]])
        # sum_k a[k,i]*b[p[k],v] - a[k,i]*b[p[k],u] = sum_k a[k,i]*(b[p[k],v] - b[p[k],u])
        # sum_k a[k,j]*b[p[k],u] - a[k,j]*b[p[k],v] = sum_k a[k,j]*(b[p[k],u] - b[p[k],v])
        # Note: when k==i, p[k]==u, so b[p[k],v]-b[p[k],u] = b[u,v]-b[u,u], and b[v,p[k]]-b[u,p[k]] = b[v,u]-b[u,u]
        # When k==j, p[k]==v, so b[p[k],v]-b[p[k],u] = b[v,v]-b[v,u], and b[v,p[k]]-b[u,p[k]] = b[v,v]-b[u,v]
        # These are handled correctly by the general formula.
        
        # Vectorized computation
        # For term 1: sum_k a[i,k] * (b[v,p[k]] - b[u,p[k]])
        # b[v,p[k]] is row v of b indexed by p
        # b[u,p[k]] is row u of b indexed by p
        diff_row = b[v, p] - b[u, p]  # shape (n,)
        term1 = np.dot(a[i, :], diff_row)
        
        # For term 2: sum_k a[j,k] * (b[u,p[k]] - b[v,p[k]]) = -sum_k a[j,k] * (b[v,p[k]] - b[u,p[k]])
        term2 = -np.dot(a[j, :], diff_row)
        
        # For term 3: sum_k a[k,i] * (b[p[k],v] - b[p[k],u])
        # b[p[k],v] is column v of b indexed by p -> b[p, v]
        # b[p[k],u] is column u of b indexed by p -> b[p, u]
        diff_col = b[p, v] - b[p, u]  # shape (n,)
        term3 = np.dot(a[:, i], diff_col)
        
        # For term 4: sum_k a[k,j] * (b[p[k],u] - b[p[k],v]) = -sum_k a[k,j] * (b[p[k],v] - b[p[k],u])
        term4 = -np.dot(a[:, j], diff_col)
        
        return term1 + term2 + term3 + term4

    # 2-opt local search
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        # Limit passes to avoid infinite loops
        if no_improve_count > 5:
            break
        for i in range(n - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                d = swap_delta(i, j, p, a, b)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += d
                    improved = True
                    no_improve_count = 0
                    if best_cost < compute_cost(p):  # occasional sanity check
                        pass  # trust incremental if consistent

    # Final verification
    final_cost = compute_cost(p)
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = p.copy()
    else:
        # Revert to best
        p = best_p

    return p.tolist()
# EVOLVE-BLOCK-END
