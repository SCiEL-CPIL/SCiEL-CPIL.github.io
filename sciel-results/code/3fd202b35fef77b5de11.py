import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initial Solution: Rank Matching
    # Rows of A with largest totals go to rows of B with smallest totals
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort indices
    a_order = np.argsort(-a_sums, kind="stable")
    b_order = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[a_order] = b_order
    
    # Check if A and B are symmetric to use optimized delta
    # For wil100, they are symmetric. We check to be safe.
    symmetric = np.allclose(a, a.T) and np.allclose(b, b.T)
    
    # Helper to calculate cost
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    
    # 2. Local Search: 2-opt (Swap)
    # We iterate until no improvement or time runs out
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Precompute A rows for faster access? a is already numpy.
        # For each pair i, j
        for i in range(n):
            u = p[i]
            for j in range(i + 1, n):
                v = p[j]
                
                if symmetric:
                    # Delta = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u]) + 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[v,p_k] - b[u,p_k])
                    # We can compute the sum over all k and then subtract the i,j terms if they interfere, 
                    # or just loop k.
                    # Let's try vectorized sum for the whole vector first, then correct for i,j.
                    
                    # Vector D = a[i] - a[j]
                    # Vector E = b[v] - b[u]
                    # Sum = dot(D, E[p])
                    
                    # D and E are length n.
                    # E_perm = E[p]
                    # Sum_all = np.dot(D, E_perm)
                    
                    # This includes k=i and k=j.
                    # Term for k=i: D[i] * E_perm[i] = (a[i,i] - a[j,i]) * (b[v, p_i] - b[u, p_i])
                    # p_i = u. So (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
                    # Term for k=j: D[j] * E_perm[j] = (a[i,j] - a[j,j]) * (b[v, p_j] - b[u, p_j])
                    # p_j = v. So (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
                    
                    # The formula derived earlier:
                    # Delta = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u]) + 2 * Sum_{k!=i,j}
                    # Let S = Sum_{k} D[k] * E_perm[k]
                    # S = Term_i + Term_j + Sum_{k!=i,j}
                    # So Sum_{k!=i,j} = S - Term_i - Term_j
                    
                    # Delta = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u]) + 2 * (S - Term_i - Term_j)
                    
                    D = a[i] - a[j]
                    E = b[v] - b[u]
                    E_perm = E[p]
                    S = np.dot(D, E_perm)
                    
                    Term_i = D[i] * E_perm[i]
                    Term_j = D[j] * E_perm[j]
                    
                    Sum_excl = S - Term_i - Term_j
                    
                    delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + 2 * Sum_excl
                else:
                    # General case fallback (slower)
                    # Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
                    # + sum_{k!=i,j} (a[i,k](b[v,p_k]-b[u,p_k]) + a[j,k](b[u,p_k]-b[v,p_k]) + a[k,i](b[p_k,v]-b[p_k,u]) + a[k,j](b[p_k,u]-b[p_k,v]))
                    
                    # This is complex to vectorize cleanly without masks.
                    # Given the task is wil100 (symmetric), we assume symmetric.
                    # If not symmetric, this code might fail or be slow.
                    # For robustness, let's just use the general O(n) loop if not symmetric.
                    delta = 0
                    delta += a[i, i] * (b[v, v] - b[u, u])
                    delta += a[j, j] * (b[u, u] - b[v, v])
                    delta += a[i, j] * (b[v, u] - b[u, v])
                    delta += a[j, i] * (b[u, v] - b[v, u])
                    
                    # Sum over k
                    # We can vectorize the sum part too
                    # Let's construct vectors for the sum
                    # For k != i, j:
                    # term_k = a[i,k](b[v,p_k]-b[u,p_k]) + a[j,k](b[u,p_k]-b[v,p_k]) + a[k,i](b[p_k,v]-b[p_k,u]) + a[k,j](b[p_k,u]-b[p_k,v])
                    
                    # This is effectively:
                    # (a[i,:] - a[j,:]) * (b[v,p] - b[u,p])  -- wait, signs
                    # a[i,k] * (b[v,p_k] - b[u,p_k])
                    # - a[j,k] * (b[v,p_k] - b[u,p_k])
                    # a[k,i] * (b[p_k,v] - b[p_k,u])
                    # - a[k,j] * (b[p_k,v] - b[p_k,u])
                    
                    # Let V1 = b[v, p] - b[u, p]
                    # Let V2 = b[p, v] - b[p, u]
                    # Sum = dot(a[i] - a[j], V1) + dot(a[:,i] - a[:,j], V2)
                    
                    V1 = b[v, p] - b[u, p]
                    V2 = b[p, v] - b[p, u]
                    
                    S1 = np.dot(a[i] - a[j], V1)
                    S2 = np.dot(a[:, i] - a[:, j], V2)
                    
                    # Subtract k=i and k=j contributions from S1 and S2
                    # S1 includes k=i: (a[i,i]-a[j,i]) * V1[i]
                    # V1[i] = b[v,p_i] - b[u,p_i] = b[v,u] - b[u,u]
                    # S1 includes k=j: (a[i,j]-a[j,j]) * V1[j]
                    # V1[j] = b[v,p_j] - b[u,p_j] = b[v,v] - b[u,v]
                    
                    # S2 includes k=i: (a[i,i]-a[i,j]) * V2[i]
                    # V2[i] = b[p_i,v] - b[p_i,u] = b[u,v] - b[u,u]
                    # S2 includes k=j: (a[j,i]-a[j,j]) * V2[j]
                    # V2[j] = b[p_j,v] - b[p_j,u] = b[v,v] - b[v,u]
                    
                    S1_excl = S1 - (a[i, i] - a[j, i]) * (b[v, u] - b[u, u]) - (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    S2_excl = S2 - (a[i, i] - a[i, j]) * (b[u, v] - b[u, u]) - (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                    
                    delta += S1_excl + S2_excl

                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i != -1:
            # Apply swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True

    return p.tolist()
