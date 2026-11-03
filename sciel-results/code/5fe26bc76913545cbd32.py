import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Convert to int64 arrays for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    best_p = None
    best_cost = np.iinfo(np.int64).max

    # Precompute column sums of b for fast delta calculation
    # b_col_sums[j] = sum_i b[i, j]
    b_col_sums = b.sum(axis=0).astype(np.int64)

    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This is a trace: a @ b_perm @ a.T? No.
        # sum_{i,j} a[i,j] b[p[i], p[j]] = sum_{i,j} a[i,j] b[p[i], p[j]]
        # Let B_p be the permutation of b such that B_p[i,j] = b[p[i], p[j]]
        # Then cost = sum_{i,j} a[i,j] B_p[i,j] = np.sum(a * B_p)
        # Or equivalently np.einsum('ij,kl->', a, b[np.ix_(p, p)])
        # More efficient: use matrix multiplication if possible?
        # Actually, np.sum(a * b[np.ix_(p, p)]) is clear and O(n^2).
        return np.sum(a * b[np.ix_(p, p)])

    def delta_cost_swap(p, i, j):
        """
        Calculate the change in cost when swapping locations of facilities i and j.
        p[i] = u, p[j] = v. After swap, p'[i] = v, p'[j] = u.
        
        Cost = sum_{m,n} a[m,n] b[p[m], p[n]]
        Only terms involving p[i] or p[j] change.
        Specifically:
        - Terms where m=i or n=i (involving facility i's location)
        - Terms where m=j or n=j (involving facility j's location)
        
        Let u = p[i], v = p[j].
        Before:
        Terms with i: sum_n a[i,n] b[u, p[n]] + sum_m a[m,i] b[p[m], u] - a[i,i] b[u,u] (double counted)
        Terms with j: sum_n a[j,n] b[v, p[n]] + sum_m a[m,j] b[p[m], v] - a[j,j] b[v,v] (double counted)
        
        After:
        Terms with i: sum_n a[i,n] b[v, p[n]] + sum_m a[m,i] b[p[m], v] - a[i,i] b[v,v]
        Terms with j: sum_n a[j,n] b[u, p[n]] + sum_m a[m,j] b[p[m], u] - a[j,j] b[u,u]
        
        Note: p[i] and p[j] are swapped, so in the sums above, when m=j in the "i" terms, p[m] changes from v to u, and when m=i in the "j" terms, p[m] changes from u to v.
        Wait, the sums are over all m,n. Let's be precise.
        
        C = sum_{m,n} a[m,n] b[p[m], p[n]]
        
        Let S_i = { (m,n) | m==i or n==i }
        Let S_j = { (m,n) | m==j or n==j }
        
        The change is:
        sum_{(m,n) in S_i ∪ S_j} a[m,n] * (b[p'[m], p'[n]] - b[p[m], p[n]])
        
        Since p' only differs at i and j:
        p'[m] = p[m] for m != i,j
        p'[i] = v, p'[j] = u
        
        So b[p'[m], p'[n]] != b[p[m], p[n]] only if m in {i,j} or n in {i,j}.
        
        We can compute this by iterating over the relevant indices.
        However, a more efficient vectorized approach:
        
        Let A_i = a[i, :] (row i)
        Let A_j = a[j, :] (row j)
        Let A_col_i = a[:, i] (col i)
        Let A_col_j = a[:, j] (col j)
        
        The terms involving i:
        sum_n a[i,n] b[p[i], p[n]] = a[i,:] @ b[p[i], p]  (dot product of a[i,:] with row p[i] of b permuted by p)
        sum_m a[m,i] b[p[m], p[i]] = a[:,i] @ b[p, p[i]]  (dot product of a[:,i] with col p[i] of b permuted by p)
        
        Actually, let's use the property:
        delta = (a[i,:] @ (b[v, p] - b[u, p])) + (a[:,i] @ (b[p, v] - b[p, u])) 
               + (a[j,:] @ (b[u, p] - b[v, p])) + (a[:,j] @ (b[p, u] - b[p, v]))
               - corrections for double counting (i,j) and (j,i) and (i,i), (j,j)
        
        This is getting complicated. Given n=256, a simple O(n) loop for delta calculation is fast enough.
        O(n) per swap, 256^2/2 pairs per pass = ~32k pairs * 256 ops = ~8M ops per pass.
        NumPy can do this in milliseconds.
        
        Let's implement the delta calculation by explicitly computing the cost contribution of the affected terms.
        """
        u = p[i]
        v = p[j]
        
        # If u == v, no change (shouldn't happen in a permutation)
        if u == v:
            return 0
        
        # Before:
        # Contribution of i: sum_n a[i,n] b[u, p[n]]
        # Contribution of j: sum_n a[j,n] b[v, p[n]]
        # Contribution of col i: sum_m a[m,i] b[p[m], u]
        # Contribution of col j: sum_m a[m,j] b[p[m], v]
        # Subtract double counted: a[i,i] b[u,u], a[j,j] b[v,v], a[i,j] b[u,v], a[j,i] b[v,u]
        
        # After:
        # Contribution of i: sum_n a[i,n] b[v, p[n]]
        # Contribution of j: sum_n a[j,n] b[u, p[n]]
        # Contribution of col i: sum_m a[m,i] b[p[m], v]
        # Contribution of col j: sum_m a[m,j] b[p[m], u]
        # Subtract double counted: a[i,i] b[v,v], a[j,j] b[u,u], a[i,j] b[v,u], a[j,i] b[u,v]
        
        # Note: in the sums, when n=j in the first sum, p[n]=v becomes p'[n]=u.
        # When n=i in the first sum, p[n]=u becomes p'[n]=v.
        # The formula above using p (not p') is correct because we are comparing b[u,p[n]] vs b[v,p[n]] where p is the OLD permutation.
        # But wait, p[n] is the same for n != i,j. For n=i, p[i]=u, for n=j, p[j]=v.
        # So the formula is correct as stated.
        
        # Let's vectorize this.
        # row_i = a[i, :]
        # row_j = a[j, :]
        # col_i = a[:, i]
        # col_j = a[:, j]
        
        # Before contributions:
        # sum_n a[i,n] b[u, p[n]] = a[i,:] @ b[u, p]
        # sum_n a[j,n] b[v, p[n]] = a[j,:] @ b[v, p]
        # sum_m a[m,i] b[p[m], u] = a[:,i] @ b[p, u]
        # sum_m a[m,j] b[p[m], v] = a[:,j] @ b[p, v]
        
        # After contributions:
        # sum_n a[i,n] b[v, p[n]] = a[i,:] @ b[v, p]
        # sum_n a[j,n] b[u, p[n]] = a[j,:] @ b[u, p]
        # sum_m a[m,i] b[p[m], v] = a[:,i] @ b[p, v]
        # sum_m a[m,j] b[p[m], u] = a[:,j] @ b[p, u]
        
        # Double counts to subtract:
        # Before: a[i,i]b[u,u] + a[j,j]b[v,v] + a[i,j]b[u,v] + a[j,i]b[v,u]
        # After:  a[i,i]b[v,v] + a[j,j]b[u,u] + a[i,j]b[v,u] + a[j,i]b[u,v]
        
        # This requires indexing b[u,p], b[v,p], etc. which is O(n).
        # Let's just do it directly.
        
        bp = b[np.ix_(p, p)]  # This is O(n^2) to create, too slow for delta.
        # Better: use b[u, p], b[v, p], b[p, u], b[p, v] directly.
        
        before_i = np.dot(a[i, :], b[u, p])
        before_j = np.dot(a[j, :], b[v, p])
        before_ci = np.dot(a[:, i], b[p, u])
        before_cj = np.dot(a[:, j], b[p, v])
        before_dd = a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
        
        after_i = np.dot(a[i, :], b[v, p])
        after_j = np.dot(a[j, :], b[u, p])
        after_ci = np.dot(a[:, i], b[p, v])
        after_cj = np.dot(a[:, j], b[p, u])
        after_dd = a[i,i]*b[v,v] + a[j,j]*b[u,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v]
        
        delta = (after_i + after_j + after_ci + after_cj - after_dd) - \
                (before_i + before_j + before_ci + before_cj - before_dd)
        
        return delta

    def local_search_2opt(p, max_time=None):
        """Perform 2-opt local search."""
        p = p.copy()
        current_cost = compute_cost(p)
        improved = True
        passes = 0
        
        while improved:
            improved = False
            passes += 1
            for i in range(n):
                if max_time and time.time() > max_time:
                    return p, current_cost
                for j in range(i+1, n):
                    delta = delta_cost_swap(p, i, j)
                    if delta < 0:
                        # Swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # After swap, we could restart, but continuing is fine for speed
                        # We don't update precomputed sums, but we don't use them here.
        return p, current_cost

    # Generate initial solutions
    # 1. Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    best_p, best_cost = local_search_2opt(p_init, max_time=deadline - 5)

    # 2. Try multiple random restarts
    random.seed(42)
    np_random = np.random.RandomState(42)
    
    num_restarts = 0
    while time.time() < deadline - 10:
        p_rand = np_random.permutation(n).astype(np.int64)
        p_ls, cost_ls = local_search_2opt(p_rand, max_time=deadline - 5)
        num_restarts += 1
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    # Final check
    if best_p is None:
        # Fallback to rank matching if no search completed
        best_p = p_init
    
    return best_p.tolist()
