import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Initialize
    # Use the greedy heuristic from seed as it's fast and good
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def compute_cost(p):
        return float(np.sum(af * bf[np.ix_(p, p)]))
        
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115
    
    # SA Parameters
    T = 10000.0
    cooling = 0.999 # or dynamic
    
    # Precompute for fast delta
    # We will compute delta in O(n) using vectorization
    
    while time.time() - start_time < time_limit:
        # Pick random swap
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
            
        x = p[i]
        y = p[j]
        
        # Calculate Delta
        # Sum over all k, then subtract k=i, k=j
        # Term for i moving x->y:
        # S_i = sum_k a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
        # = dot(af[i,:], bf[y,p] - bf[x,p]) + dot(af[:,i], bf[p,y] - bf[p,x])
        
        diff_b_yx = bf[y, p] - bf[x, p]
        diff_b_pyx = bf[p, y] - bf[p, x]
        
        sum_i = np.dot(af[i, :], diff_b_yx) + np.dot(af[:, i], diff_b_pyx)
        
        # Term for j moving y->x:
        diff_b_xy = bf[x, p] - bf[y, p] # -diff_b_yx
        diff_b_pxy = bf[p, x] - bf[p, y] # -diff_b_pyx
        
        sum_j = np.dot(af[j, :], diff_b_xy) + np.dot(af[:, j], diff_b_pxy)
        
        # These sums include k=i and k=j.
        # We need to subtract the contributions of k=i and k=j from sum_i and sum_j,
        # and then add the correct diagonal terms.
        
        # In sum_i (i moving x->y), the term k=i is:
        # a[i,i]*(b[y,p[i]] - b[x,p[i]]) + a[i,i]*(b[p[i],y] - b[p[i],x])
        # p[i] is x. So b[y,x] - b[x,x] and b[x,y] - b[x,x].
        # Term k=i in sum_i = a[i,i]*(b[y,x] - b[x,x]) + a[i,i]*(b[x,y] - b[x,x])
        
        # The term k=j in sum_i is:
        # a[i,j]*(b[y,p[j]] - b[x,p[j]]) + a[j,i]*(b[p[j],y] - b[p[j],x])
        # p[j] is y. So b[y,y] - b[x,y] and b[y,y] - b[y,x].
        # Term k=j in sum_i = a[i,j]*(b[y,y] - b[x,y]) + a[j,i]*(b[y,y] - b[y,x])
        
        # In sum_j (j moving y->x), the term k=j is:
        # a[j,j]*(b[x,p[j]] - b[y,p[j]]) + a[j,j]*(b[p[j],x] - b[p[j],y])
        # p[j] is y. So b[x,y] - b[y,y] and b[y,x] - b[y,y].
        # Term k=j in sum_j = a[j,j]*(b[x,y] - b[y,y]) + a[j,j]*(b[y,x] - b[y,y])
        
        # The term k=i in sum_j is:
        # a[j,i]*(b[x,p[i]] - b[y,p[i]]) + a[i,j]*(b[p[i],x] - b[p[i],y])
        # p[i] is x. So b[x,x] - b[y,x] and b[x,x] - b[x,y].
        # Term k=i in sum_j = a[j,i]*(b[x,x] - b[y,x]) + a[i,j]*(b[x,x] - b[x,y])
        
        # The "true" delta is the change in cost.
        # Old terms involving i,j:
        # a[i,j]b[x,y] + a[j,i]b[y,x]
        # + (sum_k!=i,j a[i,k]b[x,p[k]] + a[k,i]b[p[k],x])
        # + (sum_k!=i,j a[j,k]b[y,p[k]] + a[k,j]b[p[k],y])
        # + a[i,i]b[x,x] + a[j,j]b[y,y]
        
        # New terms:
        # a[i,j]b[y,x] + a[j,i]b[x,y]
        # + (sum_k!=i,j a[i,k]b[y,p[k]] + a[k,i]b[p[k],y])
        # + (sum_k!=i,j a[j,k]b[x,p[k]] + a[k,j]b[p[k],x])
        # + a[i,i]b[y,y] + a[j,j]b[x,x]
        
        # Delta = New - Old
        
        # Let's just compute the full sums including i and j, then adjust.
        # Sum_i_all = sum_k a[i,k](b[y,p[k]] - b[x,p[k]]) + a[k,i](b[p[k],y] - b[p[k],x])
        # This is the change in contribution of facility i if it moved from x to y, 
        # assuming other facilities stay at p[k].
        # But p[j] changes too.
        
        # The formula Delta = Sum_i_all + Sum_j_all - Correction is tricky.
        
        # Easier:
        # Delta = (a[i,j](b[y,x]-b[x,y]) + a[j,i](b[x,y]-b[y,x]))  # Cross terms
        #        + (a[i,i](b[y,y]-b[x,x]) + a[j,j](b[x,x]-b[y,y])) # Self terms
        #        + Sum_{k!=i,j} ( a[i,k](b[y,p[k]]-b[x,p[k]]) + a[k,i](b[p[k],y]-b[p[k],x])
        #                          + a[j,k](b[x,p[k]]-b[y,p[k]]) + a[k,j](b[p[k],x]-b[p[k],y]) )
        
        # We have Sum_i_all which includes k=i and k=j.
        # Sum_i_all = Term_ii + Term_ij_in_i + Sum_{k!=i,j} part_i
        # Sum_j_all = Term_jj + Term_ji_in_j + Sum_{k!=i,j} part_j
        
        # So Sum_{k!=i,j} part_i = Sum_i_all - Term_ii - Term_ij_in_i
        # Sum_{k!=i,j} part_j = Sum_j_all - Term_jj - Term_ji_in_j
        
        # Term_ii = a[i,i](b[y,x]-b[x,x]) + a[i,i](b[x,y]-b[x,x])  # Wait, p[i]=x.
        # In Sum_i, k=i term: a[i,i](b[y, p[i]] - b[x, p[i]]) + a[i,i](b[p[i], y] - b[p[i], x])
        # = a[i,i](b[y,x] - b[x,x]) + a[i,i](b[x,y] - b[x,x])
        
        # Term_ij_in_i (k=j in Sum_i):
        # a[i,j](b[y, p[j]] - b[x, p[j]]) + a[j,i](b[p[j], y] - b[p[j], x])
        # p[j]=y.
        # = a[i,j](b[y,y] - b[x,y]) + a[j,i](b[y,y] - b[y,x])
        
        # Term_jj = a[j,j](b[x,y] - b[y,y]) + a[j,j](b[y,x] - b[y,y])
        
        # Term_ji_in_j (k=i in Sum_j):
        # a[j,i](b[x, p[i]] - b[y, p[i]]) + a[i,j](b[p[i], x] - b[p[i], y])
        # p[i]=x.
        # = a[j,i](b[x,x] - b[y,x]) + a[i,j](b[x,x] - b[x,y])
        
        # Delta = Cross + Self + (Sum_i_all - Term_ii - Term_ij_in_i) + (Sum_j_all - Term_jj - Term_ji_in_j)
        
        cross = af[i,j]*(bf[y,x]-bf[x,y]) + af[j,i]*(bf[x,y]-bf[y,x])
        self_i = af[i,i]*(bf[y,y]-bf[x,x])
        self_j = af[j,j]*(bf[x,x]-bf[y,y])
        
        term_ii = af[i,i]*(bf[y,x] - bf[x,x]) + af[i,i]*(bf[x,y] - bf[x,x])
        term_ij_in_i = af[i,j]*(bf[y,y] - bf[x,y]) + af[j,i]*(bf[y,y] - bf[y,x])
        term_jj = af[j,j]*(bf[x,y] - bf[y,y]) + af[j,j]*(bf[y,x] - bf[y,y])
        term_ji_in_j = af[j,i]*(bf[x,x] - bf[y,x]) + af[i,j]*(bf[x,x] - bf[x,y])
        
        delta = cross + self_i + self_j + (sum_i - term_ii - term_ij_in_i) + (sum_j - term_jj - term_ji_in_j)
        
        # Acceptance
        if delta < 0 or (T > 0 and random.random() < np.exp(delta / T)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cooling
        T *= cooling
        if T < 1.0:
            T = 1.0 # or break? Keep at 1.0 to allow some exploration
            
    return best_p.tolist()
