"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Initial Heuristic: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit (5s buffer)
    
    # Simulated Annealing parameters
    T = 1000.0
    T_min = 1.0
    alpha = 0.9998
    
    # Precompute for faster delta calculation
    # We will use the O(n) delta formula for swaps.
    
    while True:
        if time.time() - start_time > timeout:
            break
        
        # Perform swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        x = p[i]
        y = p[j]
        
        # Calculate delta for swapping p[i] and p[j]
        # Let x = p[i], y = p[j].
        # We swap locations x and y for facilities i and j.
        
        # The change in cost is:
        # Delta = sum_{k != i, j} [ a[i,k](b[y,p[k]] - b[x,p[k]]) + a[k,i](b[p[k],y] - b[p[k],x])
        #                         + a[j,k](b[x,p[k]] - b[y,p[k]]) + a[k,j](b[p[k],x] - b[p[k],y]) ]
        #       + a[i,j](b[y,x] - b[x,y]) + a[j,i](b[x,y] - b[y,x])
        #       - (a[i,j]b[x,y] + a[j,i]b[y,x])  # Wait, the terms above already include the change for k=i,j?
        # Let's be precise.
        # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Terms involving i or j:
        # 1. u=i, v=j: a[i,j] b[x,y] -> a[i,j] b[y,x]. Change: a[i,j](b[y,x]-b[x,y])
        # 2. u=j, v=i: a[j,i] b[y,x] -> a[j,i] b[x,y]. Change: a[j,i](b[x,y]-b[y,x])
        # 3. u=i, v=k (k!=i,j): a[i,k] b[x,p[k]] -> a[i,k] b[y,p[k]]. Change: a[i,k](b[y,p[k]]-b[x,p[k]])
        # 4. u=k, v=i (k!=i,j): a[k,i] b[p[k],x] -> a[k,i] b[p[k],y]. Change: a[k,i](b[p[k],y]-b[p[k],x])
        # 5. u=j, v=k (k!=i,j): a[j,k] b[y,p[k]] -> a[j,k] b[x,p[k]]. Change: a[j,k](b[x,p[k]]-b[y,p[k]])
        # 6. u=k, v=j (k!=i,j): a[k,j] b[p[k],y] -> a[k,j] b[p[k],x]. Change: a[k,j](b[p[k],x]-b[p[k],y])
        
        # We can vectorize this.
        # Let's define vectors:
        # row_i = a[i, :]
        # col_i = a[:, i]
        # row_j = a[j, :]
        # col_j = a[:, j]
        
        # We need sums over k != i, j.
        # Sum_3 = sum_{k!=i,j} a[i,k] (b[y,p[k]] - b[x,p[k]])
        # This is dot(a[i,:], b[y,p] - b[x,p]) - a[i,i](b[y,p[i]]-b[x,p[i]]) - a[i,j](b[y,p[j]]-b[x,p[j]])
        # Note p[i]=x, p[j]=y.
        # So term for k=i: a[i,i](b[y,x]-b[x,x])
        # Term for k=j: a[i,j](b[y,y]-b[x,y])
        
        # Sum_4 = sum_{k!=i,j} a[k,i] (b[p[k],y] - b[p[k],x])
        # This is dot(a[:,i], b[p,y] - b[p,x]) - a[i,i](b[x,y]-b[x,x]) - a[j,i](b[y,y]-b[y,x])
        
        # Sum_5 = sum_{k!=i,j} a[j,k] (b[x,p[k]] - b[y,p[k]])
        # This is -Sum_3 structure but for j.
        # dot(a[j,:], b[x,p] - b[y,p]) - a[j,i](b[x,x]-b[y,x]) - a[j,j](b[x,y]-b[y,y])
        
        # Sum_6 = sum_{k!=i,j} a[k,j] (b[p[k],x] - b[p[k],y])
        # This is -Sum_4 structure but for j.
        # dot(a[:,j], b[p,x] - b[p,y]) - a[i,j](b[x,x]-b[x,y]) - a[j,j](b[y,x]-b[y,y])
        
        # Let's compute the full dot products first.
        # V_y = b[y, p]
        # V_x = b[x, p]
        # W_y = b[p, y]
        # W_x = b[p, x]
        
        V_y = b[y, p]
        V_x = b[x, p]
        W_y = b[p, y]
        W_x = b[p, x]
        
        # Full sums (including k=i and k=j)
        s3_full = np.dot(a[i, :], V_y - V_x)
        s4_full = np.dot(a[:, i], W_y - W_x)
        s5_full = np.dot(a[j, :], V_x - V_y)
        s6_full = np.dot(a[:, j], W_x - W_y)
        
        # Subtract terms for k=i and k=j
        # For s3 (u=i, v=k):
        # k=i term: a[i,i] (b[y, x] - b[x, x])
        # k=j term: a[i,j] (b[y, y] - b[x, y])
        s3 = s3_full - a[i, i] * (b[y, x] - b[x, x]) - a[i, j] * (b[y, y] - b[x, y])
        
        # For s4 (u=k, v=i):
        # k=i term: a[i,i] (b[x, y] - b[x, x])
        # k=j term: a[j,i] (b[y, y] - b[y, x])
        s4 = s4_full - a[i, i] * (b[x, y] - b[x, x]) - a[j, i] * (b[y, y] - b[y, x])
        
        # For s5 (u=j, v=k):
        # k=i term: a[j,i] (b[x, x] - b[y, x])
        # k=j term: a[j,j] (b[x, y] - b[y, y])
        s5 = s5_full - a[j, i] * (b[x, x] - b[y, x]) - a[j, j] * (b[x, y] - b[y, y])
        
        # For s6 (u=k, v=j):
        # k=i term: a[i,j] (b[x, x] - b[x, y])
        # k=j term: a[j,j] (b[y, x] - b[y, y])
        s6 = s6_full - a[i, j] * (b[x, x] - b[x, y]) - a[j, j] * (b[y, x] - b[y, y])
        
        # Cross terms (u=i, v=j) and (u=j, v=i)
        cross = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
        
        delta = s3 + s4 + s5 + s6 + cross
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        new_cost = current_cost + delta
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
        else:
            p[i], p[j] = p[j], p[i]
            # current_cost remains unchanged
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
