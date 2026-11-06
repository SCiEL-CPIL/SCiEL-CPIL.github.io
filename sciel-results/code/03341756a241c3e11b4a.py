import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Try n from 35 down to 33
    for n in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Use bitsets for adjacency
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_adj = adj[:]
        best_cost = float('inf')
        
        # Simulated Annealing
        T = 100.0
        T_min = 0.01
        cooling = 0.99
        
        while T > T_min and time.time() < deadline:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Calculate current cost
            curr_cost = count_c5(n, adj)
            
            # Flip edge
            if adj[u] & (1 << v):
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_cost = count_c5(n, adj)
            
            # Acceptance
            if new_cost < curr_cost or random.random() < np.exp((curr_cost - new_cost) / T):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                # If cost is 0, check alpha
                if new_cost == 0:
                    alpha = calc_alpha(n, adj)
                    if alpha < 9:
                        # Convert to edge list
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            T *= cooling
            
        # If SA didn't find a solution, try a few more restarts if time permits
        if time.time() < deadline - 10:
             # Restart with different p
             for _ in range(5):
                 if time.time() > deadline:
                     break
                 adj = [0] * n
                 p = random.uniform(0.3, 0.7)
                 for i in range(n):
                     for j in range(i + 1, n):
                         if random.random() < p:
                             adj[i] |= (1 << j)
                             adj[j] |= (1 << i)
                 
                 # Quick local search
                 for _ in range(1000):
                     if time.time() > deadline:
                         break
                     u = random.randint(0, n - 1)
                     v = random.randint(0, n - 1)
                     if u == v:
                         continue
                     curr_cost = count_c5(n, adj)
                     if adj[u] & (1 << v):
                         adj[u] &= ~(1 << v)
                         adj[v] &= ~(1 << u)
                     else:
                         adj[u] |= (1 << v)
                         adj[v] |= (1 << u)
                     new_cost = count_c5(n, adj)
                     if new_cost < curr_cost:
                         if new_cost == 0:
                             alpha = calc_alpha(n, adj)
                             if alpha < 9:
                                 edges = []
                                 for i in range(n):
                                     for j in range(i + 1, n):
                                         if adj[i] & (1 << j):
                                             edges.append([i, j])
                                 return {'n': n, 'edges': edges}
                     else:
                         # Revert
                         if adj[u] & (1 << v):
                             adj[u] &= ~(1 << v)
                             adj[v] &= ~(1 << u)
                         else:
                             adj[u] |= (1 << v)
                             adj[v] |= (1 << u)

    # Fallback to known good solution (n=32)
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}

def count_c5(n, adj):
    """Count number of 5-cycles. Returns 0 if none."""
    # For each edge (u,v), check for path u-a-b-c-v
    # This counts each C5 multiple times, but we just need to know if > 0
    # Actually, we want to minimize the count.
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            # u and v are connected
            # Neighbors of u (excluding v)
            nu = adj[u] & ~(1 << v)
            # Neighbors of v (excluding u)
            nv = adj[v] & ~(1 << u)
            
            # For each a in nu, c in nv
            # We need b in N(a) & N(c)
            # b not in {u, v, a, c}
            
            # Iterate over bits in nu
            a = 0
            nu_mask = nu
            while nu_mask:
                # Get lowest set bit
                lsb = nu_mask & -nu_mask
                a = lsb.bit_length() - 1
                nu_mask ^= lsb
                
                # For each c in nv
                c = 0
                nv_mask = nv
                while nv_mask:
                    lsb_c = nv_mask & -nv_mask
                    c = lsb_c.bit_length() - 1
                    nv_mask ^= lsb_c
                    
                    if a == c:
                        continue
                        
                    # Common neighbors of a and c
                    common = adj[a] & adj[c]
                    # Exclude u, v, a, c
                    mask_out = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                    common &= ~mask_out
                    
                    if common:
                        # Found a C5
                        # We can return 1 early if we just want to know existence
                        # But for SA, we want a score.
                        # Let's just return 1 if found, to keep it fast?
                        # No, better to count or return 1.
                        # If we return 1, SA might get stuck.
                        # Let's return the number of such pairs (a,c) that work.
                        count += 1
                        # Early exit for speed?
                        # If count is large, it's bad.
                        # Let's cap it.
                        if count > 100:
                            return count
    return count

def calc_alpha(n, adj):
    """Calculate independence number exactly using branch and bound."""
    # Convert adj to sets for easier manipulation?
    # Or use bitsets.
    # We want max subset S such that no two vertices in S are adjacent.
    # This is equivalent to max clique in complement graph.
    # Complement graph: edge (u,v) if NOT adj[u] & (1<<v)
    
    # Use a simple recursive search with pruning
    # Candidate set C, current size S
    # Upper bound: |C| + S
    # If upper bound <= best, prune.
    
    best = [0]
    
    # Order vertices by degree in complement?
    # For now, just use 0..n-1
    
    def dfs(cand, size):
        if size + len(cand) <= best[0]:
            return
        if not cand:
            if size > best[0]:
                best[0] = size
            return
            
        # Pick a vertex v from cand
        # Heuristic: pick vertex with max degree in complement induced by cand?
        # Or just first one.
        v = cand[0]
        rest = cand[1:]
        
        # Option 1: Include v
        # New candidates: neighbors of v in complement, which are non-neighbors in original
        # Non-neighbors of v in original: ~(adj[v]) & mask(cand)
        # But we need to intersect with rest
        # mask_rest = bitmask of rest
        # new_cand = [x for x in rest if not (adj[v] & (1 << x))]
        
        new_cand = []
        for x in rest:
            if not (adj[v] & (1 << x)):
                new_cand.append(x)
        
        dfs(new_cand, size + 1)
        
        # Option 2: Exclude v
        dfs(rest, size)
        
    dfs(list(range(n)), 0)
    return best[0]

# EVOLVE-BLOCK-END
