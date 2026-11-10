import random
import itertools
import time
import math

def construct():
    n = 20
    
    # Precompute all triples for potential use, but we will use localized checks
    # Bitmask representation: adj[i] is an integer where bit j is set if (i,j) is an edge
    
    def popcount(x):
        return bin(x).count('1')
        
    def check_validity_local(adj, u, v, old_u, old_v):
        """
        Check if adding edge (u,v) to a graph with old adjacencies old_u, old_v
        results in a K3,3.
        We only need to check triples that include u or v.
        A triple {a,b,c} is bad if |N(a) & N(b) & N(c)| >= 3.
        Adding edge (u,v) changes N(u) and N(v).
        Any new violation must involve u or v in the triple, because only N(u) and N(v) changed.
        Specifically, if u is in the triple, the intersection might include v (newly added to N(u)'s neighbors? No, v is in N(u)).
        Wait, N(u) is the set of neighbors of u. If we add edge (u,v), v becomes a neighbor of u.
        So for any triple containing u, the common neighbors are the intersection of N(u), N(x), N(y).
        Since v is now in N(u), v might be in the intersection if v is also in N(x) and N(y).
        So we check all triples containing u or v.
        """
        # New adjacencies
        new_u = adj[u] | (1 << v)
        new_v = adj[v] | (1 << u)
        
        # We need to check all triples {a, b, c} such that u in {a,b,c} or v in {a,b,c}.
        # We can iterate over all pairs (x,y) from remaining vertices.
        # Case 1: Triple is {u, x, y}
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                # Check triple {u, x, y}
                # N(u) is new_u. N(x) is adj[x]. N(y) is adj[y].
                common = new_u & adj[x] & adj[y]
                if popcount(common) >= 3:
                    return False
        # Case 2: Triple is {v, x, y}
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                # Check triple {v, x, y}
                # N(v) is new_v. N(x) is adj[x]. N(y) is adj[y].
                common = new_v & adj[x] & adj[y]
                if popcount(common) >= 3:
                    return False
        # Case 3: Triple is {u, v, x}
        for x in range(n):
            if x == u or x == v: continue
            # Check triple {u, v, x}
            # N(u) is new_u. N(v) is new_v. N(x) is adj[x].
            common = new_u & new_v & adj[x]
            if popcount(common) >= 3:
                return False
        return True

    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def is_valid_global(adj):
        # Full check for initialization or verification
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if popcount(common) >= 3:
                        return False
        return True

    def local_search(adj, steps=5000):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for step in range(steps):
            # Choose move: Add, Remove, Swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Add edge
                u, v = random.choice(all_edges)
                if not (current_adj[u] & (1 << v)):
                    if check_validity_local(current_adj, u, v, current_adj[u], current_adj[v]):
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = current_adj[:]
                    else:
                        # Reject
                        pass
            elif move_type < 0.7:
                # Remove edge
                u, v = random.choice(all_edges)
                if (current_adj[u] & (1 << v)):
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges -= 1
                    # Always valid to remove
            else:
                # Swap: Add (u,v), Remove (x,y)
                # Pick random missing edge
                u, v = random.choice(all_edges)
                if (current_adj[u] & (1 << v)):
                    continue
                
                # Pick random existing edge
                x = random.randint(0, n-1)
                neighbors_x = current_adj[x]
                if neighbors_x == 0:
                    continue
                # Pick a random set bit
                # Use a simple method: find a random index and check if bit is set, or use a list
                # For n=20, we can just iterate
                candidates = [i for i in range(n) if (neighbors_x >> i) & 1]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                # Perform swap
                old_u = current_adj[u]
                old_v = current_adj[v]
                old_x = current_adj[x]
                old_y = current_adj[y]
                
                # Add u-v
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                # Remove x-y
                current_adj[x] &= ~(1 << y)
                current_adj[y] &= ~(1 << x)
                
                # Check validity
                # This is complex to localize perfectly for a swap.
                # However, the only vertices whose neighborhoods changed are u, v, x, y.
                # Any new violation must involve at least one of u, v, x, y.
                # We can do a partial check on triples involving {u,v,x,y}.
                # Or just do a global check if the swap is promising?
                # Given the low n, a global check on 1140 triples is ~1140 * 3 ANDs + popcount.
                # 1140 * 3 = 3420 ops. This is fast enough in Python if done in C (bit ops are C).
                # Let's just do a global check for swaps to be safe and simple, 
                # as the number of swaps is limited by the 30% probability.
                
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            common = current_adj[i] & current_adj[j] & current_adj[k]
                            if popcount(common) >= 3:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if not valid:
                    # Revert
                    current_adj[u] = old_u
                    current_adj[v] = old_v
                    current_adj[x] = old_x
                    current_adj[y] = old_y

        return best_adj, best_edges

    best_adj = None
    best_edges = 0
    
    # Try multiple random starts
    # Start with a sparse random graph to ensure validity, then densify
    for trial in range(20):
        random.seed(trial)
        adj = [0] * n
        
        # Greedy initialization with random order
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                if check_validity_local(adj, u, v, adj[u], adj[v]):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local search
        res_adj, res_edges = local_search(adj, steps=3000)
        
        if res_edges > best_edges:
            best_edges = res_edges
            best_adj = res_adj

    if best_adj is None:
        return []

    # Convert to list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
    return edges
