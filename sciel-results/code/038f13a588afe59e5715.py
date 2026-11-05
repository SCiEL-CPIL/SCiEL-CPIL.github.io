"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """Construct a graph with no K5 and no independent set of size 5."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    k = 5
    
    # Helper to find a K5 or Independent Set of size 5
    # Returns (type, vertices) where type is 'clique' or 'independent'
    # vertices is a list of 5 ints
    def find_violation(adj, n):
        # adj is list of bitmasks
        # Check for K5
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Iterate w in common
                    w = 0
                    while common:
                        w_bit = common & -common
                        w = w_bit.bit_length() - 1
                        common ^= w_bit
                        
                        common2 = common & adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                        x = 0
                        while common2:
                            x_bit = common2 & -common2
                            x = x_bit.bit_length() - 1
                            common2 ^= x_bit
                            
                            common3 = common2 & adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                            if common3:
                                y_bit = common3 & -common3
                                y = y_bit.bit_length() - 1
                                return ('clique', [u, v, w, x, y])
        
        # Check for Independent Set of size 5
        # non_adj[i] has bits for j where j != i and not connected
        non_adj = [0] * n
        for i in range(n):
            # non_adj[i] should contain all j != i such that not (adj[i] & (1<<j))
            # mask of all vertices
            all_mask = (1 << n) - 1
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
            
        for u in range(n):
            for v in range(u + 1, n):
                if non_adj[u] & (1 << v):
                    common = non_adj[u] & non_adj[v]
                    w = 0
                    while common:
                        w_bit = common & -common
                        w = w_bit.bit_length() - 1
                        common ^= w_bit
                        
                        common2 = common & non_adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                        x = 0
                        while common2:
                            x_bit = common2 & -common2
                            x = x_bit.bit_length() - 1
                            common2 ^= x_bit
                            
                            common3 = common2 & non_adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                            if common3:
                                y_bit = common3 & -common3
                                y = y_bit.bit_length() - 1
                                return ('independent', [u, v, w, x, y])
                                
        return None

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    best_n = q
    best_adj = adj[:]
    
    # Try to grow
    for target_n in range(current_n + 1, 47):
        if time.time() > deadline - 30:
            break
            
        success = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Create new graph
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj[i]
            
            # Connect new vertex (index target_n - 1)
            new_idx = target_n - 1
            for i in range(current_n):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)
            
            # Local Search
            local_deadline = time.time() + 2.0
            valid = False
            while time.time() < local_deadline:
                viol = find_violation(new_adj, target_n)
                if viol is None:
                    valid = True
                    break
                
                vtype, verts = viol
                if vtype == 'clique':
                    # Remove an edge. Pick one randomly from the clique?
                    # Or pick one that seems "strongest"?
                    # Let's pick a random pair from the clique to remove
                    u, v = random.sample(verts, 2)
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    # Add an edge
                    u, v = random.sample(verts, 2)
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
            
            if valid:
                current_n = target_n
                adj = new_adj
                best_n = current_n
                best_adj = adj[:]
                success = True
                break
        
        if not success:
            # If we couldn't grow to this N, try next N?
            # Usually if we can't get N, we can't get N+1 easily.
            # But maybe a different random seed works?
            # Let's stop growing if we fail a few times to save time for verification?
            # The task says "as many vertices as possible".
            # If we fail to get 40, we might still have 39.
            # We should keep the best found so far.
            # Let's try to see if we can get back to a valid state for current_n?
            # Actually, if we failed to grow, current_n is still the previous valid one.
            # We can break or continue trying higher N with different randoms?
            # It's unlikely to jump over a failed N.
            pass

    # Final verification of best_adj
    if find_violation(best_adj, best_n) is not None:
        # Should not happen, but fallback to Paley
        best_n = 37
        best_adj = [0] * 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)

    return best_n, get_edges(best_adj, best_n)

# EVOLVE-BLOCK-END
