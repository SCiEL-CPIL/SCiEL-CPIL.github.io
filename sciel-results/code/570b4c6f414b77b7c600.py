import random
import time

def construct():
    n = 20
    deadline = time.time() + 110
    
    def is_k33_free(adj):
        # Check all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Common neighbors of i, j, k
                    cn = adj[i] & adj[j] & adj[k]
                    if cn.bit_count() >= 3:
                        return False
        return True

    best_edges = 0
    best_graph = None
    
    while time.time() < deadline:
        # Greedy initialization
        adj = [0] * n
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        current_edges = 0
        for u, v in edges_to_try:
            # Fast check for addition
            # Only check triples involving u or v
            valid = True
            for x in range(n):
                if x == u: continue
                for y in range(x+1, n):
                    if y == u: continue
                    if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
                        if (adj[u] & adj[x] & adj[y]).bit_count() >= 2:
                            valid = False
                            break
                if not valid: break
            if not valid:
                for x in range(n):
                    if x == v: continue
                    for y in range(x+1, n):
                        if y == v: continue
                        if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
                            if (adj[v] & adj[x] & adj[y]).bit_count() >= 2:
                                valid = False
                                break
                    if not valid: break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = adj[:]
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swaps
            # Get list of edges and non-edges
            edge_list = []
            non_edge_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        edge_list.append((u, v))
                    else:
                        non_edge_list.append((u, v))
            
            random.shuffle(edge_list)
            random.shuffle(non_edge_list)
            
            for e_new in non_edge_list:
                if time.time() > deadline: break
                u_new, v_new = e_new
                # Try to add e_new
                # We need to check if it's valid.
                # If not, we need to remove an edge.
                # First, check if adding e_new alone is valid.
                valid_add = True
                # Re-use fast check logic
                for x in range(n):
                    if x == u_new: continue
                    for y in range(x+1, n):
                        if y == u_new: continue
                        if (adj[x] >> v_new) & 1 and (adj[y] >> v_new) & 1:
                            if (adj[u_new] & adj[x] & adj[y]).bit_count() >= 2:
                                valid_add = False
                                break
                    if not valid_add: break
                if not valid_add:
                    for x in range(n):
                        if x == v_new: continue
                        for y in range(x+1, n):
                            if y == v_new: continue
                            if (adj[x] >> u_new) & 1 and (adj[y] >> u_new) & 1:
                                if (adj[v_new] & adj[x] & adj[y]).bit_count() >= 2:
                                    valid_add = False
                                    break
                        if not valid_add: break
                
                if valid_add:
                    # Add it
                    adj[u_new] |= (1 << v_new)
                    adj[v_new] |= (1 << u_new)
                    current_edges += 1
                    improved = True
                    # Update lists? Too expensive. Just restart inner loop or continue.
                    # Since we added an edge, non_edge_list is stale.
                    # Break to re-shuffle?
                    break 
                
                # If not valid, try swapping with a random edge
                # Pick a random edge to remove
                e_old = random.choice(edge_list)
                u_old, v_old = e_old
                
                # Remove e_old
                adj[u_old] &= ~(1 << v_old)
                adj[v_old] &= ~(1 << u_old)
                # Add e_new
                adj[u_new] |= (1 << v_new)
                adj[v_new] |= (1 << u_new)
                
                # Check full validity
                if is_k33_free(adj):
                    # Accept swap
                    # Edge count remains same
                    improved = True
                    # Update lists?
                    break
                else:
                    # Revert
                    adj[u_old] |= (1 << v_old)
                    adj[v_old] |= (1 << u_old)
                    adj[u_new] &= ~(1 << v_new)
                    adj[v_new] &= ~(1 << u_new)
                    
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = adj[:]
            
    # Convert best_graph to list of edges
    if best_graph is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    res = []
    for u in range(n):
        mask = best_graph[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                res.append([u, v])
            mask ^= lsb
    return res
