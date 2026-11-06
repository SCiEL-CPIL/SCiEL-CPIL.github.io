import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    
    # Precompute edge bits for fast lookup
    # edge_mask[u] is a bitmask of neighbors of u
    # We will represent the graph as a list of masks: adj_mask[u]
    
    def is_k4_free(adj_mask):
        # Check for K4
        # For each pair (u, v), if edge exists, check common neighbors.
        # If two common neighbors w, x have an edge, it's a K4.
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                # Check if any edge exists between vertices in common
                # Iterate through set bits in common
                c1 = common
                while c1:
                    lsb1 = c1 & -c1
                    w = lsb1.bit_length() - 1
                    c1 ^= lsb1
                    # Check if w connects to any other in common
                    # neighbors of w in common
                    common_neighbors_of_w = adj_mask[w] & common
                    if common_neighbors_of_w:
                        return False
        return True

    def max_triangle_free_size(adj_mask):
        # Find largest subset S such that G[S] is triangle-free.
        # Recursive B&B on bitmasks.
        
        # Precompute triangles? 
        # Or generate on the fly?
        # Generating triangles for a given mask S:
        # Iterate u in S, v in S > u, if edge(u,v), then common = adj[u] & adj[v] & S & ~(1<<u) & ~(1<<v).
        # Any w in common forms a triangle (u,v,w).
        
        best_size = [0]
        
        def count_bits(x):
            return bin(x).count('1')
            
        def dfs(mask):
            # Pruning: if mask has <= best_size bits, return
            if count_bits(mask) <= best_size[0]:
                return
            
            # Check if mask is triangle-free
            # If yes, update best and return
            is_tf = True
            # Find a triangle to branch on
            branch_triangle = None
            
            # Iterate u in mask
            m_u = mask
            while m_u:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                # Iterate v in mask > u
                m_v = mask & ~(lsb_u - 1) # bits higher than u? No, mask & ~(1<<u) & ~((1<<u)-1)
                # Simpler: just iterate all v > u in mask
                # To be efficient, we can just check all pairs
                pass
            
            # Let's do a simple scan for a triangle
            # If no triangle found, it's triangle free
            found_triangle = None
            m_u = mask
            while m_u:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                # v must be in mask and v > u
                m_v = mask & ~((1 << (u + 1)) - 1)
                while m_v:
                    lsb_v = m_v & -m_v
                    v = lsb_v.bit_length() - 1
                    m_v ^= lsb_v
                    
                    if not (adj_mask[u] >> v) & 1:
                        continue
                    
                    # Common neighbors of u and v in mask
                    common = adj_mask[u] & adj_mask[v] & mask
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    if common:
                        # Found triangle (u, v, w) where w is any bit in common
                        lsb_w = common & -common
                        w = lsb_w.bit_length() - 1
                        found_triangle = (u, v, w)
                        break
                if found_triangle:
                    break
            
            if not found_triangle:
                size = count_bits(mask)
                if size > best_size[0]:
                    best_size[0] = size
                return
            
            # Branch: remove u, v, or w
            u, v, w = found_triangle
            dfs(mask & ~(1 << u))
            dfs(mask & ~(1 << v))
            dfs(mask & ~(1 << w))
            
        dfs((1 << n) - 1)
        return best_size[0]

    def random_k4_free_graph():
        adj_mask = [0] * n
        # Start with empty graph
        # Add random edges, remove if K4 created
        # Or start with random and repair
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        while not is_k4_free(adj_mask):
            # Find a K4
            # Reuse logic from is_k4_free but return the K4
            k4_found = None
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] >> v) & 1:
                        continue
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c1 = common
                    while c1:
                        lsb1 = c1 & -c1
                        w = lsb1.bit_length() - 1
                        c1 ^= lsb1
                        common_neighbors_of_w = adj_mask[w] & common
                        if common_neighbors_of_w:
                            lsb2 = common_neighbors_of_w & -common_neighbors_of_w
                            x = lsb2.bit_length() - 1
                            k4_found = (u, v, w, x)
                            break
                    if k4_found:
                        break
                if k4_found:
                    break
            
            if not k4_found:
                break
                
            u, v, w, x = k4_found
            # Remove a random edge from this K4
            edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            e = random.choice(edges)
            a, b = e
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
        return adj_mask

    best_graph = None
    best_score = 18
    
    # Initial random search
    for i in range(20):
        if time.time() > deadline: break
        g = random_k4_free_graph()
        score = max_triangle_free_size(g)
        if score < best_score:
            best_score = score
            best_graph = g[:]
            
    # Simulated Annealing / Local Search
    if best_graph:
        current_graph = best_graph[:]
        current_score = best_score
        T = 1.0
        cooling = 0.99
        
        while time.time() < deadline - 5:
            T *= cooling
            if T < 0.01:
                break
                
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            # Flip edge
            has_edge = (current_graph[u] >> v) & 1
            if has_edge:
                current_graph[u] &= ~(1 << v)
                current_graph[v] &= ~(1 << u)
            else:
                current_graph[u] |= (1 << v)
                current_graph[v] |= (1 << u)
                
            if not is_k4_free(current_graph):
                # Revert
                if has_edge:
                    current_graph[u] |= (1 << v)
                    current_graph[v] |= (1 << u)
                else:
                    current_graph[u] &= ~(1 << v)
                    current_graph[v] &= ~(1 << u)
                continue
                
            new_score = max_triangle_free_size(current_graph)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_graph = current_graph[:]
            else:
                # Revert
                if has_edge:
                    current_graph[u] |= (1 << v)
                    current_graph[v] |= (1 << u)
                else:
                    current_graph[u] &= ~(1 << v)
                    current_graph[v] &= ~(1 << u)
                    
    if best_graph is None:
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_graph[u] >> v) & 1:
                edges.append([u, v])
    return edges
