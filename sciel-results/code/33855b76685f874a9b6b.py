import random
import time

def construct():
    n = 20
    # Bitset representation for adjacency
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    
    def is_valid_addition(adj, u, v):
        # Check if adding (u, v) violates the K2,3 constraint
        # 1. Common neighbors of u and v must be <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 2. For any neighbor x of u, common neighbors of v and x must be <= 1
        #    Because if we add (u,v), then for any x in N(u), 
        #    N(v) and N(x) will now share u. 
        #    Wait, the condition is "no two vertices have three common neighbours".
        #    If we add (u,v):
        #    - Pair (u,v): common neighbors are N(u) \ {v} intersect N(v) \ {u}. 
        #      Since it's a simple graph, u not in N(v) initially, v not in N(u).
        #      So common neighbors of u and v after addition is (N(u) & N(v)).
        #      We need |N(u) & N(v)| <= 2.
        #    - Pair (v, x) for x in N(u):
        #      Common neighbors of v and x after addition:
        #      N(v) & N(x). Note that u is in N(x) and u will be in N(v).
        #      So u is a common neighbor.
        #      Are there others? Yes, N(v) & N(x) might have others.
        #      The total common neighbors of v and x will be (N(v) & N(x)) | {u} ?
        #      No, N(v) & N(x) is the set of common neighbors *before* adding (u,v).
        #      After adding (u,v), u becomes a neighbor of v. u is already a neighbor of x.
        #      So u becomes a common neighbor of v and x.
        #      Does u count towards the existing common neighbors?
        #      u is not in N(v) before, so u was not in N(v) & N(x).
        #      So the new set of common neighbors for (v, x) is (N(v) & N(x)) U {u}.
        #      We need |(N(v) & N(x)) U {u}| <= 2.
        #      Since u is not in N(v) & N(x), this is |N(v) & N(x)| + 1 <= 2 => |N(v) & N(x)| <= 1.
        #    - Similarly for (u, y) for y in N(v):
        #      New common neighbors of u and y: (N(u) & N(y)) U {v}.
        #      We need |N(u) & N(y)| + 1 <= 2 => |N(u) & N(y)| <= 1.
        
        # So checks:
        # 1. |N(u) & N(v)| <= 2
        # 2. For all x in N(u): |N(v) & N(x)| <= 1
        # 3. For all y in N(v): |N(u) & N(y)| <= 1
        
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Check 1
        if (mask_u & mask_v).bit_count() > 2:
            return False
            
        # Check 2: For all x in N(u), |N(v) & N(x)| <= 1
        m = mask_u
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            if (mask_v & adj[x]).bit_count() > 1:
                return False
                
        # Check 3: For all y in N(v), |N(u) & N(y)| <= 1
        m = mask_v
        while m:
            lsb = m & -m
            y = lsb.bit_length() - 1
            m ^= lsb
            if (mask_u & adj[y]).bit_count() > 1:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Initialize with a greedy graph to give a good starting point
    adj = [0] * n
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.seed(42)
    random.shuffle(pairs)
    for u, v in pairs:
        if is_valid_addition(adj, u, v):
            add_edge(adj, u, v)
            
    current_edges = sum(adj[i].bit_count() for i in range(n)) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < timeout:
        # Try to add a random edge
        # Pick random u, v
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if adj[u] & (1 << v):
            # Edge exists, try to remove and swap? Or just skip
            # For simplicity, let's try a "kick" if we are stuck
            pass
        else:
            if is_valid_addition(adj, u, v):
                add_edge(adj, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    T = 10.0 # Reset temperature on improvement
            else:
                # Edge invalid. 
                # If we are stuck (no improvements recently), we might need to remove an edge.
                # Simple heuristic: with small probability, remove a random edge to escape local max
                if random.random() < 0.05:
                    # Pick a random existing edge to remove
                    # Find a random edge
                    edges_list = []
                    for i in range(n):
                        m = adj[i]
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            m ^= lsb
                            if i < j:
                                edges_list.append((i, j))
                    if edges_list:
                        ru, rv = random.choice(edges_list)
                        remove_edge(adj, ru, rv)
                        current_edges -= 1
                        T = max(T * 0.9, 0.1) # Cool down? Or keep T?
                        
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # Convert best_adj to list of edges
    best_edges_list = []
    for u in range(n):
        m = best_adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                best_edges_list.append([u, v])
                
    return best_edges_list
