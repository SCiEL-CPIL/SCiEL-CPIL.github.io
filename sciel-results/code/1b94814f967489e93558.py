import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 105  # safety margin

    # Bitmask adjacency: adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n

    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def has_edge(u, v):
        return (adj[u] >> v) & 1

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_add_valid(u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        Adding (u,v) can only create a K3,3 where u and v are in the same part.
        So for each w != u,v, check if N(u) ∩ N(v) ∩ N(w) has size >= 3.
        """
        if has_edge(u, v):
            return True
        # Common neighbors of u and v (excluding each other)
        common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        # For each w that is a common neighbor of u and v, check if w shares >= 3 common neighbors with u and v
        # Actually: K3,3 with parts {u,v,w} and {x,y,z} where x,y,z in N(u)∩N(v)∩N(w)
        # So we need: for some w, |N(u) ∩ N(v) ∩ N(w)| >= 3
        # But w must also be connected to x,y,z, which is automatic if x,y,z in N(w)
        temp = common_uv
        while temp:
            lb = temp & (-temp)
            w = lb.bit_length() - 1
            temp ^= lb
            # Count common neighbors of u, v, w
            common_uvw = adj[u] & adj[v] & adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            if bin(common_uvw).count('1') >= 3:
                return False
        return True

    def find_violation():
        """Find a K3,3 in the current graph. Returns list of 9 edges or None."""
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if bin(common).count('1') >= 3:
                        # Get 3 common neighbors
                        temp = common
                        b_list = []
                        while len(b_list) < 3:
                            lb = temp & (-temp)
                            b = lb.bit_length() - 1
                            b_list.append(b)
                            temp ^= lb
                        b1, b2, b3 = b_list
                        return [(i, b1), (i, b2), (i, b3),
                                (j, b1), (j, b2), (j, b3),
                                (k, b1), (k, b2), (k, b3)]
        return None

    def get_edges_from_adj(adj_list):
        edges = []
        for i in range(n):
            temp = adj_list[i]
            while temp:
                lb = temp & (-temp)
                j = lb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                temp ^= lb
        return edges

    def is_valid_graph(adj_list):
        """Verify no K3,3 exists."""
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj_list[i] & adj_list[j] & adj_list[k]
                    if bin(common).count('1') >= 3:
                        return False
        return True

    # Initialize with a structured approach: start with random graph and clean
    random.seed(42)
    
    # Start with a moderately dense random graph
    p_init = 0.5
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p_init:
                add_edge(u, v)
    
    # Remove all K3,3 violations
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline - 30:
            break
        viol = find_violation()
        if viol is None:
            break
        # Remove a random edge from the violation
        e = random.choice(viol)
        remove_edge(e[0], e[1])
    
    best_edges = count_edges()
    best_adj = adj[:]
    
    # Local search with 1-edge add and 2-edge swap
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try adding a single edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or has_edge(u, v):
                no_improve += 1
                continue
            if is_add_valid(u, v):
                add_edge(u, v)
                new_count = count_edges()
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    remove_edge(u, v)
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.85:
            # Try 2-edge swap: remove edge e1, add edge e2
            existing_edges = []
            for i in range(n):
                temp = adj[i]
                while temp:
                    lb = temp & (-temp)
                    j = lb.bit_length() - 1
                    if i < j:
                        existing_edges.append((i, j))
                    temp ^= lb
            
            if not existing_edges:
                no_improve += 1
                continue
                
            e1 = random.choice(existing_edges)
            u1, v1 = e1
            
            # Pick a random non-edge to potentially add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                no_improve += 1
                continue
            if has_edge(u2, v2):
                no_improve += 1
                continue
            # Make sure it's not the same edge
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                no_improve += 1
                continue
            
            # Remove e1 temporarily
            remove_edge(u1, v1)
            
            # Check if adding e2 is valid
            if is_add_valid(u2, v2):
                add_edge(u2, v2)
                # Check if graph is still valid (no K3,3)
                if is_valid_graph(adj):
                    new_count = count_edges()
                    if new_count > best_edges:
                        best_edges = new_count
                        best_adj = adj[:]
                        no_improve = 0
                    else:
                        # Revert: remove e2, add back e1
                        remove_edge(u2, v2)
                        add_edge(u1, v1)
                        no_improve += 1
                else:
                    # Invalid, revert
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
                    no_improve += 1
            else:
                # Add back e1
                add_edge(u1, v1)
                no_improve += 1
        else:
            # Perturbation: remove a random edge
            existing_edges = []
            for i in range(n):
                temp = adj[i]
                while temp:
                    lb = temp & (-temp)
                    j = lb.bit_length() - 1
                    if i < j:
                        existing_edges.append((i, j))
                    temp ^= lb
            
            if existing_edges:
                e = random.choice(existing_edges)
                remove_edge(e[0], e[1])
                no_improve += 1
            else:
                no_improve += 1
    
    # Final polish: try to add any remaining valid edges (multiple passes)
    adj = best_adj[:]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(u, v):
                    if is_add_valid(u, v):
                        add_edge(u, v)
                        improved = True
    
    final_count = count_edges()
    if final_count > best_edges:
        best_adj = adj[:]
        best_edges = final_count
    
    # Convert to list of edges
    result = get_edges_from_adj(best_adj)
    return result
