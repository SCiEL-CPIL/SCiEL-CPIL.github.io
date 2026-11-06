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
        We simulate the addition and check for K3,3.
        """
        if has_edge(u, v):
            return True
        
        # Temporarily add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check for K3,3
        # A K3,3 exists if there are 3 vertices with >= 3 common neighbors.
        # We only need to check triples involving u or v, as other triples were already valid.
        # Triples: {u, i, j}, {v, i, j}, {u, v, w}
        
        # Helper to check if a triple (a,b,c) has >= 3 common neighbors
        def check_triple(a, b, c):
            common = adj[a] & adj[b] & adj[c]
            # Exclude a, b, c themselves from common neighbors count
            common &= ~(1 << a) & ~(1 << b) & ~(1 << c)
            return bin(common).count('1') >= 3
        
        # Check triples with u
        for i in range(n):
            if i == u: continue
            for j in range(i + 1, n):
                if j == u: continue
                if check_triple(u, i, j):
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return False
        
        # Check triples with v (but not u, as {u,v,i} is covered above if i!=u,v)
        # Wait, {u, v, i} was covered in the first loop if i != u, v?
        # In first loop: i runs 0..n-1. j runs i+1..n-1.
        # If i=u, skipped.
        # If j=u, skipped.
        # So {u, i, j} covers all pairs (i,j) from remaining vertices.
        # This includes pairs where one is v. e.g. i=v, j=k. Triple {u, v, k}.
        # So we don't need to check triples with v separately?
        # Let's verify.
        # Loop 1: `for i in range(n): if i==u: continue`.
        #   Inner: `for j in range(i+1, n): if j==u: continue`.
        #   This iterates over all pairs {i, j} from V \ {u}.
        #   The triple is {u, i, j}.
        #   This covers all triples containing u.
        #   Does it cover {u, v, w}? Yes, if i=v, j=w (assuming v<w or w<v and order adjusted).
        #   Since i and j are just indices, and we check all pairs from V\{u}, yes.
        
        # So checking all triples containing u is sufficient?
        # What about a new K3,3 that does NOT contain u or v?
        # Adding an edge (u,v) only changes N(u) and N(v).
        # It does not change N(i) for i != u, v.
        # Therefore, any triple {i, j, k} with i,j,k != u,v has the same common neighbors as before.
        # Since the graph was K3,3-free before, these triples are still safe.
        # Thus, we only need to check triples containing u or v.
        # My loop above checks all triples containing u.
        # Does it check all triples containing v?
        # A triple containing v is {v, i, j}.
        # If u is in the triple, it's {v, u, i} which is {u, v, i}. This was checked in the loop (as {u, v, i} or {u, i, v}).
        # If u is NOT in the triple, it's {v, i, j} with i, j != u, v.
        # This triple does NOT contain u.
        # My loop only checks triples containing u.
        # So I MISSED triples {v, i, j} where u is not involved.
        # I must also check triples containing v (and not u).
        
        for i in range(n):
            if i == v or i == u: continue
            for j in range(i + 1, n):
                if j == v or j == u: continue
                if check_triple(v, i, j):
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return False
        
        # Revert
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
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
    
    # Local search: Try to add edges. If stuck, remove random edges to escape local optima.
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Get list of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(u, v):
                    non_edges.append((u, v))
        
        if non_edges:
            u, v = random.choice(non_edges)
            if is_add_valid(u, v):
                add_edge(u, v)
                new_count = count_edges()
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    # This shouldn't happen if is_add_valid is correct and count_edges is consistent
                    remove_edge(u, v)
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # No non-edges? Graph is complete? Impossible for K3,3 free on 20 nodes.
            # Or we are stuck. Perturb.
            pass

        # If no improvement in a while, perturb by removing a random edge
        if no_improve > 500:
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
                no_improve = 0 # Reset counter after perturbation
    
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
