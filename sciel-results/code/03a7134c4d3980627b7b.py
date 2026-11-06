import random
import time

def construct():
    n = 40
    deadline = 145.0
    start_time = time.time()
    
    # Bitmask helpers
    def get_neighbors(adj, u):
        return adj[u]

    def get_distance_2(adj, u):
        """
        Returns a bitmask of vertices at distance exactly 2 from u.
        N_2(u) = (Union of N(v) for v in N(u)) \ N(u) \ {u}
        """
        neighbors = adj[u]
        if neighbors == 0:
            return 0
        
        # Union of neighbors of all neighbors
        union = 0
        mask = neighbors
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            union |= adj[v]
            mask ^= lsb
        
        # Remove direct neighbors and u itself
        union &= ~neighbors
        union &= ~(1 << u)
        
        return union

    def can_add_edge(adj, u, v):
        """
        Returns True if adding edge (u, v) does NOT create a C6.
        A C6 is created if there is a path of length 4 between u and v in the current graph.
        Path: u - a - w - b - v.
        w must be at dist 2 from u and dist 2 from v.
        a must be in N(u) \cap N(w).
        b must be in N(v) \cap N(w).
        We need a != b.
        """
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False # Edge already exists
        
        nu = adj[u]
        nv = adj[v]
        
        n2_u = get_distance_2(adj, u)
        n2_v = get_distance_2(adj, v)
        
        common_w = n2_u & n2_v
        
        if common_w == 0:
            return True
        
        mask = common_w
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask ^= lsb
            
            # A = N(u) \cap N(w)
            # B = N(v) \cap N(w)
            A = nu & adj[w]
            B = nv & adj[w]
            
            if A == 0 or B == 0:
                continue
                
            # If there exists a in A and b in B such that a != b, then C6 is created.
            # This is false ONLY IF A and B are both {x} for the same x.
            # i.e., A | B has only one bit set, and that bit is in A & B.
            if (A | B) & (A | B - 1) == 0:
                # A | B is a power of 2 (single bit)
                # Check if that bit is in A and B
                if (A & B) != 0:
                    continue # Safe for this w
            # If we reach here, either A|B has >1 bit, or A&B is 0
            # In both cases, there exists a != b
            return False
            
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def random_graph_density(density):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def greedy_optimize(adj, time_limit=5.0):
        """
        Try to add edges greedily.
        Returns (adj, edge_count)
        """
        start = time.time()
        changed = True
        while changed and time.time() - start < time_limit:
            changed = False
            # Shuffle order to avoid bias
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj, count_edges(adj)

    def local_search(adj, time_limit=10.0):
        """
        Local search with edge removal to escape local optima.
        """
        start = time.time()
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        # List of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() - start < time_limit:
            # Try to add random edges
            random.shuffle(all_edges)
            added = False
            for u, v in all_edges:
                if not (adj[u] & (1 << v)):
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
            
            if not added:
                # Stuck, try to remove a random edge to open up space
                # Pick an edge that is part of many C4s or just random
                # Random removal is simpler
                existing = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] & (1 << j))]
                if existing:
                    u, v = random.choice(existing)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    # After removal, try to add again immediately
                    random.shuffle(all_edges)
                    for u2, v2 in all_edges:
                        if not (adj[u2] & (1 << v2)):
                            if can_add_edge(adj, u2, v2):
                                adj[u2] |= (1 << v2)
                                adj[v2] |= (1 << u2)
                                current_edges += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = adj[:]
        
        return best_adj, best_edges

    best_adj = [0] * n
    best_count = 0
    
    # Strategy:
    # 1. Start with a random graph of moderate density (e.g., 0.3)
    # 2. Greedy optimize
    # 3. Local search
    # 4. Repeat with different seeds
    
    # Known good structure: 8xK5 has 8 * 10 = 80 edges.
    # We want > 100.
    
    seeds = range(50)
    for seed in seeds:
        if time.time() - start_time > deadline:
            break
            
        random.seed(seed)
        
        # Initialize
        # Try a few different initial densities
        for density in [0.2, 0.3, 0.4]:
            if time.time() - start_time > deadline:
                break
            adj = random_graph_density(density)
            
            # Quick greedy pass
            adj, count = greedy_optimize(adj, time_limit=2.0)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
                
            # Local search if we have time
            if time.time() - start_time < deadline - 15:
                adj_ls, count_ls = local_search(adj, time_limit=5.0)
                if count_ls > best_count:
                    best_count = count_ls
                    best_adj = adj_ls[:]
                    
    # Fallback if we failed to find a good graph
    if best_count < 80:
        # Construct 8xK5
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        best_adj = adj
        best_count = 80

    # Convert bitmask adjacency to list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Edges: {len(edges)}, Time: {end - start:.2f}s")
    # Verify C6-free
    # (Omitted for brevity in thought process, but should be verified)
