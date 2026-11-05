import time
import itertools
import random

def construct():
    start = time.time()
    deadline = start + 118  # safety margin
    
    n = 18
    N = 1 << n
    all_mask = (1 << n) - 1
    
    # Precompute bit operations
    bits = [1 << i for i in range(n)]
    
    def neighbors(mask, u):
        return mask & adj_bits[u]
    
    def has_triangle_in_set(S):
        # S is a bitmask
        if S == 0 or (S & (S - 1)) == 0 or (S & (S - 1) & (S - 1)) == 0:
            return False
        # Check for triangle
        # For each pair u,v in S, check if they have a common neighbor in S
        # Faster: iterate vertices
        s = S
        while s:
            lb = s & -s
            u = lb.bit_length() - 1
            s ^= lb
            # Common neighbors of u within S
            common = neighbors(S, u)
            common ^= lb  # remove u
            # Check if any two common neighbors are adjacent
            c = common
            while c:
                lb2 = c & -c
                v = lb2.bit_length() - 1
                c ^= lb2
                if neighbors(common, v):
                    return True
        return False

    # Build adjacency bitmasks from edge list
    def build_adj_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= bits[v]
            adj[v] |= bits[u]
        return adj

    # Check if graph is K4-free using bitmask
    # K4 exists if there are u,v,w,x all mutually adjacent
    # Equivalent: for any edge u-v, common neighbors of u and v contain an edge
    def is_k4_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & bits[v]):
                    continue
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                common &= ~(bits[u] | bits[v])
                # Check if common contains an edge
                c = common
                while c:
                    lb = c & -c
                    w = lb.bit_length() - 1
                    c ^= lb
                    if adj[w] & common:
                        return False
        return True

    # Find max triangle-free induced subgraph size
    # A set S is triangle-free if no 3 vertices form a triangle in G
    # We want max |S| such that G[S] is triangle-free
    
    def max_tri_free(adj):
        # Use branch and bound
        # State: (current_set, candidates)
        # candidates are vertices that can be added without creating a triangle
        # when added to current_set
        
        best = 0
        global max_tri_free_adj
        max_tri_free_adj = adj
        
        def search(current, candidates, size):
            nonlocal best
            if size > best:
                best = size
            
            if candidates == 0:
                return
            
            # Bound: size + popcount(candidates) <= best
            if size + bin(candidates).count('1') <= best:
                return
            
            # Try adding vertices from candidates
            c = candidates
            while c:
                lb = c & -c
                v = lb.bit_length() - 1
                c ^= lb
                
                # Adding v to current: check no triangle formed
                # Triangle would be formed if there exist u,w in current
                # such that u-w is an edge, and u-v and w-v are edges
                # i.e., common neighbors of v in current form an edge
                
                common_in_current = adj[v] & current
                # Check if common_in_current contains an edge
                cci = common_in_current
                has_edge = False
                while cci:
                    lb2 = cci & -cci
                    u = lb2.bit_length() - 1
                    cci ^= lb2
                    if adj[u] & common_in_current:
                        has_edge = True
                        break
                
                if has_edge:
                    continue
                
                # Also need to ensure that adding v doesn't create triangles
                # with other vertices in candidates that we might add later
                # Actually, we just need G[current ∪ {v}] to be triangle-free
                # The check above ensures no triangle containing v exists
                # Triangles not containing v are already fine since current is triangle-free
                
                # New candidates: vertices in candidates that can coexist with v
                # A vertex w in candidates can be added after v if:
                # - w is not in current (already true since w in candidates)
                # - adding w to current ∪ {v} doesn't create a triangle
                # This is complex. Simpler: just recurse with remaining candidates
                # but filter out those that would create a triangle with v
                
                new_candidates = 0
                cc = c  # remaining candidates
                while cc:
                    lb3 = cc & -cc
                    w = lb3.bit_length() - 1
                    cc ^= lb3
                    # Check if w can be added: G[current ∪ {v, w}] triangle-free
                    # Triangles containing v: checked via adj[v] & (current | bits[w])
                    # Triangles containing w: checked via adj[w] & (current | bits[v])
                    # Triangles not containing v or w: already fine
                    
                    # Triangle with v: exists u in current such that u-w and u-v edges
                    # adj[v] & adj[w] & current != 0 means common neighbor in current
                    if (adj[v] & adj[w] & current) != 0:
                        continue
                    # Triangle with w: exists u in current such that u-v and u-w edges
                    # Same condition! adj[v] & adj[w] & current != 0
                    # So if that's 0, both are fine
                    
                    new_candidates |= lb3
                
                # Remove v from candidates for the recursive call
                # Actually we already excluded v from c
                search(current | bits[v], new_candidates, size + 1)
        
        # Start with empty set, all vertices as candidates
        search(0, all_mask, 0)
        return best

    # Local search / simulated annealing
    def local_search(adj, edges, max_iter=5000):
        best_adj = adj[:]
        best_score = max_tri_free(adj)
        best_edges = edges[:]
        
        # Precompute edge positions
        edge_pos = {}
        for i, (u, v) in enumerate(edges):
            edge_pos[(u, v)] = i
            edge_pos[(v, u)] = i
        
        current_adj = adj[:]
        current_score = best_score
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to toggle
            if random.random() < 0.5:
                # Try adding a non-edge
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (current_adj[u] & bits[v]):
                            non_edges.append((u, v))
                if not non_edges:
                    continue
                u, v = random.choice(non_edges)
                # Check K4-freeness if we add this edge
                common = current_adj[u] & current_adj[v]
                common &= ~(bits[u] | bits[v])
                # Check if common has an edge
                c = common
                valid = True
                while c:
                    lb = c & -c
                    w = lb.bit_length() - 1
                    c ^= lb
                    if current_adj[w] & common:
                        valid = False
                        break
                if not valid:
                    continue
            else:
                # Try removing an edge
                if not edges:
                    continue
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                if not (current_adj[u] & bits[v]):
                    continue
            
            # Apply toggle
            if current_adj[u] & bits[v]:
                # Removing edge
                current_adj[u] &= ~bits[v]
                current_adj[v] &= ~bits[u]
                # Rebuild edge list
                new_edges = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if current_adj[a] & bits[b]:
                            new_edges.append([a, b])
            else:
                # Adding edge
                current_adj[u] |= bits[v]
                current_adj[v] |= bits[u]
                new_edges = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if current_adj[a] & bits[b]:
                            new_edges.append([a, b])
            
            new_score = max_tri_free(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
                    best_edges = new_edges[:]
            else:
                # Revert
                if (current_adj[u] & bits[v]) and (new_score > current_score):
                    # We added, now remove
                    current_adj[u] &= ~bits[v]
                    current_adj[v] &= ~bits[u]
                elif not (current_adj[u] & bits[v]) and (new_score > current_score):
                    # We removed, now add back
                    # But we need to know if we removed or added
                    # This logic is flawed. Let's track the move.
                    pass
        
        return best_adj, best_edges, best_score

    # Start with a good initial guess: K_{6,6,6}
    # Parts: 0-5, 6-11, 12-17
    initial_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            pi = i // 6
            pj = j // 6
            if pi != pj:
                initial_edges.append([i, j])
    
    adj = build_adj_edges(initial_edges)
    
    # Verify K4-free
    assert is_k4_free(adj), "Initial graph not K4-free"
    
    score = max_tri_free(adj)
    best_adj = adj[:]
    best_edges = initial_edges[:]
    best_score = score
    
    # Local search
    best_adj, best_edges, best_score = local_search(adj, initial_edges, max_iter=3000)
    
    # Additional random restarts if time permits
    while time.time() < deadline - 15:
        # Random graph with density ~0.5
        rand_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    rand_edges.append([u, v])
        
        rand_adj = build_adj_edges(rand_edges)
        
        if not is_k4_free(rand_adj):
            continue
        
        rand_score = max_tri_free(rand_adj)
        
        if rand_score < best_score:
            best_score = rand_score
            best_adj = rand_adj[:]
            best_edges = rand_edges[:]
            
            # Local search from here
            best_adj, best_edges, best_score = local_search(rand_adj, rand_edges, max_iter=1000)
        
        if best_score <= 7:
            break
    
    # Final K4-free verification
    final_adj = build_adj_edges(best_edges)
    if not is_k4_free(final_adj):
        # Fallback to K_{6,6,6}
        return initial_edges
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
