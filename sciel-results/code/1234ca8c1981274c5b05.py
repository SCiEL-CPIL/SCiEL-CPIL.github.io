import time
import random

def construct():
    deadline = time.time() + 115
    n = 16
    full_mask = (1 << n) - 1
    
    # Precompute all 8-vertex subsets? Too many (12870). Instead enumerate Q3 embeddings.
    # A Q3 has 8 vertices and 12 edges. We need to find if current graph contains Q3.
    
    # Represent graph as list of bitmasks
    adj = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def find_q3(adj):
        """Find a Q3 subgraph in adj. Returns list of 8 vertex indices or None."""
        # A Q3 can be seen as two disjoint 4-cycles (faces) connected by 4 matching edges.
        # Alternative: enumerate all 8-subsets and check if they induce a Q3.
        # 16 choose 8 = 12870 subsets. For each, check 12 edges.
        # That's about 154k edge checks per call - fast enough.
        
        # Better: use the structure of Q3. Q3 vertices can be labeled 0..7 as 3-bit numbers.
        # Edges connect vertices differing in exactly one bit.
        # We need to find an injective map from {0..7} to {0..15} preserving adjacency.
        
        # Enumerate all 8-subsets
        import itertools
        verts = list(range(n))
        for combo in itertools.combinations(verts, 8):
            # Check if these 8 vertices induce a Q3
            # Build local adjacency
            local_adj = {}
            for i, v in enumerate(combo):
                mask = adj[v] & full_mask
                # Count neighbors within combo
                cnt = 0
                for j, w in enumerate(combo):
                    if i != j and (mask >> w) & 1:
                        cnt += 1
                local_adj[i] = cnt
            
            # Q3 is 3-regular, so each vertex must have exactly 3 neighbors in the set
            if any(cnt != 3 for cnt in local_adj.values()):
                continue
            
            # Verify it's exactly Q3 (connected, girth 4, etc.)
            # Simpler: check that the induced subgraph is isomorphic to Q3
            # Q3 has 8 vertices, 12 edges, is bipartite, 3-regular, connected
            # Check edge count
            ec = sum(local_adj.values()) // 2
            if ec != 12:
                continue
            
            # Check bipartite: 2-coloring
            color = [-1] * 8
            color[0] = 0
            from collections import deque
            queue = deque([0])
            valid = True
            while queue and valid:
                u = queue.popleft()
                for v in range(8):
                    if u == v:
                        continue
                    if (adj[combo[u]] >> combo[v]) & 1:
                        if color[v] == -1:
                            color[v] = color[u] ^ 1
                            queue.append(v)
                        elif color[v] == color[u]:
                            valid = False
                            break
            if not valid:
                continue
            
            # Check connected
            visited = [False] * 8
            visited[0] = True
            q2 = deque([0])
            while q2:
                u = q2.popleft()
                for v in range(8):
                    if u != v and not visited[v] and (adj[combo[u]] >> combo[v]) & 1:
                        visited[v] = True
                        q2.append(v)
            if not all(visited):
                continue
            
            # It's a Q3!
            return list(combo)
        
        return None
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Strategy: Start from K16, repeatedly find Q3 and remove an edge.
    # To be smart about which edge to remove, try all 12 edges of the found Q3
    # and pick the one that breaks the most Q3s (approximated by trying each).
    
    best_adj = adj[:]
    best_count = edge_count(adj)
    
    # Phase 1: Greedy removal from K16
    cur_adj = adj[:]
    cur_count = 120
    
    # Find Q3s and remove edges until Q3-free
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        q3 = find_q3(cur_adj)
        if q3 is None:
            break
        
        # Found a Q3. Try removing each of its 12 edges, pick best.
        # Get the 12 edges of this Q3
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if (cur_adj[q3[i]] >> q3[j]) & 1:
                    q3_edges.append((q3[i], q3[j]))
        
        # There should be exactly 12 edges
        if len(q3_edges) != 12:
            # Something wrong, just remove first edge
            u, v = q3_edges[0]
            remove_edge(cur_adj, u, v)
            cur_count -= 1
            continue
        
        # Try each edge removal, measure how many Q3s remain (approx: count Q3s after removal)
        # This is expensive. Instead, use a heuristic: remove an edge that is in many Q3s.
        # Approximation: remove a random edge from the Q3, but try a few.
        best_remove = None
        best_score = -1
        
        random.shuffle(q3_edges)
        for u, v in q3_edges[:6]:  # Try up to 6 edges
            trial = cur_adj[:]
            remove_edge(trial, u, v)
            # Quick check: is it still Q3-free? (expensive)
            # Instead, just count remaining Q3s approximately
            # For now, just remove and move on
            pass
        
        # Simpler: just remove a random edge from the Q3
        u, v = random.choice(q3_edges)
        remove_edge(cur_adj, u, v)
        cur_count -= 1
    
    if cur_count > best_count:
        best_count = cur_count
        best_adj = cur_adj[:]
    
    # Phase 2: Local search - try to add back removed edges
    # Find missing edges
    missing = []
    for u in range(n):
        for v in range(u+1, n):
            if not (best_adj[u] >> v) & 1:
                missing.append((u, v))
    
    # Try adding each missing edge if it doesn't create Q3
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 10:
            break
        trial = best_adj[:]
        add_edge(trial, u, v)
        if find_q3(trial) is None:
            best_adj = trial
            best_count += 1
            # Remove from missing
            try:
                missing.remove((u, v))
            except ValueError:
                pass
    
    # Phase 3: Multi-start with different removal orders
    for seed in range(5):
        if time.time() > deadline - 15:
            break
        
        random.seed(seed)
        trial_adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                trial_adj[u] |= (1 << v)
                trial_adj[v] |= (1 << u)
        
        trial_count = 120
        
        for _ in range(300):
            if time.time() > deadline - 15:
                break
            q3 = find_q3(trial_adj)
            if q3 is None:
                break
            q3_edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (trial_adj[q3[i]] >> q3[j]) & 1:
                        q3_edges.append((q3[i], q3[j]))
            if q3_edges:
                u, v = random.choice(q3_edges)
                remove_edge(trial_adj, u, v)
                trial_count -= 1
        
        if trial_count > best_count:
            best_count = trial_count
            best_adj = trial_adj[:]
    
    # Final: try to add back edges
    if time.time() < deadline - 5:
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not (best_adj[u] >> v) & 1:
                    missing.append((u, v))
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            trial = best_adj[:]
            add_edge(trial, u, v)
            if find_q3(trial) is None:
                best_adj = trial
                best_count += 1
    
    # Convert to edge list
    result = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            bit = mask & (-mask)
            v = bit.bit_length() - 1
            if v > u:
                result.append([u, v])
            mask -= bit
    
    return result
