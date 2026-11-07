"""Triangle-free diameter-four augmentation with Iterated Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14],
                        [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
                        [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _is_valid(n, edges):
    # edges: iterable of [u, v]. Checks all task rules.
    seen = set()
    for e in edges:
        if not isinstance(e, (list, tuple)):
            return False
        if len(e) != 2:
            return False
        u, v = e[0], e[1]
        if not (isinstance(u, int) and not isinstance(u, bool)):
            return False
        if not (isinstance(v, int) and not isinstance(v, bool)):
            return False
        if not (0 <= u < n and 0 <= v < n):
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)

    base = set()
    for e in P['edges']:
        base.add((min(e[0], e[1]), max(e[0], e[1])))
    for key in seen:
        if key in base:
            return False

    adj = {i: [] for i in range(n)}
    for (u, v) in seen:
        adj[u].append(v)
        adj[v].append(u)
    for e in P['edges']:
        a, b = e[0], e[1]
        adj[a].append(b)
        adj[b].append(a)

    # triangle-free: for every edge, endpoints share no common neighbor
    for (u, v) in seen:
        su = set(adj[u]); su.discard(v)
        sv = set(adj[v]); sv.discard(u)
        if su & sv:
            return False
    for (u, v) in base:
        su = set(adj[u]); su.discard(v)
        sv = set(adj[v]); sv.discard(u)
        if su & sv:
            return False

    # connected
    comp = {0}
    stack = [0]
    while stack:
        w = stack.pop()
        for nb in adj[w]:
            if nb not in comp:
                comp.add(nb)
                stack.append(nb)
    if len(comp) != n:
        return False

    # diameter <= 4 via BFS from each vertex
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            w = queue[head]
            head += 1
            for nb in adj[w]:
                if dist[nb] == -1:
                    dist[nb] = dist[w] + 1
                    queue.append(nb)
        for i in range(n):
            if dist[i] == -1:
                return False
            if dist[i] > P['diameter']:
                return False
    return True

def _seed_fallback():
    base = {(min(e[0], e[1]), max(e[0], e[1])) for e in P['edges']}
    out = []
    for u in range(P['n']):
        for v in range(u + 1, P['n']):
            if (u - v) % 2 and (u, v) not in base:
                out.append([u, v])
    return out

def construct():
    n = P['n']
    D = P['diameter']
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    # Start with a dense valid graph: all same-parity edges
    # This ensures connectivity and low diameter initially
    base_edges = {(min(e[0], e[1]), max(e[0], e[1])) for e in P['edges']}
    all_same_parity = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 0 and (u, v) not in base_edges:
                all_same_parity.add((u, v))
    
    current_edges = set(all_same_parity)
    if not _is_valid(n, [[u, v] for u, v in current_edges]):
        # Fallback if the dense graph somehow violates rules (shouldn't happen for same parity)
        # Actually, same parity edges on a path do not form triangles.
        # Path is 0-1-2... Edge (0,2) connects 0-2. 0-1-2 is a path, not triangle.
        # Triangle needs (0,1), (1,2), (0,2). (0,1) and (1,2) are base. (0,2) is added.
        # Wait, triangle-free check: for edge (0,2), neighbors of 0 are 1, 2... neighbors of 2 are 1, 3...
        # Common neighbor is 1. So (0,2) + (0,1) + (1,2) forms a triangle!
        # Ah, the base graph is a path. Adding (0,2) creates triangle 0-1-2-0.
        # So we CANNOT add (u, v) if |u-v| == 2.
        # We can add (u, v) if |u-v| >= 4 and same parity.
        pass

    # Correct Initial State:
    # Edges (u,v) with u < v, (u-v)%2==0, and v-u >= 4.
    # This avoids triangles with the path edges.
    current_edges = set()
    for u in range(n):
        for v in range(u + 4, n):
            if (u - v) % 2 == 0:
                current_edges.add((u, v))
                
    # Verify initial state
    if not _is_valid(n, [[u, v] for u, v in current_edges]):
        # If even this fails (unlikely), use a known valid small set or fallback
        # But this set should be valid. Diameter of P24 with all chords >=4 is small.
        # Let's just use the seed fallback if it fails, though it shouldn't.
        best_edges = _seed_fallback()
        return [[int(u), int(v)] for u, v in best_edges]

    best_edges = set(current_edges)
    best_count = len(best_edges)
    
    # Local Search: Greedily remove edges if valid
    def try_remove(edge_set):
        # Returns (new_set, removed_count)
        removed = 0
        # Shuffle to avoid bias
        edges_list = list(edge_set)
        random.shuffle(edges_list)
        for e in edges_list:
            if time.time() > deadline:
                break
            test_set = edge_set - {e}
            if _is_valid(n, [[u, v] for u, v in test_set]):
                edge_set = test_set
                removed += 1
        return edge_set, removed

    # Iterated Local Search
    while time.time() < deadline:
        # 1. Local Optimization (Removal)
        current_edges, _ = try_remove(current_edges)
        
        # Update best
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
        # If we reached a very low number, we might stop or continue to find 0?
        # Minimum is likely > 0.
        
        # 2. Perturbation
        # Add some random valid edges to escape local minima
        # Candidate edges: same parity, dist >= 4
        candidates = []
        for u in range(n):
            for v in range(u + 4, n):
                if (u - v) % 2 == 0 and (u, v) not in base_edges:
                    candidates.append((u, v))
        
        if candidates:
            # Add 2-5 random edges
            num_add = random.randint(2, 5)
            to_add = random.sample(candidates, min(num_add, len(candidates)))
            for e in to_add:
                # Check if adding this edge keeps it triangle-free?
                # We just add it, and the next local search will fix validity if it breaks?
                # No, _is_valid is expensive. We should only add if it *might* be valid.
                # Actually, adding an edge can create a triangle.
                # Better perturbation: Remove some edges, then add others?
                # Or just swap?
                
                # Let's try a "Swap" perturbation: Remove 1, Add 1.
                # Or just Add 1 and let local search remove the bad ones?
                # If we add an edge that creates a triangle, the graph becomes invalid.
                # Our local search only REMOVES edges. It cannot fix an invalid graph.
                # So we must ensure perturbation results in a VALID graph, or handle invalidity.
                
                # Strategy: Perturb by removing a few edges, then adding a few.
                # This maintains validity if we check after each step? Too slow.
                
                # Alternative: Random Restart from a different valid configuration?
                # Or: Add an edge ONLY IF it doesn't create a triangle.
                
                pass
        
        # Simpler Perturbation: 
        # Pick a random edge in current set, remove it. 
        # Pick a random candidate edge, add it if valid (triangle-free check).
        
        # Let's implement a safer perturbation:
        # 1. Remove 1-3 random edges from current set.
        # 2. Add 1-3 random edges from candidates that don't form triangles with existing edges.
        
        # Step 1: Remove
        if current_edges:
            remove_count = random.randint(1, 3)
            to_remove = random.sample(list(current_edges), min(remove_count, len(current_edges)))
            for e in to_remove:
                current_edges.remove(e)
        
        # Step 2: Add
        # We need to add edges that don't create triangles.
        # An edge (u,v) creates a triangle if there is a w such that (u,w) and (v,w) are in the graph.
        # Since the graph is sparse, we can check this.
        
        add_count = random.randint(1, 3)
        attempts = 0
        while add_count > 0 and attempts < 50:
            attempts += 1
            if not candidates: break
            e = random.choice(candidates)
            u, v = e
            # Check if adding (u,v) creates a triangle
            # Neighbors of u in current_edges + base
            # Neighbors of v in current_edges + base
            # If intersection is non-empty, it's a triangle.
            
            # Build adjacency quickly for check?
            # Or just check against known edges.
            # Let's build a set of edges for fast lookup
            edge_set = current_edges | base_edges
            adj_u = set()
            adj_v = set()
            for (a, b) in edge_set:
                if a == u: adj_u.add(b)
                if b == u: adj_u.add(a)
                if a == v: adj_v.add(b)
                if b == v: adj_v.add(a)
            
            if adj_u & adj_v:
                continue # Triangle formed
            
            # Check if (u,v) is already in current_edges
            if e in current_edges:
                continue
            
            current_edges.add(e)
            add_count -= 1

    # Final verification and return
    if not _is_valid(n, [[u, v] for u, v in best_edges]):
        best_edges = _seed_fallback()
        return [[int(u), int(v)] for u, v in best_edges]
        
    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
