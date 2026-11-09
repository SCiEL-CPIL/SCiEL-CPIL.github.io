"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    
    # 1. Generate the Grötzsch graph (Mycielski of C5) on vertices 0..10
    def get_groetzsch_edges():
        edges = []
        # C5: 0-1-2-3-4-0
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Mycielski: 5..9
        for i in range(5):
            neighbors = [(i + 1) % 5, (i - 1) % 5]
            for n_idx in neighbors:
                edges.append((5 + i, n_idx))
        # Vertex 10
        for i in range(5):
            edges.append((10, 5 + i))
        return edges

    # Initialize graph
    edge_set = set()
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u == v: return False
        if (u, v) in edge_set or (v, u) in edge_set: return False
        # Check triangle free
        if adj[u] & adj[v]:
            return False
        # Add
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        return True

    def remove_edge(u, v):
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        adj[u].discard(v)
        adj[v].discard(u)

    # Load Grötzsch
    for u, v in get_groetzsch_edges():
        add_edge(u, v)

    best_edges_count = len(edge_set)
    best_edge_list = [(u, v) for u, v in edge_set]
    
    # Helper to check 3-colorability
    def is_not_3_colorable():
        # Backtracking 3-coloring
        color = [-1] * n
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Available colors
            used = set()
            for u in adj[v]:
                if color[u] != -1:
                    used.add(color[u])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack(0)

    # Phase 1: Greedy Edge Addition
    # Add all possible edges that maintain triangle-free property
    # We sort candidates by a heuristic: prefer edges that connect high-degree nodes or "fill" gaps
    while time.time() < deadline:
        current_edges_count = len(edge_set)
        
        # Find all valid non-edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    # Check if valid (triangle free)
                    if not (adj[u] & adj[v]):
                        # Heuristic: lower degree sum is "cheaper" to add in terms of blocking future edges?
                        # Or maybe higher degree sum is better to make the graph denser?
                        # Actually, adding an edge (u,v) blocks (u,y) for y in N(v) and (v,y) for y in N(u).
                        # So cost = deg(u) + deg(v).
                        # We want to minimize cost to keep options open?
                        # Or we just want to add as many as possible.
                        # Let's try adding the one with the smallest impact on future options first (greedy).
                        cost = len(adj[u]) + len(adj[v])
                        candidates.append((cost, u, v))
        
        if not candidates:
            break
            
        # Sort by cost (lowest first)
        candidates.sort()
        
        # Try to add the cheapest one
        added = False
        for cost, u, v in candidates:
            if add_edge(u, v):
                added = True
                current_edges_count += 1
                break
        
        if not added:
            break
            
        if current_edges_count > best_edges_count:
            best_edges_count = current_edges_count
            best_edge_list = [(u, v) for u, v in edge_set]

    # Phase 2: Local Search / Swap
    # Try to swap edges to increase density while maintaining triangle-free and 4-chromaticity
    # Since we started with a 4-chromatic graph and only added edges, it remains 4-chromatic.
    # However, if we remove edges, we might lose 4-chromaticity.
    # The task requires the final graph to be not 3-colorable.
    # The greedy phase maintains this.
    # Can we do better than simple greedy?
    # Maybe try to remove a "bad" edge and add two "good" edges?
    
    # Let's try a simple local search:
    # 1. Pick a random edge e1.
    # 2. Remove it.
    # 3. Try to add a different edge e2 that was previously blocked by e1.
    # 4. If e2 can be added, keep it.
    # 5. Check if still 4-chromatic (it should be if we didn't disconnect components too much, but check anyway).
    
    # Actually, the greedy phase usually finds a maximal triangle-free graph.
    # The maximum number of edges in a triangle-free graph on 16 vertices is given by Turan's theorem:
    # T(16, 2) is a complete bipartite graph K(8,8) with 64 edges.
    # But K(8,8) is bipartite (2-colorable).
    # We need 4-chromatic.
    # The Grötzsch graph has 20 edges.
    # The greedy phase on top of Grötzsch might add more.
    
    # Let's see how many edges the greedy phase finds.
    # If it finds a maximal triangle-free graph, it might be close to Turan's bound but not bipartite.
    
    # Let's try to improve the greedy phase by randomization/restarts if time permits.
    # But the current greedy is deterministic.
    
    # Let's try a different heuristic for candidate sorting:
    # Maybe sort by random order to escape local optima?
    # Or try multiple random orderings?
    
    # For now, let's stick with the deterministic greedy which is fast.
    # If we have time left, we can try to "shake" the graph.
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_edge_list

    # Try to improve by swapping
    # We will try to remove an edge and add another.
    # This is complex. Let's just return the greedy result for now, as it's likely good.
    
    # Wait, the inspiration says "greedy edge-addition phase followed by targeted local search".
    # Let's implement a simple local search.
    
    # Save current state
    current_edges = list(edge_set)
    
    # Try random swaps
    for _ in range(1000): # Limit iterations
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random edge to remove
        if not current_edges:
            break
        e1 = random.choice(current_edges)
        u1, v1 = e1
        
        # Remove it
        remove_edge(u1, v1)
        
        # Find edges that were blocked by e1
        # An edge (u, x) is blocked if x in N(v) or x in N(u) was the conflict.
        # Actually, (u, x) is invalid if N(u) & N(x) is not empty.
        # Before removing e1, N(u) contained v1.
        # So any x in N(v1) might have been blocked from u.
        # And any y in N(u1) might have been blocked from v1.
        
        potential_new_edges = []
        for x in adj[v1]:
            if (u1, x) not in edge_set and not (adj[u1] & adj[x]):
                potential_new_edges.append((u1, x))
        for y in adj[u1]:
            if (v1, y) not in edge_set and not (adj[v1] & adj[y]):
                potential_new_edges.append((v1, y))
        
        # Also check other edges?
        # Just try to add one of these
        random.shuffle(potential_new_edges)
        
        added = False
        for u2, v2 in potential_new_edges:
            if add_edge(u2, v2):
                added = True
                break
        
        if added:
            # Check 3-colorability
            if is_not_3_colorable():
                # Accept
                current_edges = list(edge_set)
                if len(edge_set) > best_edges_count:
                    best_edges_count = len(edge_set)
                    best_edge_list = [(u, v) for u, v in edge_set]
            else:
                # Reject, revert
                remove_edge(u2, v2)
                add_edge(u1, v1)
                current_edges = list(edge_set)
        else:
            # Revert
            add_edge(u1, v1)
            current_edges = list(edge_set)

    return best_edge_list
# EVOLVE-BLOCK-END
