import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    # Initialize adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    def has_q3(subset):
        """
        Check if the induced subgraph on 'subset' (list of 8 vertices) is isomorphic to Q3.
        Q3 is bipartite, 3-regular, and has 8 vertices.
        A faster check: 
        1. Check if all 8 vertices have degree 3 within the subset.
        2. Check if the graph is bipartite.
        3. Check if it is connected.
        4. Q3 is the unique 3-regular bipartite connected graph on 8 vertices.
        """
        verts = list(subset)
        # Check degree
        for v in verts:
            deg = 0
            for u in verts:
                if u != v and adj[v][u]:
                    deg += 1
            if deg != 3:
                return False
        
        # Check bipartiteness using BFS
        color = {v: -1 for v in verts}
        for start_v in verts:
            if color[start_v] != -1:
                continue
            color[start_v] = 0
            queue = [start_v]
            while queue:
                u = queue.pop(0)
                for w in verts:
                    if w != u and adj[u][w]:
                        if color[w] == -1:
                            color[w] = color[u] ^ 1
                            queue.append(w)
                        elif color[w] == color[u]:
                            return False # Odd cycle found
        
        # Check connectivity (implicitly done if BFS visited all, but let's be sure)
        visited = set()
        queue = [verts[0]]
        visited.add(verts[0])
        while queue:
            u = queue.pop(0)
            for w in verts:
                if w != u and adj[u][w] and w not in visited:
                    visited.add(w)
                    queue.append(w)
        if len(visited) != 8:
            return False
            
        return True

    def is_q3_free_with_edge(u, v):
        """
        Check if adding edge (u,v) creates a Q3.
        We only need to check 8-vertex subsets that include u and v.
        There are C(14, 6) = 3003 such subsets.
        """
        others = [x for x in range(n) if x != u and x != v]
        # Iterate over combinations of 6 other vertices
        for comb in itertools.combinations(others, 6):
            subset = (u, v) + comb
            if has_q3(subset):
                return False
        return True

    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Strategy: 
    # 1. Start with a known good dense seed to save time? 
    #    The K3,13 + K3 inside part is 39+3=42 edges. 
    #    Let's start with K3,13 (39 edges) + edges inside the size-3 part (3 edges) = 42 edges.
    #    Actually, let's just start empty and let the greedy fill it. 
    #    Or better, start with the seed from the inspiration: K7 + K7 + K2.
    #    K7 has 21 edges. Two K7s have 42 edges. K2 has 1 edge. Total 43 edges.
    #    Is K7+K7+K2 Q3-free? Yes, because Q3 is connected and 3-regular. 
    #    K7 is 6-regular. A Q3 cannot be a subgraph of a disjoint union of cliques 
    #    because Q3 is bipartite (triangle-free) and cliques of size >=3 have triangles.
    #    Wait, Q3 is triangle-free. K7 has triangles. So Q3 cannot be a subgraph of K7.
    #    So any graph composed of cliques is Q3-free? 
    #    No, if we have edges between cliques, we might form a Q3.
    #    But the disjoint union of K7 and K7 and K2 is definitely Q3-free.
    #    Edges: 21 + 21 + 1 = 43.
    
    # Let's seed with this 43-edge graph.
    seed_edges = []
    # K7 on 0-6
    for i in range(7):
        for j in range(i + 1, 7):
            seed_edges.append((i, j))
    # K7 on 7-13
    for i in range(7, 14):
        for j in range(i + 1, 14):
            seed_edges.append((i, j))
    # K2 on 14-15
    seed_edges.append((14, 15))
    
    # Add seed edges to adj
    for u, v in seed_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    best_edge_count = len(seed_edges)
    current_edges = set(seed_edges)
    
    # Remaining edges to try
    remaining_edges = [e for e in all_edges if e not in current_edges]
    
    # Greedy loop with randomization/restart to escape local optima?
    # Since N=16 is small, we can try multiple random orders.
    
    while time.time() < deadline:
        random.shuffle(remaining_edges)
        # Reset to seed
        for u, v in seed_edges:
            adj[u][v] = True
            adj[v][u] = True
        # Clear non-seed edges (we need a way to distinguish or just rebuild)
        # Actually, rebuilding adj is fast.
        for i in range(n):
            for j in range(n):
                adj[i][j] = False
        
        for u, v in seed_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        added_any = False
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            # Check if edge is already present
            if adj[u][v]:
                continue
            if is_q3_free_with_edge(u, v):
                adj[u][v] = True
                adj[v][u] = True
                added_any = True
                best_edge_count = max(best_edge_count, sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j]))
        
        if not added_any:
            # If no edges were added in this pass, we are stuck in this random order.
            # Try another random order.
            continue

    # Return the best graph found
    # We need to reconstruct the edge list from the adj matrix of the last run
    # But we might have overwritten adj in the loop. 
    # Let's keep the best adj matrix.
    
    # Optimization: Keep a copy of the best adj matrix found.
    # Since we reset adj in the loop, we need to save the state when best_edge_count increases.
    
    # Let's refactor the loop to save the best state.
    
    # Re-run with state saving
    start_time = time.time()
    deadline = start_time + 145
    
    best_adj = [[False] * n for _ in range(n)]
    best_count = 0
    
    # Initial seed
    for u, v in seed_edges:
        best_adj[u][v] = True
        best_adj[v][u] = True
    best_count = len(seed_edges)
    
    while time.time() < deadline:
        # Reset current adj to seed
        curr_adj = [[False] * n for _ in range(n)]
        for u, v in seed_edges:
            curr_adj[u][v] = True
            curr_adj[v][u] = True
        
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if curr_adj[u][v]:
                continue
            
            # Check if adding (u,v) creates Q3
            # We need to update the check function to use curr_adj
            # Let's inline the check for speed or pass adj
            others = [x for x in range(n) if x != u and x != v]
            valid = True
            for comb in itertools.combinations(others, 6):
                subset = (u, v) + comb
                # Check Q3 on subset in curr_adj
                # Degree check
                ok = True
                for v_node in subset:
                    deg = 0
                    for u_node in subset:
                        if u_node != v_node and curr_adj[v_node][u_node]:
                            deg += 1
                    if deg != 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Bipartite check
                color = {v_node: -1 for v_node in subset}
                is_bip = True
                for start_v in subset:
                    if color[start_v] != -1:
                        continue
                    color[start_v] = 0
                    queue = [start_v]
                    while queue and is_bip:
                        u_node = queue.pop(0)
                        for w_node in subset:
                            if w_node != u_node and curr_adj[u_node][w_node]:
                                if color[w_node] == -1:
                                    color[w_node] = color[u_node] ^ 1
                                    queue.append(w_node)
                                elif color[w_node] == color[u_node]:
                                    is_bip = False
                                    break
                if not is_bip:
                    continue
                
                # Connectivity check
                visited = set()
                queue = [subset[0]]
                visited.add(subset[0])
                while queue:
                    u_node = queue.pop(0)
                    for w_node in subset:
                        if w_node != u_node and curr_adj[u_node][w_node] and w_node not in visited:
                            visited.add(w_node)
                            queue.append(w_node)
                if len(visited) != 8:
                    continue
                    
                # It is a Q3
                valid = False
                break
            
            if valid:
                curr_adj[u][v] = True
                curr_adj[v][u] = True
                current_count = sum(1 for i in range(n) for j in range(i+1, n) if curr_adj[i][j])
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [row[:] for row in curr_adj]
    
    # Convert best_adj to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                final_edges.append([i, j])
                
    return final_edges
