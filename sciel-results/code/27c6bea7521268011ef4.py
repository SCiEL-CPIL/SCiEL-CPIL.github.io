import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n = 16
    # Generate all edges and their base colors
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            c = (i + j) % 17
            edges.append((i, j, c))
            
    # Map edge (i,j) to index
    edge_index = {}
    for idx, (i, j, c) in enumerate(edges):
        edge_index[(i, j)] = idx
        
    # Generate all K4s
    k4s = []
    for verts in itertools.combinations(range(n), 4):
        # Get the 6 edges
        e_indices = []
        for v1, v2 in itertools.combinations(verts, 2):
            e_indices.append(edge_index[(v1, v2)])
        k4s.append(e_indices)
        
    # Build conflict graph H on 17 colors
    # H[c1][c2] = True if c1 and c2 appear together in some K4
    conflict = [[False]*17 for _ in range(17)]
    
    for e_indices in k4s:
        colors_in_k4 = set()
        for idx in e_indices:
            colors_in_k4.add(edges[idx][2])
        # Add conflicts for all pairs in this K4
        for c1, c2 in itertools.combinations(colors_in_k4, 2):
            conflict[c1][c2] = True
            conflict[c2][c1] = True
            
    # Now, we want to find the maximum number of disjoint non-edges in H.
    # A non-edge is a pair (u, v) such that not conflict[u][v].
    # We want to select k pairs (u1, v1), ..., (uk, vk) such that all ui, vi are distinct.
    # This is a matching problem in the graph of non-edges.
    
    # Build graph of non-edges
    non_edge_adj = [[] for _ in range(17)]
    for u in range(17):
        for v in range(u + 1, 17):
            if not conflict[u][v]:
                non_edge_adj[u].append(v)
                non_edge_adj[v].append(u)
                
    # Find maximum matching in this graph using a simple DFS based algorithm (Kuhn's algorithm)
    # Since 17 is small, we can even use brute force or simple greedy.
    # Let's use a simple greedy first, then try to improve if time permits.
    
    # Greedy matching
    matched = [False] * 17
    pairs = []
    # Sort nodes by degree in non_edge_adj to prioritize high degree? Or just iterate.
    for u in range(17):
        if matched[u]:
            continue
        for v in non_edge_adj[u]:
            if not matched[v]:
                matched[u] = True
                matched[v] = True
                pairs.append((u, v))
                break
                
    # Check if we can do better with backtracking?
    # For now, let's see how many pairs we found.
    # If we found 4, we use 13 colors.
    # If 3, 14 colors.
    # If 2, 15 colors.
    # If 1, 16 colors.
    # If 0, 17 colors.
    
    num_merges = len(pairs)
    
    # Construct the mapping
    # We want to map the merged pairs to a single color.
    # Let's keep the smaller color index as the representative.
    color_map = list(range(17))
    for u, v in pairs:
        # Map v to u
        color_map[v] = u
        
    # Now, relabel colors to 0..K-1
    used_colors = sorted(set(color_map))
    final_map = {old: new for new, old in enumerate(used_colors)}
    
    # Apply to edges
    result = []
    for i, j, c in edges:
        mapped_c = color_map[c]
        final_c = final_map[mapped_c]
        result.append(final_c)
        
    # Verify validity
    # Check all K4s
    valid = True
    for e_indices in k4s:
        colors_in_k4 = set()
        for idx in e_indices:
            i, j, c = edges[idx]
            mapped_c = color_map[c]
            final_c = final_map[mapped_c]
            colors_in_k4.add(final_c)
        if len(colors_in_k4) < 5:
            valid = False
            break
            
    if not valid:
        # Fallback to original
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    # Count colors used
    num_colors = len(used_colors)
    
    # If we have time, try to improve?
    # The greedy matching might not be maximum.
    # But 17 is small. We can implement a proper max matching.
    
    # Let's implement a simple max matching for 17 nodes.
    # This is a bipartite matching problem if we split the set? No, it's a general graph matching.
    # For 17 nodes, we can use a bitmask DP or just simple backtracking.
    
    def find_max_matching(adj):
        n_nodes = len(adj)
        best = []
        
        def backtrack(node, current_pairs, visited):
            nonlocal best
            if node == n_nodes:
                if len(current_pairs) > len(best):
                    best = current_pairs[:]
                return
            
            # Option 1: Don't match node
            backtrack(node + 1, current_pairs, visited)
            
            # Option 2: Match node with some unvisited neighbor
            for neighbor in adj[node]:
                if neighbor > node and not visited[neighbor]:
                    visited[neighbor] = True
                    current_pairs.append((node, neighbor))
                    backtrack(node + 1, current_pairs, visited)
                    current_pairs.pop()
                    visited[neighbor] = False
                    # Pruning: if even matching all remaining nodes doesn't beat best, stop
                    # Max possible new pairs = (n_nodes - node - 1) // 2
                    if len(current_pairs) + (n_nodes - node - 1) // 2 <= len(best):
                        return
                        
        visited = [False] * n_nodes
        backtrack(0, [], visited)
        return best
        
    # Use the better matching
    if time.time() < deadline - 10:
        max_pairs = find_max_matching(non_edge_adj)
        if len(max_pairs) > num_merges:
            pairs = max_pairs
            # Reconstruct mapping
            color_map = list(range(17))
            for u, v in pairs:
                color_map[v] = u
            used_colors = sorted(set(color_map))
            final_map = {old: new for new, old in enumerate(used_colors)}
            
            # Re-verify
            valid = True
            for e_indices in k4s:
                colors_in_k4 = set()
                for idx in e_indices:
                    i, j, c = edges[idx]
                    mapped_c = color_map[c]
                    final_c = final_map[mapped_c]
                    colors_in_k4.add(final_c)
                if len(colors_in_k4) < 5:
                    valid = False
                    break
            if valid:
                result = []
                for i, j, c in edges:
                    mapped_c = color_map[c]
                    final_c = final_map[mapped_c]
                    result.append(final_c)
                num_colors = len(used_colors)
            else:
                # Revert to greedy result if it was valid
                pass # result and num_colors already set from greedy if valid
                
    return result
