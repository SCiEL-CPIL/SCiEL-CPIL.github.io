"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build conflict graph for MWIS approach
        # Nodes are (v, c)
        # Total nodes N_conf = sum(len(lists[v]) for v in range(n))
        # Map (v, c) to index
        node_index = {}
        idx = 0
        for v in range(n):
            for c in lists[v]:
                node_index[(v, c)] = idx
                idx += 1
        N_conf = idx
        
        # Build adjacency bitsets
        # adj[i] is a bitset of nodes conflicting with i
        adj = [0] * N_conf
        
        # 1. Conflicts within same vertex: (v, c) conflicts with (v, c') for c != c'
        for v in range(n):
            v_nodes = [node_index[(v, c)] for c in lists[v]]
            for i in range(len(v_nodes)):
                for j in range(len(v_nodes)):
                    if i != j:
                        u = v_nodes[i]
                        w = v_nodes[j]
                        adj[u] |= (1 << w)
        
        # 2. Conflicts between adjacent vertices: (v, c) conflicts with (u, c) if u ~ v
        # First, map color c to list of nodes (v, c)
        color_nodes = {c: [] for c in range(5)}
        for v in range(n):
            for c in lists[v]:
                color_nodes[c].append(node_index[(v, c)])
        
        for u, v in edges:
            # For each color c, all nodes (u, c) conflict with all nodes (v, c)
            for c in range(5):
                if c in lists[u] and c in lists[v]:
                    # Get indices
                    # This is O(1) if we precomputed, but lists are small
                    # We can just iterate
                    for i in color_nodes[c]:
                        if i < N_conf: # sanity
                            pass # We need to know if i corresponds to u or v?
                            # Actually, we just need to add edges between ALL (u,c) and ALL (v,c)
                            # But we don't know which node is which from just the index in color_nodes[c].
                            # We need to filter.
        
        # Better way for step 2:
        # For each edge (u, v):
        #   For each c in intersection(lists[u], lists[v]):
        #     For each node i in color_nodes[c] that belongs to u:
        #     For each node j in color_nodes[c] that belongs to v:
        #        adj[i] |= 1 << j
        #        adj[j] |= 1 << i
        
        # Precompute vertex_nodes[v] = list of node indices for vertex v
        vertex_nodes = [[] for _ in range(n)]
        for v in range(n):
            for c in lists[v]:
                vertex_nodes[v].append(node_index[(v, c)])
        
        for u, v in edges:
            common_colors = set(lists[u]) & set(lists[v])
            for c in common_colors:
                # Find nodes of u with color c
                nodes_u = [i for i in vertex_nodes[u] if lists[u][lists[u].index(c)] == c] # This is inefficient
                # Better: store node_index by (v,c)
                # We already have node_index
                # Let's just use the mapping
                i_u = node_index.get((u, c), -1)
                i_v = node_index.get((v, c), -1)
                if i_u != -1 and i_v != -1:
                    adj[i_u] |= (1 << i_v)
                    adj[i_v] |= (1 << i_u)
                    
        # Now we have the conflict graph.
        # We want to find an independent set of size n.
        # This is a decision problem: Does an independent set of size n exist?
        # We use recursive B&B.
        
        # State: candidates (bitset), needed (int)
        # Initial candidates: all nodes
        all_nodes = (1 << N_conf) - 1
        
        # To speed up, we can precompute the degree of each node in the initial graph?
        # No, dynamic.
        
        # Heuristic: Pick vertex with highest degree in candidates
        def search(candidates, needed):
            if needed == 0:
                return True
            if candidates == 0:
                return False
            if candidates.bit_count() < needed:
                return False
            
            # Pick a vertex u from candidates
            # Heuristic: Highest degree in the induced subgraph
            # To avoid O(N^2) calculation, we can just pick the first one or use a simple heuristic
            # Given N_conf ~ 1200, O(N^2) per node might be too slow if depth is large.
            # However, the branching reduces candidates significantly.
            
            # Simple heuristic: Pick the vertex with the most neighbors in candidates
            best_u = -1
            max_deg = -1
            # Iterate through set bits in candidates
            c = candidates
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                deg = (adj[u] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_u = u
                c ^= lsb
            
            if best_u == -1:
                return False
                
            # Branch 1: Include best_u
            # Remove best_u and its neighbors
            new_candidates = candidates & ~(adj[best_u] | (1 << best_u))
            if search(new_candidates, needed - 1):
                return True
            
            # Branch 2: Exclude best_u
            new_candidates = candidates & ~(1 << best_u)
            if search(new_candidates, needed):
                return True
                
            return False
        
        # We need to extract the actual coloring if search succeeds.
        # The above search only returns True/False.
        # We need to modify it to store the path or return the set.
        
        solution_set = 0
        def search_with_path(candidates, needed, current_set):
            if needed == 0:
                return True
            if candidates == 0:
                return False
            if candidates.bit_count() < needed:
                return False
            
            c = candidates
            best_u = -1
            max_deg = -1
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                deg = (adj[u] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_u = u
                c ^= lsb
            
            if best_u == -1:
                return False
                
            # Branch 1: Include best_u
            new_candidates = candidates & ~(adj[best_u] | (1 << best_u))
            if search_with_path(new_candidates, needed - 1, current_set | (1 << best_u)):
                return True
            
            # Branch 2: Exclude best_u
            new_candidates = candidates & ~(1 << best_u)
            if search_with_path(new_candidates, needed, current_set):
                return True
                
            return False
        
        if search_with_path(all_nodes, n, 0):
            # Extract coloring
            colour = [0] * n
            # Iterate through bits in solution_set
            s = solution_set
            while s:
                lsb = s & -s
                u = lsb.bit_length() - 1
                # Find which (v, c) this is
                # We need reverse mapping
                # Let's create a list node_to_vc
                # We didn't create it explicitly, but we can derive it
                # Or just iterate v, c and check
                # For efficiency, create reverse map
                pass
            # Create reverse map
            reverse_map = [None] * N_conf
            for v in range(n):
                for c in lists[v]:
                    idx = node_index[(v, c)]
                    reverse_map[idx] = (v, c)
            
            s = solution_set
            while s:
                lsb = s & -s
                u = lsb.bit_length() - 1
                v, c = reverse_map[u]
                colour[v] = c
                s ^= lsb
            
            return [int(c) for c in colour]
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        col = solve_instance(n, edges, lists)
        out.append(col)
    
    final_out = []
    for col_list in out:
        final_out.append([int(c) for c in col_list])
    
    return final_out
# EVOLVE-BLOCK-END
