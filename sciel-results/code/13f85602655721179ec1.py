import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Internal edges of C
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        
        # Used edges in C
        used_edges = set()
        
        partition = []
        
        # Strategy:
        # 1. Handle cross edges by grouping I vertices.
        #    For each I vertex v, it forms cliques {v} U S where S is a subset of N(v) in C.
        #    To minimize cliques, we want S to be as large as possible.
        #    However, edges in S must not be "used" by previous cliques.
        #    Wait, the condition is that every edge lies in EXACTLY ONE clique.
        #    So if we use an edge (u,w) in a clique {v, u, w}, we cannot use it in any other clique.
        #    This means the sets S_v for different v's must be edge-disjoint in C.
        #    
        #    Current approach in the seed:
        #    For each v in I (sorted by degree), look at N_v = N(v) in C.
        #    Build graph H on N_v where edges are those ALREADY USED in C.
        #    Color H. Each color class is an independent set in H, meaning no edges in that class are used.
        #    So we can form a clique {v} U S for each color class S.
        #    This uses up the edges in S (marking them as used).
        #    
        #    This approach is solid. The key to improvement is the coloring heuristic and the order of processing.
        #    Also, the remaining edges in C need to be partitioned.
        
        # Sort I by degree in C (descending) to process "dense" connections first
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
                
            # Build graph H on N_v using used_edges
            # Vertices are N_v
            # Edge (a,b) in H if (a,b) in used_edges
            
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for u in h_vertices:
                for w in h_vertices:
                    if u == w: continue
                    e = (min(u,w), max(u,w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        
            # Greedy coloring of H
            # Order vertices by degree in H (descending) for better coloring
            h_vertices_sorted = sorted(h_vertices, key=lambda x: len(h_adj[x]), reverse=True)
            
            colors = {}
            color_count = 0
            for u in h_vertices_sorted:
                # Find smallest available color
                used_colors = {colors[w] for w in h_adj[u] if w in colors}
                c = 0
                while c in used_colors:
                    c += 1
                colors[u] = c
                color_count = max(color_count, c + 1)
                
            # Group vertices by color
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                    
                clique = S + [v]
                # Check if clique is valid (it should be by construction)
                # S is independent in H, so no edges in S are in used_edges.
                # So all edges in S are available.
                # v is connected to all in S.
                partition.append(sorted(clique))
                
                # Mark edges in S as used
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
        
        # Partition remaining_edges into cliques
        # We want to minimize the number of cliques.
        # This is equivalent to finding a clique cover of the graph defined by remaining_edges.
        # A clique cover is a partition of the edges into cliques.
        # Note: The cliques themselves don't need to be a partition of vertices, just edges.
        # But the problem says "clique partition", which usually implies edge partition.
        # "Every listed set must be a clique ... and every edge must lie in exactly one listed clique."
        
        # Greedy Maximal Clique Algorithm:
        # 1. Pick an edge.
        # 2. Extend it to a maximal clique in the remaining graph.
        # 3. Remove all edges in that clique from the remaining graph.
        # 4. Repeat.
        
        rem_set = set(remaining_edges)
        while rem_set:
            # Pick an arbitrary edge (or one with high degree?)
            # To get large cliques, picking edges in dense areas is good.
            # Let's pick the edge with the smallest vertex id for determinism, 
            # but maybe sorting by degree of endpoints is better?
            # Let's stick to simple: pick first edge.
            u, v = next(iter(rem_set))
            
            # Find largest clique containing u and v in the graph defined by rem_set
            current_clique = {u, v}
            
            # Candidate vertices: neighbors of u and v in rem_set
            # Actually, we can just check all other vertices in C
            candidates = [w for w in range(k) if w not in current_clique]
            
            # Sort candidates by number of connections to current_clique in rem_set
            # This is a heuristic to grow the clique fast.
            
            def score(w):
                count = 0
                for x in current_clique:
                    e = (min(w,x), max(w,x))
                    if e in rem_set:
                        count += 1
                return count
            
            candidates.sort(key=score, reverse=True)
            
            for w in candidates:
                # Check if w can be added to current_clique
                can_add = True
                for x in current_clique:
                    e = (min(w,x), max(w,x))
                    if e not in rem_set:
                        can_add = False
                        break
                if can_add:
                    current_clique.add(w)
                    
            # Add this clique to partition
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            # Remove edges of this clique from rem_set
            for i in range(len(current_clique)):
                for j in range(i+1, len(current_clique)):
                    a, b = list(current_clique)[i], list(current_clique)[j]
                    rem_set.discard((min(a, b), max(a, b)))
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback if we run out of time: return edge-based partition (always valid)
            # But we need to finish the current instance or skip?
            # Better to just return what we have if we can, but we need 3 lists.
            # If we are here, we haven't finished even the first one? Unlikely.
            # Let's just break and return a valid fallback for remaining.
            pass
            
        res = solve_instance(*spec)
        # Ensure types are correct
        # res is list of lists of ints
        results.append(res)
        
    # If we ran out of time, we might have incomplete results.
    # The task says "Any broken rule makes the whole result invalid".
    # So we must ensure we return a valid result.
    # If we hit the deadline, we should probably just return a simple valid partition
    # for any instances not yet processed, or just the processed ones if we have 3.
    # But the loop processes all 3. If it takes too long, we might not finish the 3rd.
    # Let's add a check.
    
    # Actually, the current program is fast. 90 nodes is small.
    # The bottleneck might be the greedy clique finding on the remaining edges.
    # But k=30, so remaining edges are in a graph of 30 nodes.
    # Greedy clique cover on 30 nodes is very fast.
    
    return results
