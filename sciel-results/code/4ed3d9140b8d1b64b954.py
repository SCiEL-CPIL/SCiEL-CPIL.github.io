import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []

        # 1. Build graph structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 2. Global Greedy Trail Cover with Warnsdorff Heuristic
        # We repeatedly pick an edge, extend the trail in both directions 
        # choosing the neighbor with the fewest remaining available edges.
        
        remaining_edges = set(edges)
        # To speed up lookup, we maintain a set of available edges per node
        avail = defaultdict(set)
        for u, v in edges:
            avail[u].add(v)
            avail[v].add(u)
            
        paths = []
        
        while remaining_edges:
            if time.time() > deadline - 5:
                # Fallback: just add remaining as single edges
                for e in remaining_edges:
                    paths.append([e[0], e[1]])
                return paths

            # Pick an arbitrary edge to start
            # Heuristic: pick an edge incident to a vertex with high degree? 
            # Or just arbitrary? Arbitrary is fine, Warnsdorff handles the rest.
            start_edge = next(iter(remaining_edges))
            u, v = start_edge
            
            # Remove start edge
            remaining_edges.discard(start_edge)
            avail[u].remove(v)
            avail[v].remove(u)
            
            path = [u, v]
            left_end = u
            right_end = v
            
            # Extend left
            while True:
                # Find best next node for left_end
                best_nb = None
                min_avail = float('inf')
                for nb in avail[left_end]:
                    # Count remaining edges from nb
                    # Note: avail[nb] contains neighbors. The size is the degree in remaining graph.
                    # Warnsdorff: pick the one with fewest options.
                    count = len(avail[nb])
                    if count < min_avail:
                        min_avail = count
                        best_nb = nb
                
                if best_nb is None:
                    break
                
                # Move left_end to best_nb
                edge = (min(left_end, best_nb), max(left_end, best_nb))
                remaining_edges.discard(edge)
                avail[left_end].remove(best_nb)
                avail[best_nb].remove(left_end)
                
                path.insert(0, best_nb)
                left_end = best_nb
            
            # Extend right
            while True:
                best_nb = None
                min_avail = float('inf')
                for nb in avail[right_end]:
                    count = len(avail[nb])
                    if count < min_avail:
                        min_avail = count
                        best_nb = nb
                
                if best_nb is None:
                    break
                
                edge = (min(right_end, best_nb), max(right_end, best_nb))
                remaining_edges.discard(edge)
                avail[right_end].remove(best_nb)
                avail[best_nb].remove(right_end)
                
                path.append(best_nb)
                right_end = best_nb
            
            paths.append(path)
            
        # 3. Local Improvement: Merge Paths
        # If two paths P1=[a,...,b] and P2=[c,...,d] have b==c or b==d, etc., 
        # and the edge (b,c) is not in either (impossible, edge must be in one) 
        # Wait, the paths are edge-disjoint. 
        # Merging is possible if the end of one path is the start of another? 
        # No, they are disjoint in edges. 
        # If path1 ends at x and path2 starts at x, we can merge them into one path?
        # Yes! A path decomposition allows any set of paths. 
        # If we have path A: [u, ..., v] and path B: [v, ..., w], we can combine them 
        # into [u, ..., v, ..., w] provided the vertex v is not repeated in the combined path 
        # (which it isn't, since A and B only share v at the junction, and A,B are simple).
        # Actually, we must ensure the combined path is simple. 
        # Since A and B are edge-disjoint and only share vertex v, and both are simple, 
        # the union is simple iff no other vertex is shared.
        # In a graph, it's possible they share other vertices. 
        # However, for sparse ER graphs, it's rare. 
        # Let's try a simple merge: 
        # Build a map: vertex -> list of path indices ending or starting there.
        
        # This step can be expensive if done naively. 
        # Given n is small (<=30), and number of paths is small (<=20), 
        # we can try all pairs.
        
        merged = True
        while merged:
            merged = False
            # Create a mapping from vertex to (path_index, is_start/is_end)
            # Actually, simpler: 
            # For each pair of paths, check if they can be concatenated.
            
            new_paths = []
            used_paths = [False] * len(paths)
            
            # To make merging efficient, we can just iterate and try to merge current list.
            # But a simpler heuristic: 
            # If path i ends at v, and path j starts at v, merge them.
            # If path i ends at v, and path j ends at v (so j is [v, ...]), merge i + reversed(j)?
            # Yes.
            
            # Let's use a queue of merges.
            # Actually, since n is small, let's just do a few passes.
            
            for _ in range(3): # Max 3 passes of merging
                changed = False
                # Map end of path to path index
                # start_map: vertex -> list of (path_idx, 0)
                # end_map: vertex -> list of (path_idx, 1)
                
                # We'll just iterate all pairs. O(P^2 * L)
                P = len(paths)
                for i in range(P):
                    if not paths[i]: continue
                    for j in range(i+1, P):
                        if not paths[j]: continue
                        
                        # Check if paths[i] and paths[j] can be merged
                        # Case 1: end(i) == start(j)
                        # Case 2: end(i) == end(j) -> merge i + rev(j)
                        # Case 3: start(i) == start(j) -> merge rev(i) + j
                        # Case 4: start(i) == end(j) -> merge rev(i) + rev(j)
                        
                        merged_path = None
                        
                        if paths[i][-1] == paths[j][0]:
                            # Check simplicity: no common vertices except the junction
                            set_i = set(paths[i])
                            set_j = set(paths[j])
                            common = set_i & set_j
                            if common == {paths[i][-1]}:
                                merged_path = paths[i] + paths[j][1:]
                        elif paths[i][-1] == paths[j][-1]:
                            set_i = set(paths[i])
                            set_j = set(paths[j])
                            common = set_i & set_j
                            if common == {paths[i][-1]}:
                                merged_path = paths[i] + paths[j][::-1][1:]
                        elif paths[i][0] == paths[j][0]:
                            set_i = set(paths[i])
                            set_j = set(paths[j])
                            common = set_i & set_j
                            if common == {paths[i][0]}:
                                merged_path = paths[i][::-1] + paths[j][1:]
                        elif paths[i][0] == paths[j][-1]:
                            set_i = set(paths[i])
                            set_j = set(paths[j])
                            common = set_i & set_j
                            if common == {paths[i][0]}:
                                merged_path = paths[i][::-1] + paths[j][::-1][1:]
                        
                        if merged_path:
                            paths[i] = merged_path
                            paths[j] = []
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
            
            # Remove empty paths
            paths = [p for p in paths if p]
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 10:
            # Fallback for remaining graphs
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # Ensure we have 3 results
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
        
    return results
