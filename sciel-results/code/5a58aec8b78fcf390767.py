import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    q = 5
    # Generate points of PG(2, q)
    # 25 affine points: (x, y, 1)
    # 5 points at infinity: (1, m, 0) for m in 0..4
    # Note: (0, 1, 0) is also a point at infinity if we consider all lines.
    # Standard set:
    # (x, y, 1) for x,y in F_q
    # (1, m, 0) for m in F_q
    # (0, 1, 0)
    # Total q^2 + q + 1 = 25 + 5 + 1 = 31.
    
    points = []
    # Affine
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Infinity 1
    for m in range(q):
        points.append((1, m, 0))
    # Infinity 2
    points.append((0, 1, 0))
    
    n = len(points)
    
    # Build adjacency matrix for polarity graph
    # Two points P, Q are adjacent if P . Q = 0
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            dot = (p[0] * q_pt[0] + p[1] * q_pt[1] + p[2] * q_pt[2]) % q
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append((i, j))
                
    # edges currently has 93 entries
    # Find all 4-cycles
    # A 4-cycle is a set of 4 vertices {u, v, w, x} such that u-v, v-w, w-x, x-u are edges.
    # We can find them by looking at common neighbors.
    # For each pair (u, v), let C = N(u) & N(v).
    # If |C| >= 2, then for any pair (w, x) in C, u-v-w-x-u is a 4-cycle.
    # We store the 4 edges of the cycle.
    
    c4s = []
    # Precompute neighbors
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    seen_cycles = set()
    for u in range(n):
        for v in neighbors[u]:
            if u < v:
                common = neighbors[u] & neighbors[v]
                if len(common) >= 2:
                    common_list = list(common)
                    for k in range(len(common_list)):
                        for l in range(k + 1, len(common_list)):
                            w = common_list[k]
                            x = common_list[l]
                            # Cycle u-v-w-x-u
                            # Edges: (u,v), (v,w), (w,x), (x,u)
                            # Normalize edge representation
                            e1 = (min(u,v), max(u,v))
                            e2 = (min(v,w), max(v,w))
                            e3 = (min(w,x), max(w,x))
                            e4 = (min(x,u), max(x,u))
                            cycle_edges = frozenset([e1, e2, e3, e4])
                            if cycle_edges not in seen_cycles:
                                seen_cycles.add(cycle_edges)
                                c4s.append(cycle_edges)
                                
    # We need to remove 3 edges to hit all c4s
    # Current edges: list of tuples
    # We want to find a subset R of 3 edges from 'edges' such that every c4 in c4s intersects R.
    
    # Heuristic: Greedy
    # Sort edges by frequency of appearance in c4s
    edge_freq = {e: 0 for e in edges}
    for c4 in c4s:
        for e in c4:
            edge_freq[e] += 1
            
    # Try to find a hitting set of size 3
    # Since 3 is small, we can backtrack if greedy fails, or just try a few permutations.
    # But greedy is likely to work for this structure.
    
    def find_hitting_set(edges_list, c4_list, k):
        if not c4_list:
            return []
        if k == 0:
            return None
        if time.time() > deadline:
            return None
            
        # Pick edge with max freq in remaining c4s
        freq = {e: 0 for e in edges_list}
        for c4 in c4_list:
            for e in c4:
                if e in freq:
                    freq[e] += 1
        
        # Find max
        max_freq = -1
        best_edge = None
        for e in edges_list:
            if freq[e] > max_freq:
                max_freq = freq[e]
                best_edge = e
                
        if best_edge is None:
            return None
            
        # Try removing best_edge
        new_edges = [e for e in edges_list if e != best_edge]
        new_c4s = [c4 for c4 in c4_list if best_edge not in c4]
        
        res = find_hitting_set(new_edges, new_c4s, k - 1)
        if res is not None:
            return [best_edge] + res
            
        # If greedy choice fails, we might need to try other edges.
        # For k=3, backtracking is feasible if the search space is pruned.
        # However, for this specific graph, the greedy choice is often correct or close.
        # To be safe, we can try a few top candidates.
        
        # Let's implement a limited backtrack for the top 5 most frequent edges
        candidates = sorted([e for e in edges_list if freq[e] > 0], key=lambda e: -freq[e])[:5]
        
        for e in candidates:
            if e == best_edge:
                continue # Already tried
            new_edges = [ed for ed in edges_list if ed != e]
            new_c4s = [c4 for c4 in c4_list if e not in c4]
            res = find_hitting_set(new_edges, new_c4s, k - 1)
            if res is not None:
                return [e] + res
                
        return None

    # We need to remove 3 edges.
    # The polarity graph has 93 edges. We want 90.
    # So we remove 3 edges.
    # But wait, if we remove 3 edges, do we guarantee it becomes C4-free?
    # Yes, if the hitting set hits all C4s.
    
    removal_set = find_hitting_set(edges, c4s, 3)
    
    if removal_set is None:
        # Fallback: Just remove 3 edges arbitrarily? Or use a different base graph?
        # If we can't find a C4-free subgraph of size 90, we might have more C4s.
        # But ex(31, C4)=90 implies such a graph exists.
        # Let's try a greedy removal without strict hitting set requirement, just minimizing C4s.
        # Or simply return the graph with 91 edges from a greedy C4-free construction.
        
        # Alternative: Build a C4-free graph greedily.
        # Start with empty graph. Add edges that don't create C4s.
        # Stop at 90 edges. Then add 1 edge.
        
        adj_greedy = [[False] * n for _ in range(n)]
        edges_greedy = []
        
        # Order edges by some heuristic? Random?
        # To make it deterministic, use the edge list from polarity graph but shuffle?
        # Or just iterate.
        
        import random
        random.seed(42)
        candidate_edges = list(edges)
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if len(edges_greedy) >= 90:
                break
            if adj_greedy[u][v]:
                continue
            # Check if adding (u,v) creates a C4
            # Check common neighbors
            common = neighbors[u] & neighbors[v]
            # Wait, neighbors is from the FULL polarity graph.
            # We need neighbors in the GREEDY graph.
            
            # This approach is flawed because 'neighbors' is static.
            # We need dynamic neighbor sets.
            
        # Let's restart the greedy construction properly if hitting set fails.
        
        adj_greedy = [[False] * n for _ in range(n)]
        edges_greedy = []
        neigh_greedy = [set() for _ in range(n)]
        
        # Use a fixed order for reproducibility, maybe sorted by index
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Prioritize edges from the polarity graph
        polarity_edges_set = set(edges)
        # Sort: Polarity edges first, then others
        all_possible_edges.sort(key=lambda e: 0 if e in polarity_edges_set else 1)
        
        for u, v in all_possible_edges:
            if len(edges_greedy) >= 90:
                break
            # Check if u and v have a common neighbor in the current greedy graph
            common = neigh_greedy[u] & neigh_greedy[v]
            if len(common) == 0:
                # Safe to add
                edges_greedy.append((u, v))
                adj_greedy[u][v] = True
                adj_greedy[v][u] = True
                neigh_greedy[u].add(v)
                neigh_greedy[v].add(u)
                
        if len(edges_greedy) < 90:
            # Should not happen if ex=90
            pass
            
        # Now add the 91st edge
        best_edge = None
        min_common = float('inf')
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj_greedy[u][v]:
                    continue
                common = neigh_greedy[u] & neigh_greedy[v]
                if len(common) < min_common:
                    min_common = len(common)
                    best_edge = (u, v)
                    
        if best_edge:
            edges_greedy.append(best_edge)
            
        return [[u, v] for u, v in edges_greedy]
        
    # If hitting set found
    # Construct final graph
    final_edges = [e for e in edges if e not in removal_set]
    
    # Rebuild neighbors for final_edges (90 edges)
    neighbors_final = [set() for _ in range(n)]
    for u, v in final_edges:
        neighbors_final[u].add(v)
        neighbors_final[v].add(u)
        
    # Find best edge to add
    best_edge = None
    min_common = float('inf')
    
    final_edges_set = set(final_edges)
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in final_edges_set:
                continue
            # Check if (v, u) is in set? No, we use min/max tuple
            if (min(u,v), max(u,v)) in final_edges_set:
                continue
                
            common = neighbors_final[u] & neighbors_final[v]
            if len(common) < min_common:
                min_common = len(common)
                best_edge = (u, v)
                
    if best_edge:
        final_edges.append(best_edge)
        
    # Ensure we return exactly 91 edges
    if len(final_edges) != 91:
        # Fallback to greedy if something went wrong
        # (This shouldn't happen if logic is correct)
        pass
        
    return [[u, v] for u, v in final_edges]

# EVOLVE-BLOCK-END
