import itertools
import random
import time
import sys

def construct():
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    edge_set = set(all_edges)
    
    # Helper to check if a specific set of edges forms a Q3
    # We need to check if the graph contains a subgraph isomorphic to Q3.
    
    # Precompute Q3 edges (0-7)
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if (i ^ j) & ((i ^ j) - 1) == 0 and (i ^ j) != 0:
                q3_edges.append((i, j))
                
    # Function to check if graph (edge_set) contains Q3
    def contains_q3(edge_set):
        # To speed up, we can check if there are 8 vertices with enough edges
        # But we need exact isomorphism.
        # Since n=16 is small, we can iterate 8-subsets.
        for subset in itertools.combinations(range(n), 8):
            # Check if induced subgraph on subset contains Q3
            # Get edges in subset
            sub_edges = set()
            for u in subset:
                for v in subset:
                    if u < v and (u, v) in edge_set:
                        sub_edges.add((u, v))
            
            # If less than 12 edges, no Q3
            if len(sub_edges) < 12:
                continue
                
            # Try to map Q3 vertices to subset vertices
            # We can use a backtracking search for the embedding
            # Map q3_vertex -> graph_vertex
            mapping = {}
            
            def backtrack(q3_v, used):
                if q3_v == 8:
                    return True
                for g_v in subset:
                    if g_v in used:
                        continue
                    # Check if adding this mapping is consistent with previous ones
                    # For all q3_u < q3_v that are mapped, if (q3_u, q3_v) is an edge in Q3,
                    # (mapping[q3_u], g_v) must be an edge in graph
                    # If not an edge in Q3, no constraint (non-edge can map to edge)
                    valid = True
                    for q3_u in range(q3_v):
                        if q3_u in mapping:
                            # Check edge (q3_u, q3_v) in Q3?
                            is_q3_edge = (q3_u, q3_v) in q3_edges or (q3_v, q3_u) in q3_edges
                            is_graph_edge = (mapping[q3_u], g_v) in edge_set or (g_v, mapping[q3_u]) in edge_set
                            if is_q3_edge and not is_graph_edge:
                                valid = False
                                break
                    if valid:
                        mapping[q3_v] = g_v
                        used.add(g_v)
                        if backtrack(q3_v + 1, used):
                            return True
                        del mapping[q3_v]
                        used.remove(g_v)
                return False
            
            if backtrack(0, set()):
                return True
        return False

    # This check is too slow for local search if called frequently.
    # We need a faster check or a different strategy.
    
    # Strategy: 
    # 1. Start with a random graph with p=0.8
    # 2. While contains_q3: remove an edge from a found Q3.
    # 3. Try to add edges back.
    
    # To make contains_q3 faster, we can precompute the list of all Q3 embeddings?
    # No, too many.
    
    # Let's optimize the check.
    # The backtracking above is O(8! * 12) per subset.
    # 12870 subsets * 40320 * 12 ~ 6e9 ops. Too slow.
    
    # Better check:
    # Q3 is 3-regular.
    # In the induced subgraph of 8 vertices, if it contains Q3, 
    # there must be 8 vertices such that the subgraph formed by them has a Q3 as a subgraph.
    
    # Alternative: Use the fact that Q3 has no triangles.
    # If the graph is triangle-free, it's Q3-free? No, Q3 is triangle-free, but a triangle-free graph can contain Q3.
    # Actually, if the graph is triangle-free, it cannot contain a triangle, but Q3 doesn't have triangles.
    
    # Let's try a different heuristic construction.
    # Partition 16 vertices into 4 sets of 4: A, B, C, D.
    # Include all edges between A-B, B-C, C-D, D-A? (Cycle of parts)
    # Include no edges within parts.
    # Include edges between A-C and B-D?
    
    # Let's try to build a graph and verify it.
    # If we can't verify quickly, we can't search.
    
    # Let's use a simpler verification: 
    # Check if any 8-vertex subset has a subgraph isomorphic to Q3.
    # We can pre-compute all 8! permutations of a fixed subset? No.
    
    # Let's use a library if available? No, only stdlib/np/scipy.
    
    # I will implement a faster check using bitmasks.
    # Represent edges as a 120-bit integer or similar?
    # Or just use sets.
    
    # The bottleneck is the check.
    # Let's try to reduce the number of subsets we check.
    # Only check subsets that have at least 12 edges.
    
    # Let's try to run the local search with a time limit.
    
    random.seed(42)
    current_edges = set()
    for e in all_edges:
        if random.random() < 0.7:
            current_edges.add(e)
            
    def find_q3(edge_set):
        # Returns one Q3 (as a set of edges) or None
        for subset in itertools.combinations(range(n), 8):
            # Quick degree check?
            # Q3 has 4 vertices of degree 3 in the subgraph? No, all degree 3.
            # So in the induced subgraph, the 8 vertices must have degree >= 3?
            # Not necessarily, the Q3 is a subgraph, not induced.
            # But the 12 edges of Q3 must be present.
            
            # Count edges in subset
            cnt = 0
            for u in subset:
                for v in subset:
                    if u < v and (u, v) in edge_set:
                        cnt += 1
            if cnt < 12:
                continue
            
            # Backtracking search
            # Order vertices in subset to prune?
            # Let's just try the backtracking.
            
            # To speed up, we can pre-check degrees.
            # In Q3, every vertex has degree 3.
            # So we need to pick 8 vertices such that we can assign them to Q3 vertices
            # such that the required edges exist.
            
            # Let's use the backtracking function defined earlier but optimized.
            
            # We'll use a global mapping
            mapping = [-1] * 8
            
            def bt(q3_v, used_mask):
                if q3_v == 8:
                    return True
                for g_v in subset:
                    if (1 << g_v) & used_mask:
                        continue
                    valid = True
                    for q3_u in range(q3_v):
                        if mapping[q3_u] != -1:
                            # Check edge (q3_u, q3_v)
                            xor = q3_u ^ q3_v
                            is_q3_edge = (xor & (xor - 1) == 0) and (xor != 0)
                            if is_q3_edge:
                                # Must be edge in graph
                                u1, v1 = mapping[q3_u], g_v
                                if u1 > v1: u1, v1 = v1, u1
                                if (u1, v1) not in edge_set:
                                    valid = False
                                    break
                    if valid:
                        mapping[q3_v] = g_v
                        if bt(q3_v + 1, used_mask | (1 << g_v)):
                            return True
                        mapping[q3_v] = -1
                return False
            
            if bt(0, 0):
                # Reconstruct edges
                q3_edges_found = set()
                for u in range(8):
                    for v in range(u+1, 8):
                        xor = u ^ v
                        if (xor & (xor - 1) == 0) and (xor != 0):
                            gu, gv = mapping[u], mapping[v]
                            if gu > gv: gu, gv = gv, gu
                            q3_edges_found.add((gu, gv))
                return q3_edges_found
        return None

    # Local Search
    start_time = time.time()
    timeout = 100 # Leave 20s for safety
    
    while time.time() - start_time < timeout:
        q3 = find_q3(current_edges)
        if q3 is None:
            # Try to add edges
            improved = False
            for e in all_edges:
                if e not in current_edges:
                    current_edges.add(e)
                    if find_q3(current_edges) is None:
                        improved = True
                    else:
                        current_edges.remove(e)
            if not improved:
                break
        else:
            # Remove an edge from q3
            # Pick the one that appears in most other Q3s? Hard to know.
            # Just pick random or first.
            e_to_remove = random.choice(list(q3))
            current_edges.remove(e_to_remove)
            
    return [[a, b] for a, b in current_edges]
