import itertools
import random
import time

def construct():
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Precompute Q3 structure
    # Vertices 0..7
    # Edges where xor has popcount 1
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor & (xor - 1) == 0 and xor != 0:
                q3_edges.append((i, j))
                
    # Map edge index in q3_edges for quick lookup? 
    # For backtracking, we just need the list.
    
    def find_q3_in_subset(S, edge_set):
        """
        S: tuple of 8 vertices
        edge_set: set of edges in the full graph
        Returns a list of edges (from S) forming a Q3, or None.
        """
        # Backtracking search for bijection phi: {0..7} -> S
        # such that for all (u,v) in q3_edges, (phi(u), phi(v)) in edge_set
        
        # Optimization: Check degrees.
        # In Q3, every vertex has degree 3.
        # So for any vertex x in S, it must be connected to at least 3 other vertices in S 
        # that are part of the Q3.
        # This is a necessary condition but not sufficient for the whole graph, 
        # but helpful for pruning.
        
        # Actually, simple backtracking on 8 vertices is fast enough.
        # Order Q3 vertices to maximize constraint propagation?
        # All have degree 3.
        
        assignment = {}
        
        def backtrack(q3_idx, current_assign):
            if q3_idx == 8:
                # Verify all edges
                for u, v in q3_edges:
                    eu = current_assign[u]
                    ev = current_assign[v]
                    e = (min(eu, ev), max(eu, ev))
                    if e not in edge_set:
                        return None
                return [(min(current_assign[u], current_assign[v]), max(current_assign[u], current_assign[v])) for u, v in q3_edges]
            
            # Try assigning q3_idx to each unassigned vertex in S
            # Pruning: 
            # If we assign q3_idx to vertex v in S, 
            # v must be connected to the already assigned neighbors of q3_idx in Q3.
            
            neighbors_in_q3 = [u for u in range(8) if (u, q3_idx) in q3_edges or (q3_idx, u) in q3_edges]
            
            for v in S:
                if v in current_assign.values():
                    continue
                
                # Check compatibility with already assigned neighbors
                valid = True
                for u in neighbors_in_q3:
                    if u in current_assign:
                        eu = current_assign[u]
                        e = (min(eu, v), max(eu, v))
                        if e not in edge_set:
                            valid = False
                            break
                if not valid:
                    continue
                
                current_assign[q3_idx] = v
                result = backtrack(q3_idx + 1, current_assign)
                if result:
                    return result
                del current_assign[q3_idx]
                
            return None

        # Initial pruning: 
        # If any vertex in S has degree < 3 in the induced subgraph on S, 
        # it cannot be part of a 3-regular Q3.
        # Let's calculate degrees in induced subgraph.
        deg = {v: 0 for v in S}
        S_set = set(S)
        for u in S:
            for v in S:
                if u < v:
                    e = (u, v)
                    if e in edge_set:
                        deg[u] += 1
                        deg[v] += 1
        
        for v in S:
            if deg[v] < 3:
                return None
                
        return backtrack(0, {})

    def graph_has_q3(edge_set):
        """Find any Q3 in the graph defined by edge_set."""
        for S in itertools.combinations(range(n), 8):
            res = find_q3_in_subset(S, edge_set)
            if res:
                return res
        return None

    # Heuristic Search
    random.seed(42)
    
    # Start with a dense random graph
    # p=0.8 is a good start
    current_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.8:
                current_edges.add((u, v))
                
    # Phase 1: Remove edges to eliminate all Q3s
    # We do this repeatedly.
    # To be efficient, we can just find one Q3 and remove an edge.
    # But we want to maximize edges.
    # Strategy: 
    # 1. While Q3 exists, find one, remove an edge from it.
    #    Which edge? Randomly, or the one that appears in the most Q3s?
    #    Counting occurrences is expensive.
    #    Let's try removing a random edge from the found Q3.
    
    max_iter = 1000
    iter_count = 0
    while graph_has_q3(current_edges) and iter_count < max_iter:
        q3_edges_found = graph_has_q3(current_edges)
        if not q3_edges_found:
            break
        # Remove a random edge from this Q3
        e_to_remove = random.choice(q3_edges_found)
        current_edges.remove(e_to_remove)
        iter_count += 1
        
    # Phase 2: Add back edges
    # Try to add every missing edge.
    # If adding it doesn't create a Q3, add it.
    # To check "doesn't create a Q3", we can check if the new edge is part of a Q3.
    # A new edge (u,v) creates a Q3 only if there is a Q3 containing (u,v).
    # We can check this by looking at Q3s that contain (u,v).
    # How many such Q3s?
    # In Q3, each edge is in ... 
    # Total Q3s in K16 is huge.
    # But we can just run `graph_has_q3` on the set `current_edges | {(u,v)}`.
    # This is $O(\binom{16}{8} \times \text{backtrack})$.
    # With 120 edges, this might be slow if done for every missing edge (120 missing).
    # 120 * 12870 * backtrack ~ 1.5M * backtrack.
    # Backtrack is small (8 vars). Should be fine in Python if optimized?
    # Maybe 1-2 seconds per check? 120 * 2s = 240s. Too slow.
    
    # Optimization for Phase 2:
    # Only check Q3s that *could* be formed by the new edge.
    # A Q3 containing edge (u,v) must have u and v as adjacent vertices in the cube.
    # In Q3, adjacent vertices have Hamming distance 1.
    # So we only need to check 8-vertex subsets S that contain u and v,
    # AND such that u and v can be mapped to adjacent cube vertices.
    # This reduces the search space significantly?
    # Actually, `find_q3_in_subset` already checks the edges.
    # The bottleneck is the number of subsets (12870).
    
    # Better Phase 2:
    # Instead of checking every missing edge individually with a full scan,
    # we can iterate over missing edges, and for each, 
    # check if it completes a Q3.
    # We can precompute for each edge (u,v), the list of 8-subsets that contain u,v.
    # There are $\binom{14}{6} = 3003$ such subsets for each edge.
    # 120 edges * 3003 subsets * backtrack.
    # Still might be slow.
    
    # Alternative:
    # Just do a few iterations of "Try to add random missing edge, check full graph".
    # Or, since we just removed edges to break Q3s, the graph is "locally" Q3-free.
    # Adding an edge (u,v) only creates a Q3 if there is a path of length 3 between u and v 
    # that completes the cube structure? No, Q3 is more complex.
    
    # Let's try a simpler Phase 2:
    # Sort missing edges.
    # For each missing edge, tentatively add it.
    # Check if the graph has a Q3.
    # If yes, revert.
    # To speed up `graph_has_q3`, we can cache the last found Q3? No.
    
    # Let's optimize `find_q3_in_subset` further.
    # The backtracking is the main cost.
    # We can use bitmasks for edges in the subset?
    
    # Given the time limit, let's stick to the simple approach but limit the number of add attempts.
    # Or, just do Phase 1 more carefully.
    
    # Revised Strategy:
    # 1. Start with K16.
    # 2. While Q3 exists, find one, remove an edge.
    #    This will end up with a Q3-free graph.
    #    The number of edges removed will be the size of the hitting set.
    #    This is likely to result in a decent number of edges.
    
    current_edges = set(all_edges)
    
    # To make it faster, we can shuffle the order of edge removals?
    # No, just remove.
    
    # Optimization:
    # When finding a Q3, we can choose which edge to remove.
    # If we remove an edge that is "critical" for many Q3s, we might remove fewer total edges.
    # However, finding "critical" edges is hard.
    # Random removal is a good heuristic.
    
    # Let's run the removal loop.
    # To ensure we finish in time, we add a timeout check.
    start_time = time.time()
    
    while time.time() - start_time < 90: # Leave 30s for final check/return
        q3_found = graph_has_q3(current_edges)
        if not q3_found:
            break
        # Remove a random edge from the found Q3
        # We can try to remove the edge that appears in the most other Q3s?
        # Let's just pick random.
        e_to_remove = random.choice(q3_found)
        current_edges.remove(e_to_remove)
        
    # After the loop, we have a Q3-free graph.
    # Now try to add edges back.
    # We will try to add edges in a random order.
    missing_edges = [e for e in all_edges if e not in current_edges]
    random.shuffle(missing_edges)
    
    for e in missing_edges:
        if time.time() - start_time > 110:
            break
        # Tentatively add
        current_edges.add(e)
        # Check if Q3 free
        if not graph_has_q3(current_edges):
            # Keep it
            pass
        else:
            # Revert
            current_edges.remove(e)
            
    return [[u, v] for u, v in current_edges]
