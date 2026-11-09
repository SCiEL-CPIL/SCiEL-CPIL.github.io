import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_edges = None
    best_count = -1
    
    n = 20
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if a set of edges contains a K3,3
    # Returns a list of edges involved in K3,3s, or a dict of edge -> count
    def get_violation_counts(edges):
        # Build adjacency sets for fast intersection
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edge_to_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        violation_counts = {e: 0 for e in edges}
        
        # Iterate over all triples of vertices
        for u, v, w in itertools.combinations(range(n), 3):
            # Common neighbors of u, v, w
            common = adj[u].intersection(adj[v]).intersection(adj[w])
            # Remove u, v, w from common to ensure disjoint sets
            common.discard(u)
            common.discard(v)
            common.discard(w)
            
            if len(common) >= 3:
                # There are K3,3 subgraphs. 
                # We just need to mark the edges involved.
                # Any 3 vertices from 'common' form the other part.
                # To be thorough, we could count all combinations, but that's expensive.
                # Instead, we can just pick one set of 3 from common to identify a specific K3,3
                # and add its edges to the count.
                # Or, we can increment the count for all edges connecting {u,v,w} to 'common'
                # proportional to the number of K3,3s? 
                # Simple heuristic: Just find one K3,3.
                sample_common = list(common)[:3]
                for s1 in [u, v, w]:
                    for s2 in sample_common:
                        e = (s1, s2) if s1 < s2 else (s2, s1)
                        if e in violation_counts:
                            violation_counts[e] += 1
                            
        return violation_counts

    def greedy_optimize(initial_edges):
        edges = set(initial_edges)
        
        while time.time() < deadline:
            # Check for violations
            # We need a fast way to find if ANY violation exists and which edge to remove
            # Let's find one violation first to see if we are done.
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            found_violation = False
            edges_to_remove = None
            max_count = 0
            best_edge = None
            
            # To be more robust, let's count frequencies over all violations found in a pass
            # But scanning all 1140 triples and counting might be heavy if we do it 100 times.
            # 1140 * 100 = 114,000 operations. Very fast.
            
            counts = {e: 0 for e in edges}
            
            for u, v, w in itertools.combinations(range(n), 3):
                common = adj[u].intersection(adj[v]).intersection(adj[w])
                common.discard(u)
                common.discard(v)
                common.discard(w)
                
                if len(common) >= 3:
                    found_violation = True
                    # Pick 3 from common
                    sample_common = list(common)[:3]
                    for s1 in [u, v, w]:
                        for s2 in sample_common:
                            e = (s1, s2) if s1 < s2 else (s2, s1)
                            if e in counts:
                                counts[e] += 1
                    # We can break early if we want to save time, but counting all is better for "most central" edge
                    # However, just finding one is enough to remove an edge.
                    # Let's continue scanning to get a better "most central" estimate?
                    # For n=20, it's cheap.
            
            if not found_violation:
                break
                
            # Find the edge with the highest count
            # If counts are all 0 (shouldn't happen if found_violation is True and we updated counts), 
            # we might have a bug. 
            # Ensure we picked an edge from the violation.
            # The counts dict was updated.
            
            if max(counts.values()) > 0:
                best_edge = max(counts, key=counts.get)
                edges.remove(best_edge)
            else:
                # Fallback: remove any edge from a found violation
                # Re-scan to find one if counts failed (unlikely)
                for u, v, w in itertools.combinations(range(n), 3):
                    common = adj[u].intersection(adj[v]).intersection(adj[w])
                    common.discard(u)
                    common.discard(v)
                    common.discard(w)
                    if len(common) >= 3:
                        s2 = next(iter(common))
                        e = (u, s2) if u < s2 else (s2, u)
                        edges.remove(e)
                        break
                        
        return list(edges)

    # Run multiple trials
    # Trial 1: Complete graph
    best_edges = greedy_optimize(all_possible_edges)
    best_count = len(best_edges)
    
    # Trial 2+: Random graphs with high density
    for i in range(50):
        if time.time() > deadline:
            break
            
        # Random graph with p=0.8
        random_edges = [e for e in all_possible_edges if random.random() < 0.8]
        
        result = greedy_optimize(random_edges)
        
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result
            
    return [[u, v] for u, v in best_edges]
