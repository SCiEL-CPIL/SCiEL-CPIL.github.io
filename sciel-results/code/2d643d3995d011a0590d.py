import time
import random
import numpy as np
from itertools import combinations

def construct():
    # Define the two instances
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 150.0
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        # Leave 10s safety margin for the rest of the program
        remaining_time = total_time_limit - elapsed - 10.0
        
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time: give the first instance 40% of remaining, second gets the rest (minus margin)
            # Or simply split remaining time evenly? 
            # Let's give each instance a proportional share of the total budget, but capped by remaining.
            # Simpler: just use the remaining time for this instance, but ensure the next one has time.
            # If it's the first instance, we should reserve time for the second.
            # Let's say we want to spend at most 60s on the first and 80s on the second.
            # But we must respect the hard deadline.
            
            # Heuristic: 
            # Instance 1 (n=32): max 60s
            # Instance 2 (n=64): max 80s
            # But if we are late, we must shrink.
            
            max_time_for_this = 60.0 if idx == 0 else 80.0
            instance_time_limit = min(remaining_time, max_time_for_this)
            
            # If this is the last instance, we can use all remaining time
            if idx == len(instances) - 1:
                instance_time_limit = remaining_time
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
            
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
        
    # Build adjacency list for residual graph
    res_adj = [set() for _ in range(n)]
    edgeset = set(edges)
    for u, v in edges:
        res_adj[u].add(v)
        res_adj[v].add(u)
        
    cliques = []
    start_time = time.time()
    
    # Phase 1: Greedy extraction of large cliques
    while True:
        current_time = time.time()
        if current_time - start_time > time_limit * 0.7: # Use 70% of time for extraction
            break
            
        if not any(res_adj):
            break
            
        # Find a large clique
        clique = find_large_clique(res_adj, n)
        
        if not clique:
            # Should not happen if edges exist, but safety check
            # Find any edge
            found = False
            for u in range(n):
                if res_adj[u]:
                    v = next(iter(res_adj[u]))
                    clique = [u, v]
                    found = True
                    break
            if not found:
                break
                
        # Remove edges in clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                res_adj[u].discard(v)
                res_adj[v].discard(u)
                
        cliques.append(clique)
        
    # Phase 2: Merge optimization
    # We have a list of cliques. We want to merge them if possible to reduce count.
    # A merge is valid if the union of two cliques is also a clique.
    # This requires that every vertex in clique A is connected to every vertex in clique B.
    
    # We need to check if two cliques can be merged.
    # Let C1 and C2 be two cliques. They can be merged if for all u in C1, v in C2, (u,v) is an edge in the original graph.
    # Note: The original graph edges are fixed. The residual graph only tracks uncovered edges.
    # However, the definition of a clique partition requires that the sets themselves are cliques in the ORIGINAL graph.
    # And every edge in the original graph must be in exactly one set.
    
    # Wait, the current `cliques` list contains sets of vertices.
    # When we extracted a clique from the residual graph, we ensured it was a clique in the residual graph.
    # Since the residual graph is a subgraph of the original, it is also a clique in the original graph.
    # So each `clique` in the list is a valid clique.
    
    # Now we try to merge them.
    # If we merge C1 and C2, the new set is C1 U C2.
    # This is valid if C1 U C2 is a clique in the original graph.
    # And we must ensure that no edge is covered twice.
    # Since C1 and C2 are disjoint in terms of edges?
    # No, the vertices might overlap?
    # In our extraction, we removed edges from the residual graph.
    # The cliques extracted are edge-disjoint by construction (since we remove the edges they cover).
    # However, the vertex sets might overlap.
    # Example: C1 = {1, 2}, C2 = {2, 3}.
    # Edge (1,2) is in C1. Edge (2,3) is in C2.
    # If we merge them to {1, 2, 3}, we cover edges (1,2), (2,3), and (1,3).
    # But edge (1,3) might have been covered by a different clique, or might not exist.
    # If (1,3) exists and is not covered, then merging is good.
    # If (1,3) is already covered by another clique, merging is invalid (double coverage).
    # If (1,3) does not exist, merging is invalid (not a clique).
    
    # So, to merge C1 and C2:
    # 1. Check if all cross-edges exist in the original graph.
    # 2. Check if all cross-edges are currently uncovered (i.e., in the residual graph? No, the residual graph only contains uncovered edges).
    #    Actually, if a cross-edge is in the residual graph, it is uncovered.
    #    If a cross-edge is NOT in the residual graph, it is either covered or doesn't exist.
    #    If it doesn't exist, we can't merge.
    #    If it is covered, we can't merge.
    
    # So the condition is: For all u in C1, v in C2, (u,v) must be in the residual graph.
    # If this holds, then merging C1 and C2 creates a new clique that covers the edges of C1, edges of C2, and the cross-edges.
    # The cross-edges were previously uncovered, so they become covered.
    # The edges of C1 and C2 were already covered.
    # So the new clique covers exactly the union of edges.
    # And since the cross-edges were in the residual graph, they are valid edges in the original graph.
    # And since C1 and C2 are cliques, and all cross-edges exist, the union is a clique.
    
    # So the merge condition is simply: All cross-edges between C1 and C2 are in the residual graph.
    
    # We will perform a greedy merge loop.
    
    # To make this efficient, we can represent the current state by the list of cliques and the residual graph.
    # Note: The residual graph currently contains edges NOT covered by any clique in `cliques`.
    # If we merge C1 and C2, we must add the cross-edges to the "covered" set, meaning we should remove them from the residual graph?
    # No, the residual graph is just a helper for extraction.
    # For the merge phase, we need to know which edges are covered.
    # Actually, if we merge, we are just changing the partition.
    # The set of covered edges remains the same (the union of edges in all cliques).
    # The only thing that changes is the grouping.
    # So we don't need to update the residual graph for the purpose of validity, except that we need to check if cross-edges are available.
    # "Available" means they are in the original graph AND they are not covered by any OTHER clique.
    # Since the cliques are edge-disjoint, an edge is covered if and only if it is in exactly one clique.
    # So if a cross-edge is in the residual graph, it is not covered by any clique.
    # So the condition "cross-edge is in residual graph" is sufficient.
    
    # However, as we merge, the set of cliques changes.
    # If we merge C1 and C2 into C12, the edges of C12 are E(C1) U E(C2) U Cross(C1, C2).
    # The cross-edges are removed from the "uncovered" pool.
    # So if we later try to merge C12 with C3, we need to check if cross-edges between C12 and C3 are in the residual graph.
    # But wait, if we merged C1 and C2, we "used up" the cross-edges between C1 and C2.
    # So they should no longer be in the residual graph for the purpose of future merges?
    # Yes, because they are now covered by C12.
    # So we should update the residual graph when we merge.
    
    # Let's maintain `res_adj` as the set of edges NOT covered by the current partition.
    # Initially, after extraction, `res_adj` contains edges not covered by the extracted cliques.
    # When we merge C1 and C2:
    # 1. Check if all cross-edges are in `res_adj`.
    # 2. If yes, remove them from `res_adj`.
    # 3. Replace C1 and C2 with C1 U C2 in the list.
    
    # This is a valid approach.
    
    # Optimization:
    # We want to minimize the number of cliques.
    # Merging two cliques reduces the count by 1.
    # We should prioritize merges that are "easy" or lead to larger cliques?
    # A greedy approach: repeatedly find a mergeable pair and merge it.
    # To be smarter, we can look for the pair that maximizes the number of cross-edges?
    # Or just any merge?
    # Let's try a simple greedy: find any mergeable pair, merge, repeat.
    # To speed up, we can use a priority queue or just scan.
    # Given n is small (32, 64), the number of cliques is likely small (maybe < 100).
    # Scanning all pairs is O(K^2 * N^2) which is fine.
    
    merge_start = time.time()
    merge_limit = time_limit - (merge_start - start_time) - 1.0 # 1s safety
    
    changed = True
    while changed and (time.time() - start_time < start_time + time_limit - 1.0):
        changed = False
        best_pair = None
        best_size = 0
        
        # List of current cliques
        # We need to check pairs (i, j)
        k = len(cliques)
        for i in range(k):
            c1 = cliques[i]
            if not c1: continue
            for j in range(i + 1, k):
                c2 = cliques[j]
                if not c2: continue
                
                # Check if all cross-edges are in res_adj
                # c1 and c2 are lists of vertices
                # We need to check if for all u in c1, v in c2, v in res_adj[u]
                
                # Quick check: do they share vertices?
                # If they share a vertex, say x, then (x, u) for u in c1\{x} and (x, v) for v in c2\{x} are internal edges, already covered.
                # But the cross-edges between (c1\{x}) and (c2\{x}) must be in res_adj.
                # Actually, the definition of "cross-edges" in the context of merging two cliques C1 and C2 is simply all edges (u,v) with u in C1, v in C2.
                # If u=v, it's not an edge.
                # So we check all pairs (u,v) with u in C1, v in C2, u!=v.
                
                # Optimization: If C1 and C2 are disjoint, we check all |C1|*|C2| pairs.
                # If they share vertices, we still check all pairs.
                
                valid = True
                for u in c1:
                    for v in c2:
                        if u == v: continue
                        if v not in res_adj[u]:
                            valid = False
                            break
                    if not valid: break
                    
                if valid:
                    # This pair can be merged
                    # Prefer larger merged cliques?
                    # The size of the merged clique is len(c1) + len(c2) - len(intersection)
                    # Let's just pick the first one or the one with max size.
                    merged_size = len(c1) + len(c2)
                    if merged_size > best_size:
                        best_size = merged_size
                        best_pair = (i, j)
        
        if best_pair is not None:
            i, j = best_pair
            c1 = cliques[i]
            c2 = cliques[j]
            
            # Remove cross-edges from res_adj
            for u in c1:
                for v in c2:
                    if u == v: continue
                    res_adj[u].discard(v)
                    res_adj[v].discard(u)
                    
            # Merge cliques
            # Union of sets
            new_clique = list(set(c1) | set(c2))
            
            # Replace c1 with new_clique, remove c2
            cliques[i] = new_clique
            cliques.pop(j)
            changed = True
            
    # Final check: ensure all edges are covered
    # The residual graph should be empty.
    # If not, we have missed some edges. This shouldn't happen if extraction was complete.
    # But if extraction stopped early due to time, there might be edges left.
    # In that case, we must add them as individual edges.
    
    # Collect remaining edges
    remaining_edges = []
    for u in range(n):
        for v in res_adj[u]:
            if u < v:
                remaining_edges.append([u, v])
    
    if remaining_edges:
        # Add them to cliques
        cliques.extend(remaining_edges)
        # Note: These are single edges, so they are cliques of size 2.
        # And they are disjoint from the others because they were in the residual graph.
        
    return cliques

def find_large_clique(res_adj, n):
    # Heuristic: Try greedy expansion from top K highest degree vertices
    # Calculate degrees
    degrees = [len(res_adj[u]) for u in range(n)]
    
    # Sort vertices by degree
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_clique = []
    
    # Try top K vertices
    K = min(20, n) # Try 20 starting points
    
    for start_v in sorted_vertices[:K]:
        if degrees[start_v] == 0:
            continue
            
        # Greedy expansion
        C = [start_v]
        P = set(res_adj[start_v])
        
        while P:
            # Pick u in P with max degree in P
            best_u = None
            best_deg = -1
            for u in P:
                # Degree of u in P is |P & res_adj[u]|
                deg = len(P & res_adj[u])
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
            if best_u is None:
                break
            C.append(best_u)
            P = P & res_adj[best_u]
            
        if len(C) > len(best_clique):
            best_clique = C
            
    return best_clique
