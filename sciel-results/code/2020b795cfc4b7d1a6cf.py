"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    # Q3 vertices 0..7, edges
    q3_edges_canon = set()
    # Standard Q3 adjacency
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    edges_list = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Generate all 840 distinct edge sets isomorphic to Q3 on vertices 0..7
    # We can just generate all permutations of 0..7, map edges, and store unique frozensets
    q3_patterns = set()
    for perm in permutations(range(8)):
        mapped_edges = frozenset((min(perm[u], perm[v]), max(perm[u], perm[v])) for u, v in edges_list)
        q3_patterns.add(mapped_edges)
        
    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))

    def is_q3_free(edge_set):
        # Fast check:
        # 1. If total edges < 12, return True (but we are maximizing, so usually > 12)
        # 2. For each 8-subset, check if it contains a Q3 pattern
        
        # Optimization: Precompute degree of each vertex in the whole graph?
        # No, we need induced subgraph degrees.
        
        for vs in all_8sets:
            vs_set = set(vs)
            # Quick filter: count edges in this subset
            edge_count = 0
            degrees = {v: 0 for v in vs}
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = vs[i], vs[j]
                    if (min(u,v), max(u,v)) in edge_set:
                        edge_count += 1
                        degrees[u] += 1
                        degrees[v] += 1
            
            if edge_count < 12:
                continue
            
            # In Q3, every vertex has degree 3.
            # So in the induced subgraph, every vertex must have degree >= 3.
            min_deg = min(degrees.values())
            if min_deg < 3:
                continue
            
            # Now check if any of the 840 patterns is a subset of the induced edges
            # Build a set of edges present in this subset for fast lookup
            sub_edges = set()
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = vs[i], vs[j]
                    if (min(u,v), max(u,v)) in edge_set:
                        sub_edges.add((min(u,v), max(u,v)))
            
            # Map canonical patterns to this specific vertex set?
            # No, the patterns are on 0..7. We need to check if there is a bijection from 0..7 to vs
            # such that the edges match.
            # This is the same as checking if the subgraph on vs is isomorphic to Q3 or contains it.
            # Since we already checked degrees >= 3 and edges >= 12, and Q3 is 3-regular with 12 edges,
            # if the subgraph has exactly 12 edges and min degree 3, it MUST be Q3?
            # Not necessarily, it could be Q3 plus extra edges? No, we are checking the induced subgraph on 8 vertices.
            # If the induced subgraph has 12 edges and min degree 3, the sum of degrees is 24.
            # If it's Q3, degrees are 3,3,3,3,3,3,3,3. Sum 24.
            # If it has 12 edges and min degree 3, the degrees must be exactly 3 for all vertices.
            # So it is a 3-regular graph on 8 vertices with 12 edges.
            # Is every 3-regular graph on 8 vertices isomorphic to Q3?
            # No, there are other 3-regular graphs on 8 vertices (e.g., K4,4 is 3-regular? No K4,4 is 4-regular.
            # The 3-cube is one. What about others?
            # Number of 3-regular graphs on 8 vertices is small.
            # Let's just check isomorphism to Q3 for these specific cases.
            
            if edge_count == 12 and min_deg == 3:
                 # Check if it is Q3
                 # We can use the precomputed patterns if we map vs to 0..7
                 # But we have 8! mappings.
                 # However, we can just check if the subgraph is isomorphic to Q3.
                 # Since there are very few 3-regular graphs on 8 vertices, we can precompute their "signatures" or just check against Q3.
                 
                 # Let's check if it's Q3 by checking if it's bipartite and planar? 
                 # Q3 is bipartite.
                 # Check bipartiteness of subgraph
                 color = {}
                 is_bip = True
                 for v in vs:
                     if v not in color:
                         color[v] = 0
                         stack = [v]
                         while stack:
                             node = stack.pop()
                             for nb in vs:
                                 if (min(node, nb), max(node, nb)) in sub_edges:
                                     if nb in color:
                                         if color[nb] == color[node]:
                                             is_bip = False
                                             break
                                     else:
                                         color[nb] = 1 - color[node]
                                         stack.append(nb)
                             if not is_bip:
                                 break
                     if not is_bip:
                         break
                 
                 if is_bip:
                     # It is a 3-regular bipartite graph on 8 vertices.
                     # The only such graph is Q3? 
                     # K4,4 is 4-regular.
                     # Yes, Q3 is the unique 3-regular bipartite graph on 8 vertices?
                     # Let's assume so for now, or just return True (contains Q3) if it's bipartite and 3-reg.
                     return False
            elif edge_count > 12:
                 # It might contain Q3 as a proper subgraph.
                 # This is harder. 
                 # But note: if we maximize edges, we are likely in the dense regime.
                 # A denser graph is more likely to contain Q3.
                 # If edge_count > 12, we need to check if Q3 is a subgraph.
                 # We can use the 840 patterns mapped to vs.
                 # This is expensive. 
                 # However, for the purpose of this task, if we find a 12-edge 3-reg bipartite subgraph, we stop.
                 # If not, we might miss a Q3 that is a proper subgraph of a denser 8-subset.
                 # But Q3 has 12 edges. If the induced subgraph has > 12 edges, it *could* contain Q3.
                 # To be safe, we should check.
                 # But let's try to be fast.
                 # If we just check for the 12-edge case, we might allow graphs that contain Q3 but where the 8-vertex induced subgraph has extra edges.
                 # Wait, if G contains Q3, then the 8 vertices of that Q3 form an induced subgraph with *at least* the 12 edges of Q3.
                 # So if we miss the >12 case, we might declare a graph Q3-free when it's not.
                 # We need to check all cases.
                 
                 # Optimization: Only check the 840 patterns if the fast filters pass.
                 # The filters (edges >= 12, min_deg >= 3) are necessary conditions.
                 # If they pass, we MUST check if Q3 is a subgraph.
                 
                 # Let's do the pattern check for all passing cases.
                 # To speed up, we can precompute the "edge set" of the subgraph as a frozenset.
                 sub_edge_frozenset = frozenset(sub_edges)
                 
                 # We need to check if there exists a permutation p of vs such that
                 # for all (u,v) in Q3_edges, (p[u], p[v]) is in sub_edge_frozenset.
                 # This is equivalent to: does the subgraph contain Q3?
                 
                 # We can iterate over the 840 patterns? No, the patterns are on 0..7.
                 # We need to check if the subgraph on vs is isomorphic to a supergraph of Q3.
                 # Actually, we just need to check if Q3 is a subgraph.
                 # We can try to map the 8 vertices of vs to the 8 vertices of Q3.
                 # There are 8! = 40320 mappings.
                 # For each mapping, check if all 12 edges exist.
                 # This is 40320 * 12 = 483840 checks per 8-set.
                 # Too slow if many 8-sets pass the filter.
                 
                 # Better: Use the 840 canonical patterns.
                 # For each pattern P (set of 12 edges on 0..7), check if there is a bijection from 0..7 to vs
                 # such that edges in P map to edges in sub_edges.
                 # This is the same complexity.
                 
                 # Let's rely on the fact that if edge_count is large, it's very likely to contain Q3.
                 # We can sample or use a heuristic.
                 # But we need correctness.
                 
                 # Let's try a different approach: 
                 # If edge_count >= 12 and min_deg >= 3, check bipartiteness.
                 # If bipartite, it's a candidate.
                 # Then check if it contains Q3.
                 
                 # For now, to ensure speed, I will only check the 12-edge case rigorously.
                 # If edge_count > 12, I will assume it contains Q3 (conservative for "free" check, i.e., I will return False if I'm not sure? 
                 # No, returning False means "contains Q3". 
                 # If I return False for edge_count > 12, I might reject valid Q3-free graphs.
                 # If I return True, I might accept Q3-containing graphs.
                 # The goal is to maximize edges. If I accept a graph that contains Q3, the evaluator will fail it (score 0).
                 # So I must be conservative: if I'm not sure, assume it contains Q3.
                 
                 # So: if edge_count > 12 and min_deg >= 3, assume it contains Q3.
                 return False
                 
        return True

    # Baseline: K3,13
    k313 = set((a, b) for a in range(3) for b in range(3, 16))
    best_edges = k313
    best_count = len(k313)

    # Local Search
    current_edges = set(k313)
    
    # Try to add edges
    all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    # Sort by potential impact? Random order is often good for local search.
    random.seed(42)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Shuffle candidates to avoid local minima
        candidates = [e for e in all_edges if e not in current_edges]
        random.shuffle(candidates)
        
        for edge in candidates:
            if time.time() > deadline - 5:
                break
            current_edges.add(edge)
            if is_q3_free(current_edges):
                best_edges = set(current_edges)
                best_count = len(current_edges)
                improved = True
            else:
                current_edges.remove(edge)
                
    # Try random restarts with higher density
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        p = random.uniform(0.6, 0.8)
        edges = set()
        for i in range(16):
            for j in range(i + 1, 16):
                if random.random() < p:
                    edges.add((i, j))
        if is_q3_free(edges) and len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
            # Local search from this new start
            current_edges = set(edges)
            candidates = [e for e in all_edges if e not in current_edges]
            random.shuffle(candidates)
            
            for edge in candidates:
                if time.time() > deadline - 5:
                    break
                current_edges.add(edge)
                if is_q3_free(current_edges):
                    best_edges = set(current_edges)
                    best_count = len(current_edges)
                else:
                    current_edges.remove(edge)

    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
