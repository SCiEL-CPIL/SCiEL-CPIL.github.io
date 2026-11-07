import time
import sys
from random import Random

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    Strategy:
    1. Generate the split graph.
    2. Maximize the number of size-3 cliques {v, u, w} where v in I, u,w in K.
       This is equivalent to maximizing the number of internal edges in K assigned
       to vertices in I, subject to the constraint that edges assigned to a single
       v form a matching.
    3. Use random restarts with different orderings of I vertices to find a good packing.
    4. Cover remaining cross edges with size-2 cliques.
    5. Cover remaining internal edges with one large clique (or multiple if needed, but 
       one clique on the support of remaining edges suffices).
    """
    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency lists
        neighbors_in_I = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors_in_I[v] = nbrs
        
        # Pre-compute internal edges
        # We need to check availability quickly.
        # Map edge (u, w) to a unique ID or just use a set of tuples.
        all_internal_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_internal_edges.add((u, w))
        
        best_M = 0
        best_assignments = None
        best_order = None
        
        # Number of restarts
        # k=30, n=90. Internal edges ~ 435.
        # Greedy is fast.
        num_restarts = 100
        search_rng = Random(seed + 1000)
        
        I_vertices = list(range(k, n))
        
        for _ in range(num_restarts):
            # Shuffle I vertices
            order = I_vertices[:]
            search_rng.shuffle(order)
            
            used_edges = set()
            assignments = {} # v -> list of (u, w)
            M = 0
            
            for v in order:
                nbrs = neighbors_in_I[v]
                if not nbrs:
                    continue
                
                # We need to pick disjoint pairs from nbrs
                # such that the internal edge is not in used_edges.
                
                # To make it random, we can shuffle nbrs too?
                # But the greedy logic depends on the order.
                # Let's shuffle a copy of nbrs.
                local_nbrs = nbrs[:]
                search_rng.shuffle(local_nbrs)
                
                local_used_vertices = set()
                pairs_for_v = []
                
                i = 0
                while i < len(local_nbrs):
                    u = local_nbrs[i]
                    if u in local_used_vertices:
                        i += 1
                        continue
                    
                    found = False
                    j = i + 1
                    while j < len(local_nbrs):
                        w = local_nbrs[j]
                        if w in local_used_vertices:
                            j += 1
                            continue
                        
                        edge_key = (u, w) if u < w else (w, u)
                        if edge_key not in used_edges:
                            # Use this pair
                            pairs_for_v.append((u, w))
                            used_edges.add(edge_key)
                            local_used_vertices.add(u)
                            local_used_vertices.add(w)
                            M += 1
                            found = True
                            break
                        j += 1
                    
                    if not found:
                        # u cannot be paired with any remaining w
                        # Remove u from consideration?
                        # We just skip u. It remains in local_nbrs but 
                        # since we don't remove it, we might check it again?
                        # No, we increment i.
                        # But if u is skipped, it might be paired with a w that 
                        # appears later? No, we iterated all j > i.
                        # So u is effectively unusable for this v.
                        pass
                    i += 1
                
                if pairs_for_v:
                    assignments[v] = pairs_for_v
            
            if M > best_M:
                best_M = M
                best_assignments = assignments
                best_order = order # Not strictly needed if we have assignments
        
        # Construct cliques
        cliques = []
        
        # 1. Size-3 cliques from best_assignments
        covered_cross = set()
        for v, pairs in best_assignments.items():
            for u, w in pairs:
                cliques.append([v, u, w])
                covered_cross.add((v, u))
                covered_cross.add((v, w))
        
        # 2. Remaining cross edges
        for v in range(k, n):
            for u in neighbors_in_I[v]:
                if (v, u) not in covered_cross:
                    cliques.append([v, u])
        
        # 3. Remaining internal edges
        # Find which internal edges are NOT in used_edges from the best run.
        # We need to recover the used_edges from the best run.
        # We didn't store it. Let's re-derive it.
        used_edges_best = set()
        for v, pairs in best_assignments.items():
            for u, w in pairs:
                edge_key = (u, w) if u < w else (w, u)
                used_edges_best.add(edge_key)
        
        remaining_internal_edges = []
        for e in all_internal_edges:
            if e not in used_edges_best:
                remaining_internal_edges.append(e)
        
        if remaining_internal_edges:
            # Find all vertices involved in remaining internal edges
            # and form one clique.
            vertices_involved = set()
            for u, w in remaining_internal_edges:
                vertices_involved.add(u)
                vertices_involved.add(w)
            
            if len(vertices_involved) >= 2:
                # The clique is the list of vertices
                # Ensure they are ints
                cliques.append(list(vertices_involved))
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(instance(*spec))
    
    return result
