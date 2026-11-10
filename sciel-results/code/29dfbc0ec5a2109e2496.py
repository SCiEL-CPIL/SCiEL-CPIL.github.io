import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precomputation
    # 1. Edge indices
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 120
    
    # 2. Subsets of size 8
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)
    
    # 3. Local edge indices for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # 4. Generate Q3 masks for K8
    # A Q3 is a subgraph of K8. We represent a specific set of 12 edges that form a Q3.
    # We iterate over all ways to partition 8 vertices into two sets of 4 (A and B)
    # and all bijections between them? No, that's for isomorphism.
    # We need to find all edge sets E' subset of E(K8) such that (V8, E') is isomorphic to Q3.
    # Q3 has 12 edges.
    # There are C(28, 12) possible edge sets, which is huge.
    # Better: Iterate over all graph isomorphisms of Q3 to K8.
    # Number of automorphisms of Q3 is 48.
    # Number of embeddings of Q3 in K8 is 8! / 48 = 40320 / 48 = 840.
    # So there are 840 distinct Q3 subgraphs in K8.
    
    # Let's generate them.
    # Standard Q3 vertices: 0..7. Edges: (i, j) if i^j is power of 2.
    # Let's pick one canonical Q3 edge set.
    q3_canonical_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d-1) == 0:
                q3_canonical_edges.append((i, j))
    
    # Generate all permutations of 8 vertices that map the canonical Q3 to a subgraph of K8
    # Since K8 is complete, any permutation gives a valid Q3 subgraph.
    # We just need the set of edges.
    # Two permutations that differ by an automorphism of Q3 will produce the same edge set.
    
    q3_masks_for_k8 = []
    seen_masks = set()
    
    # To speed up, we can fix the image of vertex 0 to be 0, 1, or 2?
    # No, just iterate all permutations? 8! = 40320. That's small.
    for perm in itertools.permutations(range(8)):
        mask = 0
        for (u, v) in q3_canonical_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        if mask not in seen_masks:
            seen_masks.add(mask)
            q3_masks_for_k8.append(mask)
            
    # 5. Map global edges to subsets and local indices
    # edge_to_subsets[e] = list of subset indices
    # local_idx[sub_idx][e] = local bit position
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Helper to check if a 28-bit mask contains a Q3
    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    best_G = 0
    best_count = 0
    
    # Strategy:
    # 1. Greedy construction with random restarts (current)
    # 2. Local Search: Swap edges to increase count? Or try to add more edges by removing fewer?
    #    Actually, if we have a maximal Q3-free graph, we can't just add edges.
    #    We can try "swaps": remove an edge, then try to add multiple new edges.
    #    Or, simpler: Try to find a better greedy ordering.
    
    # Let's implement a simple local search:
    # Given a graph G, try removing an edge e from G.
    # Then try to add any edge f not in G that doesn't create a Q3.
    # If we can add more than 1 edge (i.e., net gain > 0), do it.
    # This is a (1, k) exchange.
    
    def local_search(G):
        """
        G: int bitmask of edges
        Returns improved G
        """
        improved = True
        current_G = G
        current_count = current_G.bit_count()
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each edge in current_G
            # To speed up, only try a subset or randomize order
            edges_in_G = [i for i in range(total_edges) if (current_G >> i) & 1]
            random.shuffle(edges_in_G)
            
            for e_remove in edges_in_G:
                if time.time() > deadline - 5:
                    return current_G
                    
                # Remove e_remove
                trial_G = current_G & ~(1 << e_remove)
                
                # Try to add edges not in trial_G
                # We want to add as many as possible, or at least 2 to make it worthwhile
                # Let's try all missing edges in random order? Or just check which ones can be added.
                
                # For efficiency, let's just try to add one edge that allows adding another?
                # Simpler: Try to add any edge. If we add 1, net change is 0. Not useful.
                # We need net gain >= 1.
                # So we need to add at least 2 edges after removing 1.
                
                # Let's try to add edges greedily to trial_G
                # But we only care if we can add >= 2.
                
                # Optimization: Check if removing e_remove "unblocks" at least 2 edges.
                # An edge f is unblocked if it was blocked only by e_remove?
                # This is complex.
                
                # Let's just do a quick greedy fill on trial_G
                # But limit the time per trial.
                
                # Actually, a simpler local search:
                # Try to swap e_remove with e_add.
                # If we can add e_add, we are at same count.
                # But maybe e_add allows adding more later?
                # This is hill climbing with plateaus.
                
                # Let's stick to the greedy restarts but make them smarter.
                # Or, use the "remove 1, add k" with k=2 explicitly.
                
                # Let's try to find 2 edges that can be added to trial_G.
                # We iterate through missing edges.
                added_count = 0
                trial_mask = trial_G
                # We need to update subset masks for trial_G?
                # That's expensive.
                
                # Alternative: Just try to add ONE edge e_add.
                # If we can, we have the same count.
                # But if we then try to add ANOTHER edge e_add2, we might gain.
                # This is effectively a 2-step lookahead.
                
                # Let's just try to add a single edge e_add.
                # If successful, we have same count.
                # We don't improve.
                
                # So simple 1-for-1 swap doesn't improve count.
                # We need 1-for-2 or 2-for-3 etc.
                
                # Given time constraints, let's rely on multiple random restarts with different heuristics.
                # Heuristic 1: Random order
                # Heuristic 2: Order by degree potential (edges that conflict with fewer existing edges?)
                # Heuristic 3: Order by "density" of surrounding subgraphs?
                
                # Let's try a different greedy:
                # Start with empty graph.
                # At each step, pick the edge that minimizes the number of Q3s it "participates in" with existing edges?
                # Or maximizes the number of edges it can still be added with?
                
                # For now, let's just do many random restarts.
                # And maybe one restart with a specific heuristic:
                # Sort edges by the number of subsets they belong to (all are same? No, depends on vertices).
                # Actually, all edges are in C(14, 6) subsets? No.
                # An edge (u, v) is in a subset of 8 if u, v are in it.
                # Number of subsets containing u, v is C(14, 6) = 3003.
                # So all edges are symmetric in terms of subset count.
                
                # Let's just do random restarts.
                pass
        
        return current_G

    # Multiple restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        G = 0
        subset_masks = [0] * num_subsets
        
        edges = list(range(total_edges))
        random.shuffle(edges)
        
        for e_idx in edges:
            if time.time() > deadline - 5:
                break
                
            # Check if adding e_idx creates a Q3
            can_add = True
            for s_idx in edge_to_subsets[e_idx]:
                l_pos = local_idx[s_idx][e_idx]
                new_mask = subset_masks[s_idx] | (1 << l_pos)
                if contains_q3(new_mask):
                    can_add = False
                    break
            
            if can_add:
                G |= (1 << e_idx)
                # Update subset masks
                for s_idx in edge_to_subsets[e_idx]:
                    l_pos = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << l_pos)
        
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
            
            # Optional: Local search on best_G
            # Given the cost of local search with subset masks, we skip it for now
            # unless we have a lot of time left.
            # If we have > 30s left, we can try a simple local search.
            if time.time() < deadline - 30:
                # Simple local search: try to remove an edge and add two others?
                # Too complex to implement safely with masks.
                # Let's just continue restarts.
                pass

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
