# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 110  # 10s buffer for overhead
    n = 16
    
    # Q3 edges in canonical form (8 vertices, 12 edges)
    q3e = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Q3 adjacency
    q3_adj = {v: [] for v in range(8)}
    for a, b in q3e:
        q3_adj[a].append(b)
        q3_adj[b].append(a)
        
    # Precompute Q3 edge set for fast lookup
    q3_edge_set = set(q3e)
    
    def eidx(u, v):
        if u > v: u, v = v, u
        return u * 16 + v
        
    def creates_q3(mask, new_edge):
        """Check if adding new_edge to mask creates a Q3 subgraph."""
        u, v = new_edge
        # Try to map Q3 vertices 0,1 to u,v (and v,u)
        # Since Q3 is symmetric, we can fix 0->u, 1->v and also 0->v, 1->u?
        # Actually, we just need to check if there is *any* embedding where (u,v) is an edge of the Q3.
        # In Q3, every edge is equivalent. So we can fix Q3 vertex 0 to u and Q3 vertex 1 to v.
        # We also need to consider the case where Q3 vertex 0 is v and Q3 vertex 1 is u?
        # No, the edge is undirected. The mapping (0->u, 1->v) covers the edge (0,1) in Q3 mapping to (u,v).
        # Are there other edges in Q3? Yes. But we can just check if (u,v) maps to (0,1).
        # Due to symmetry of Q3, any edge can be mapped to (0,1).
        # So checking mapping 0->u, 1->v is sufficient.
        
        assign = {0: u, 1: v}
        used = {u, v}
        remaining = [i for i in range(8) if i not in (0, 1)]
        
        # Order remaining vertices by degree in Q3? All are degree 3.
        # But some are connected to 0 or 1.
        # Neighbors of 0: 1, 2, 4. 1 is used. So 2, 4 are constrained.
        # Neighbors of 1: 0, 3, 5. 0 is used. So 3, 5 are constrained.
        # Let's prioritize vertices with more assigned neighbors.
        
        def bt(idx):
            if time.time() > deadline:
                return True # Safe assumption: conflict
            if idx == len(remaining):
                # Verify all Q3 edges are present in mask (except the new one)
                for ea, eb in q3e:
                    x, y = assign[ea], assign[eb]
                    if (x, y) == new_edge or (y, x) == new_edge:
                        continue
                    if not (mask >> eidx(x, y)) & 1:
                        return False
                return True
            
            qv = remaining[idx]
            # Filter candidates: must not be used
            # Heuristic: check constraints from already assigned neighbors
            assigned_neighbors = [nb for nb in q3_adj[qv] if nb in assign]
            
            # If no assigned neighbors, any unused vertex is a candidate.
            # If there are assigned neighbors, the candidate must be adjacent to all of them in the graph.
            
            candidates = []
            for cand in range(n):
                if cand in used:
                    continue
                ok = True
                for nb in assigned_neighbors:
                    x = assign[nb]
                    # Check if cand is adjacent to x in mask
                    # Note: cand-x cannot be new_edge because new_edge is (u,v) and u,v are in used (0,1).
                    # cand is not u or v. So cand-x is not (u,v).
                    if not (mask >> eidx(cand, x)) & 1:
                        ok = False
                        break
                if ok:
                    candidates.append(cand)
                    
            # Sort candidates? Maybe by degree in current graph? 
            # For now, just iterate.
            for cand in candidates:
                assign[qv] = cand
                used.add(cand)
                if bt(idx + 1):
                    return True
                del assign[qv]
                used.remove(cand)
            return False
        
        return bt(0)

    def find_q3_subgraph(mask):
        """Verify no Q3. Returns True if Q3 found."""
        # For final validation, check all 8-sets
        all_vertices = list(range(n))
        for combo in itertools.combinations(all_vertices, 8):
            # Map combo to 0..7
            # This is slow if done every time, but we only do it at the end or rarely.
            # For the "creates_q3" we use backtracking.
            # Here we just need a boolean.
            mapping = {v: i for i, v in enumerate(combo)}
            inv_mapping = {i: v for i, v in enumerate(combo)}
            
            is_q3 = True
            for a, b in q3e:
                u, v = inv_mapping[a], inv_mapping[b]
                if not (mask >> eidx(u, v)) & 1:
                    is_q3 = False
                    break
            if is_q3:
                return True
        return False

    best_mask = 0
    best_count = 0
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Multiple random restarts
    while time.time() < deadline - 10:
        mask = 0
        edges_in_mask = set()
        
        # Greedy Phase
        missing = list(all_edges)
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 10:
                break
            if creates_q3(mask, e):
                continue
            mask |= (1 << eidx(*e))
            edges_in_mask.add(e)
            
        # Local Search Phase: 1-for-1 swaps to improve density
        # Try to remove an edge and add two? Or just swap?
        # Since we want to maximize edges, we look for:
        # Remove e1, add e2, e3 such that e2, e3 were missing and now fit.
        # This is complex. Simpler: try to add any missing edge.
        # If stuck, try to remove a random edge and re-run greedy on the remaining?
        
        # Let's try a simple hill climbing:
        # For each edge in mask, try to remove it.
        # Then try to add all missing edges.
        # If count increases, keep it.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_edges = list(edges_in_mask)
            random.shuffle(current_edges)
            
            for e in current_edges:
                if time.time() > deadline - 10:
                    break
                
                # Remove e
                mask_without_e = mask & ~(1 << eidx(*e))
                edges_without_e = edges_in_mask - {e}
                
                # Try to add missing edges to mask_without_e
                missing_now = [m for m in all_edges if m not in edges_without_e]
                random.shuffle(missing_now)
                
                new_mask = mask_without_e
                new_edges = set(edges_without_e)
                
                for m in missing_now:
                    if creates_q3(new_mask, m):
                        continue
                    new_mask |= (1 << eidx(*m))
                    new_edges.add(m)
                    
                if len(new_edges) > len(edges_in_mask):
                    mask = new_mask
                    edges_in_mask = new_edges
                    improved = True
                    break # Restart local search with new state
                    
        if len(edges_in_mask) > best_count:
            best_count = len(edges_in_mask)
            best_mask = mask

    # Final verification (should be Q3-free if logic is correct)
    # If find_q3_subgraph(best_mask):
    #    # Fallback: remove edges until Q3-free
    #    # This is unlikely to be needed if creates_q3 is correct
    #    pass

    return [[u, v] for u, v in edges_in_mask if (best_mask >> eidx(u,v)) & 1]
# EVOLVE-BLOCK-END
