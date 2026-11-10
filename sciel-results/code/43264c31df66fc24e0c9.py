import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    
    # Generate all edges and assign each an index
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = len(edges)
                edges.append((u, v))
    
    n_edges = len(edges)
    
    # Precompute for each edge, the three other edges that would complete a C4
    # For edge (u,v) differing in bit i, and another bit j:
    #   w1 = u ^ (1<<j), w2 = v ^ (1<<j)
    #   The three companion edges are: (u,w1), (v,w2), (w1,w2)
    edge_companions = []
    for e_idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        companions = []
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            w1 = u ^ mask
            w2 = v ^ mask
            e1 = (min(u, w1), max(u, w1))
            e2 = (min(v, w2), max(v, w2))
            e3 = (min(w1, w2), max(w1, w2))
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            companions.append((idx1, idx2, idx3))
        edge_companions.append(companions)
    
    # Build conflict graph: for each edge, list of "conflict triples" it participates in
    # conflict_count[e] = number of conflict triples where all other 3 edges are selected
    # We maintain this incrementally.
    
    # For each edge, precompute which triples it's part of
    # triple_to_edges: maps a triple (a,b,c) to the set of edges that conflict with it
    # Actually, let's think differently:
    # When we select edge e, it conflicts with any triple (a,b,c) where e is one of the 3 companions
    # So for each edge e, we need to know which other edges' triples include e as a companion
    
    # Let's build: for each edge e, the list of (triple_id) where e is a companion
    # And for each triple, the 4 edges that can complete it (the 3 companions + the edge itself)
    
    # Simpler approach: maintain conflict_count[e] = number of triples where e is NOT selected
    # but all 3 companion edges ARE selected.
    # When we add edge e: for each triple (a,b,c) that e is a companion of,
    #   if selected[a] and selected[b] and selected[c], then this triple is now "complete"
    #   which means the 4th edge of this triple can no longer be added.
    #   Wait, that's not quite right either.
    
    # Let me reconsider. The constraint is: no 2D face has all 4 edges selected.
    # Each 2D face corresponds to a unique set of 4 edges.
    # For edge e, it participates in (d-1) faces (one for each other bit direction).
    # In each such face, e is one of the 4 edges.
    # e can be added iff for every face containing e, at most 2 of the other 3 edges are selected.
    # Equivalently: e can be added iff for every face containing e, NOT(all 3 other edges selected).
    
    # So: conflict_count[e] = number of faces containing e where all 3 other edges are selected.
    # e can be added iff conflict_count[e] == 0.
    
    # Precompute: for each edge, the list of (face_id, other_3_edges)
    # But we can just use the companions directly.
    
    # Let's build adjacency: for each edge e, list of faces it's in, and for each face,
    # the other 3 edges.
    
    # Actually, the companions list already gives us: for edge e, for each direction j != i,
    # the 3 other edges in that face are companions[k].
    # So conflict_count[e] = sum over companions of (1 if all 3 are selected else 0)
    
    # To update efficiently:
    # When we select edge x:
    #   For each face that contains x, the conflict_count of the 4th edge in that face increases by 1
    #   (if the other 2 edges in that face are already selected)
    #   Wait, no. Let's think again.
    #   
    #   conflict_count[e] counts faces where all 3 OTHER edges are selected.
    #   When we select edge x, for each face F containing x:
    #     Let the other 3 edges in F be a, b, c.
    #     For edge e = x: not relevant (x is now selected)
    #     For the 4th edge in F (call it y): if a, b, c are all now selected, conflict_count[y] += 1
    #     But also, for edges a, b, c: if the other 2 (plus x) are all selected... 
    #     No wait. conflict_count[a] counts faces where all 3 other edges (besides a) are selected.
    #     In face F, the other 3 edges besides a are: x, b, c.
    #     So when x is selected, if b and c are also selected, then conflict_count[a] += 1.
    #     Similarly for b and c.
    
    # So when selecting edge x:
    #   For each face F containing x, with other edges a, b, c:
    #     if selected[b] and selected[c]: conflict_count[a] += 1
    #     if selected[a] and selected[c]: conflict_count[b] += 1
    #     if selected[a] and selected[b]: conflict_count[c] += 1
    #     (a, b, c are the 3 companions of x in this face)
    #   And conflict_count[x] itself doesn't change (x is now selected, we don't add it again)
    
    # When deselecting edge x:
    #   For each face F containing x, with other edges a, b, c:
    #     if selected[b] and selected[c]: conflict_count[a] -= 1
    #     if selected[a] and selected[c]: conflict_count[b] -= 1
    #     if selected[a] and selected[b]: conflict_count[c] -= 1
    
    def greedy_random_fast():
        """One randomized greedy pass with O(1) conflict checks."""
        order = list(range(n_edges))
        random.shuffle(order)
        selected = [False] * n_edges
        conflict_count = [0] * n_edges
        count = 0
        for e_idx in order:
            if conflict_count[e_idx] == 0:
                selected[e_idx] = True
                count += 1
                # Update conflict counts for companions
                for (a, b, c) in edge_companions[e_idx]:
                    if selected[b] and selected[c]:
                        conflict_count[a] += 1
                    if selected[a] and selected[c]:
                        conflict_count[b] += 1
                    if selected[a] and selected[b]:
                        conflict_count[c] += 1
        return selected, conflict_count, count
    
    def local_search_fast(selected, conflict_count, best_count):
        """Local search: try removing one edge and adding others."""
        improved = False
        # Try removing each edge
        for e_remove in range(n_edges):
            if not selected[e_remove]:
                continue
            
            # Temporarily remove
            selected[e_remove] = False
            current_count = best_count - 1
            
            # Update conflict counts: decrement for companions
            for (a, b, c) in edge_companions[e_remove]:
                if selected[b] and selected[c]:
                    conflict_count[a] -= 1
                if selected[a] and selected[c]:
                    conflict_count[b] -= 1
                if selected[a] and selected[b]:
                    conflict_count[c] -= 1
            
            # Try to add edges
            for e_add in range(n_edges):
                if selected[e_add]:
                    continue
                if conflict_count[e_add] == 0:
                    selected[e_add] = True
                    current_count += 1
                    for (a, b, c) in edge_companions[e_add]:
                        if selected[b] and selected[c]:
                            conflict_count[a] += 1
                        if selected[a] and selected[c]:
                            conflict_count[b] += 1
                        if selected[a] and selected[b]:
                            conflict_count[c] += 1
            
            if current_count > best_count:
                best_count = current_count
                improved = True
                break
            else:
                # Restore
                # First, remove all newly added edges
                for e_add in range(n_edges):
                    if selected[e_add] and e_add != e_remove:
                        # This might have been added in this iteration
                        pass
                # Actually, let's just rebuild from scratch if no improvement
                # This is getting complex. Let's use a simpler approach:
                # Just restore e_remove and undo any additions.
                
                # Undo additions
                for e_add in range(n_edges):
                    if e_add == e_remove:
                        continue
                    if selected[e_add]:
                        # Check if this was added in this pass (it wasn't selected before)
                        # We don't track this easily. Let's just restart.
                        pass
                
                # Simpler: just set e_remove back and recompute
                selected[e_remove] = True
                # Recompute conflict_count from scratch for this state
                # This is O(n_edges * 6) which is fine for occasional use
                for i in range(n_edges):
                    conflict_count[i] = 0
                for i in range(n_edges):
                    if selected[i]:
                        for (a, b, c) in edge_companions[i]:
                            if selected[b] and selected[c]:
                                conflict_count[a] += 1
                            if selected[a] and selected[c]:
                                conflict_count[b] += 1
                            if selected[a] and selected[b]:
                                conflict_count[c] += 1
            
        return selected, conflict_count, best_count, improved
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_selected = None
    best_conflict = None
    best_count = 0
    
    # Phase 1: Many randomized greedy passes
    while time.time() < deadline - 15.0:
        selected, conflict_count, count = greedy_random_fast()
        if count > best_count:
            best_count = count
            best_selected = selected
            best_conflict = conflict_count
    
    # Phase 2: Local search
    if best_selected is not None:
        for iteration in range(100):
            if time.time() > deadline - 3.0:
                break
            
            selected, conflict_count, best_count, improved = local_search_fast(
                best_selected, best_conflict, best_count)
            
            if not improved:
                # Try a few more random restarts
                for _ in range(5):
                    if time.time() > deadline - 3.0:
                        break
                    selected, conflict_count, count = greedy_random_fast()
                    if count > best_count:
                        best_count = count
                        best_selected = selected
                        best_conflict = conflict_count
                        improved = True
                    if not improved:
                        break
    
    # Final validation and repair
    if best_selected is not None:
        for u in range(n_vertices):
            for i in range(d):
                for j in range(i + 1, d):
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(u, v1), max(u, v1))
                    e2 = (min(u, v2), max(u, v2))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v3), max(v2, v3))
                    
                    idx1 = edge_index[e1]
                    idx2 = edge_index[e2]
                    idx3 = edge_index[e3]
                    idx4 = edge_index[e4]
                    
                    if best_selected[idx1] and best_selected[idx2] and best_selected[idx3] and best_selected[idx4]:
                        best_selected[idx4] = False
        
    result = []
    for e_idx, (u, v) in enumerate(edges):
        if best_selected is not None and best_selected[e_idx]:
            result.append([u, v])
    
    return result
