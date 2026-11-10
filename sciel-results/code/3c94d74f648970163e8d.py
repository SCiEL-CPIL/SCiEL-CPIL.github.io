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
    
    # Precompute for each edge, the three other edges in each face it belongs to
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
                for (a, b, c) in edge_companions[e_idx]:
                    if selected[b] and selected[c]:
                        conflict_count[a] += 1
                    if selected[a] and selected[c]:
                        conflict_count[b] += 1
                    if selected[a] and selected[b]:
                        conflict_count[c] += 1
        return selected, conflict_count, count
    
    def local_search_swap(selected, conflict_count, best_count):
        """Local search: try swapping one selected edge for one unselected edge."""
        improved = False
        # Build list of selected and unselected edges
        sel_list = [i for i in range(n_edges) if selected[i]]
        unsel_list = [i for i in range(n_edges) if not selected[i]]
        
        # Randomize order for diversity
        random.shuffle(sel_list)
        random.shuffle(unsel_list)
        
        for e_remove in sel_list:
            if time.time() > deadline - 5.0:
                return selected, conflict_count, best_count, improved
            
            # Temporarily remove e_remove
            selected[e_remove] = False
            removed_conflicts = []
            for (a, b, c) in edge_companions[e_remove]:
                if selected[b] and selected[c]:
                    conflict_count[a] -= 1
                    removed_conflicts.append((a, 1))
                if selected[a] and selected[c]:
                    conflict_count[b] -= 1
                    removed_conflicts.append((b, 1))
                if selected[a] and selected[b]:
                    conflict_count[c] -= 1
                    removed_conflicts.append((c, 1))
            
            # Find an unselected edge that can now be added
            best_add = -1
            for e_add in unsel_list:
                if conflict_count[e_add] == 0:
                    best_add = e_add
                    break
            
            if best_add >= 0:
                # Accept the swap
                selected[best_add] = True
                best_count += 0  # count stays same (removed 1, added 1)
                # Update conflict counts for the new edge
                for (a, b, c) in edge_companions[best_add]:
                    if selected[b] and selected[c]:
                        conflict_count[a] += 1
                    if selected[a] and selected[c]:
                        conflict_count[b] += 1
                    if selected[a] and selected[b]:
                        conflict_count[c] += 1
                improved = True
            else:
                # Restore e_remove
                for (a, delta) in removed_conflicts:
                    conflict_count[a] += delta
                selected[e_remove] = True
            
            if improved:
                # Rebuild lists since we changed the state
                break
        
        return selected, conflict_count, best_count, improved
    
    def local_search_remove_add(selected, conflict_count, best_count):
        """Local search: remove one edge, then greedily add all possible edges."""
        improved = False
        sel_list = [i for i in range(n_edges) if selected[i]]
        random.shuffle(sel_list)
        
        for e_remove in sel_list:
            if time.time() > deadline - 5.0:
                return selected, conflict_count, best_count, improved
            
            # Save state
            old_selected = selected[e_remove]
            selected[e_remove] = False
            current_count = best_count - 1
            
            # Update conflict counts on removal
            removed_updates = []
            for (a, b, c) in edge_companions[e_remove]:
                if selected[b] and selected[c]:
                    conflict_count[a] -= 1
                    removed_updates.append((a, -1))
                if selected[a] and selected[c]:
                    conflict_count[b] -= 1
                    removed_updates.append((b, -1))
                if selected[a] and selected[b]:
                    conflict_count[c] -= 1
                    removed_updates.append((c, -1))
            
            # Greedily add all edges with conflict_count == 0
            added_edges = []
            for e_add in range(n_edges):
                if selected[e_add]:
                    continue
                if conflict_count[e_add] == 0:
                    selected[e_add] = True
                    current_count += 1
                    added_edges.append(e_add)
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
                # Restore: remove added edges, re-add e_remove
                for e_add in added_edges:
                    selected[e_add] = False
                    for (a, b, c) in edge_companions[e_add]:
                        if selected[b] and selected[c]:
                            conflict_count[a] -= 1
                        if selected[a] and selected[c]:
                            conflict_count[b] -= 1
                        if selected[a] and selected[b]:
                            conflict_count[c] -= 1
                for (a, delta) in removed_updates:
                    conflict_count[a] -= delta
                selected[e_remove] = True
        
        return selected, conflict_count, best_count, improved
    
    def local_search_2swap(selected, conflict_count, best_count):
        """Local search: try removing 2 edges and adding up to 3 edges."""
        improved = False
        sel_list = [i for i in range(n_edges) if selected[i]]
        random.shuffle(sel_list)
        
        # Try a limited number of pairs
        max_pairs = min(50, len(sel_list))
        for pair_idx in range(max_pairs):
            if time.time() > deadline - 5.0:
                return selected, conflict_count, best_count, improved
            
            e1 = sel_list[pair_idx % len(sel_list)]
            e2 = sel_list[(pair_idx + 1) % len(sel_list)]
            if e1 == e2:
                continue
            
            # Save and remove both
            selected[e1] = False
            selected[e2] = False
            current_count = best_count - 2
            
            # Track all conflict updates for restoration
            updates = []
            for e_rem in [e1, e2]:
                for (a, b, c) in edge_companions[e_rem]:
                    if selected[b] and selected[c]:
                        conflict_count[a] -= 1
                        updates.append((a, -1))
                    if selected[a] and selected[c]:
                        conflict_count[b] -= 1
                        updates.append((b, -1))
                    if selected[a] and selected[b]:
                        conflict_count[c] -= 1
                        updates.append((c, -1))
            
            # Greedily add edges
            added_edges = []
            for e_add in range(n_edges):
                if selected[e_add]:
                    continue
                if conflict_count[e_add] == 0:
                    selected[e_add] = True
                    current_count += 1
                    added_edges.append(e_add)
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
                for e_add in added_edges:
                    selected[e_add] = False
                    for (a, b, c) in edge_companions[e_add]:
                        if selected[b] and selected[c]:
                            conflict_count[a] -= 1
                        if selected[a] and selected[c]:
                            conflict_count[b] -= 1
                        if selected[a] and selected[b]:
                            conflict_count[c] -= 1
                for (a, delta) in updates:
                    conflict_count[a] -= delta
                selected[e1] = True
                selected[e2] = True
        
        return selected, conflict_count, best_count, improved
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_selected = None
    best_conflict = None
    best_count = 0
    
    # Phase 1: Many randomized greedy passes
    while time.time() < deadline - 30.0:
        selected, conflict_count, count = greedy_random_fast()
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_conflict = conflict_count[:]
    
    # Phase 2: Local search iterations
    if best_selected is not None:
        stall_counter = 0
        max_stall = 10
        
        while time.time() < deadline - 5.0 and stall_counter < max_stall:
            improved = False
            
            # Try remove-and-add
            selected, conflict_count, best_count, imp1 = local_search_remove_add(
                best_selected, best_conflict, best_count)
            if imp1:
                best_selected = selected
                best_conflict = conflict_count
                improved = True
                stall_counter = 0
            
            # Try 2-swap
            selected, conflict_count, best_count, imp2 = local_search_2swap(
                best_selected, best_conflict, best_count)
            if imp2:
                best_selected = selected
                best_conflict = conflict_count
                improved = True
                stall_counter = 0
            
            if not improved:
                stall_counter += 1
                # Try a few random restarts
                for _ in range(3):
                    if time.time() > deadline - 5.0:
                        break
                    selected, conflict_count, count = greedy_random_fast()
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        best_conflict = conflict_count[:]
                        improved = True
                        stall_counter = 0
                        break
        
        # Final local search polish
        while time.time() < deadline - 3.0:
            selected, conflict_count, best_count, imp = local_search_remove_add(
                best_selected, best_conflict, best_count)
            if not imp:
                break
            best_selected = selected
            best_conflict = conflict_count
    
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
        
        # After repair, try to add back edges
        # Recompute conflict counts
        conflict_count = [0] * n_edges
        for i in range(n_edges):
            if best_selected[i]:
                for (a, b, c) in edge_companions[i]:
                    if best_selected[b] and best_selected[c]:
                        conflict_count[a] += 1
                    if best_selected[a] and best_selected[c]:
                        conflict_count[b] += 1
                    if best_selected[a] and best_selected[b]:
                        conflict_count[c] += 1
        
        # Try to add edges
        for e_idx in range(n_edges):
            if not best_selected[e_idx] and conflict_count[e_idx] == 0:
                best_selected[e_idx] = True
                for (a, b, c) in edge_companions[e_idx]:
                    if best_selected[b] and best_selected[c]:
                        conflict_count[a] += 1
                    if best_selected[a] and best_selected[c]:
                        conflict_count[b] += 1
                    if best_selected[a] and best_selected[b]:
                        conflict_count[c] += 1
    
    result = []
    for e_idx, (u, v) in enumerate(edges):
        if best_selected is not None and best_selected[e_idx]:
            result.append([u, v])
    
    return result
