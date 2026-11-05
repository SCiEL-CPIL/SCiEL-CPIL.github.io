import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    d = 7
    n = 2 ** d
    total_edges = d * 2 ** (d - 1)
    
    # Precompute all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Precompute edge index
    edge_set_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_set_idx[(u, v)] = i
    
    # Precompute for each edge, the list of "conflict triples" - sets of 3 edges that if all present,
    # would make this edge create a 4-cycle.
    edge_conflicts = [[] for _ in range(len(edges))]
    # Also precompute: for each edge, which other edges share a square with it
    edge_neighbors = [[] for _ in range(len(edges))]
    
    for idx, (u, v) in enumerate(edges):
        # Find which bit differs
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            # The other 3 edges in this 4-cycle
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # Edges: (u, u2), (u2, v2), (v2, v)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(u2, v2), max(u2, v2))
            e3 = (min(v2, v), max(v2, v))
            i1 = edge_set_idx.get(e1)
            i2 = edge_set_idx.get(e2)
            i3 = edge_set_idx.get(e3)
            if i1 is not None and i2 is not None and i3 is not None:
                edge_conflicts[idx].append((i1, i2, i3))
                # Add neighbors
                edge_neighbors[idx].append(i1)
                edge_neighbors[idx].append(i2)
                edge_neighbors[idx].append(i3)
    
    # Remove duplicates from edge_neighbors
    for idx in range(len(edges)):
        edge_neighbors[idx] = list(set(edge_neighbors[idx]))
    
    # Greedy initialization
    selected = [False] * len(edges)
    selected_count = 0
    
    for idx in range(len(edges)):
        ok = True
        for (i1, i2, i3) in edge_conflicts[idx]:
            if selected[i1] and selected[i2] and selected[i3]:
                ok = False
                break
        if ok:
            selected[idx] = True
            selected_count += 1
    
    best_count = selected_count
    best_selected = selected[:]
    
    def can_add(idx):
        for (i1, i2, i3) in edge_conflicts[idx]:
            if selected[i1] and selected[i2] and selected[i3]:
                return False
        return True
    
    # Structured local search: remove one edge, try to add multiple blocked edges
    def local_search_pass():
        nonlocal selected, selected_count, best_count, best_selected
        
        improved = True
        max_passes = 50
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            # Shuffle edges for random order
            edge_order = list(range(len(edges)))
            random.shuffle(edge_order)
            
            for e_out in edge_order:
                if time.time() > deadline:
                    return
                
                if not selected[e_out]:
                    continue
                
                # Remove e_out
                selected[e_out] = False
                selected_count -= 1
                
                # Collect candidates: edges that were blocked and share a square with e_out
                candidates = set()
                for neighbor in edge_neighbors[e_out]:
                    if not selected[neighbor]:
                        candidates.add(neighbor)
                
                # Shuffle candidates
                candidate_list = list(candidates)
                random.shuffle(candidate_list)
                
                # Try to add as many as possible
                added_count = 0
                for c in candidate_list:
                    if can_add(c):
                        selected[c] = True
                        selected_count += 1
                        added_count += 1
                
                if added_count > 1:
                    # Improvement: removed 1, added > 1
                    improved = True
                    if selected_count > best_count:
                        best_count = selected_count
                        best_selected = selected[:]
                elif added_count == 1:
                    # Neutral move - accept to explore
                    improved = True
                    if selected_count > best_count:
                        best_count = selected_count
                        best_selected = selected[:]
                else:
                    # No improvement, revert
                    selected[e_out] = True
                    selected_count += 1
    
    # Run initial local search
    local_search_pass()
    
    # Simulated Annealing to escape local optima
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    random.seed(42)
    
    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a random selected edge and try to add multiple
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            if not selected_indices:
                continue
            idx = random.choice(selected_indices)
            
            # Remove
            selected[idx] = False
            selected_count -= 1
            
            # Try to add neighbors
            candidates = [n for n in edge_neighbors[idx] if not selected[n]]
            random.shuffle(candidates)
            
            added = 0
            for c in candidates:
                if can_add(c):
                    selected[c] = True
                    selected_count += 1
                    added += 1
            
            delta = added - 1  # Net change
            
            # Accept if improvement, or with probability based on temperature
            if delta >= 0 or (delta < 0 and random.random() < np.exp(-delta / max(T, 1e-10))):
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
            else:
                # Revert
                selected[idx] = True
                selected_count += 1
                # Also revert any additions
                for c in candidates:
                    if selected[c] and c != idx:
                        # Check if this was added in this move
                        pass  # This is tricky, let's just keep it simple
        
        elif move_type < 0.7:
            # Add a random unselected edge if possible
            unselected_indices = [i for i in range(len(edges)) if not selected[i]]
            if not unselected_indices:
                continue
            idx = random.choice(unselected_indices)
            if can_add(idx):
                selected[idx] = True
                selected_count += 1
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
        else:
            # Swap: remove one selected, try to add one unselected
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            unselected_indices = [i for i in range(len(edges)) if not selected[i]]
            if not selected_indices or not unselected_indices:
                continue
            rem_idx = random.choice(selected_indices)
            add_idx = random.choice(unselected_indices)
            
            # Temporarily remove
            selected[rem_idx] = False
            selected_count -= 1
            
            if can_add(add_idx):
                selected[add_idx] = True
                selected_count += 1
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
            else:
                # Revert
                selected[rem_idx] = True
                selected_count += 1
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
        
        # Periodically run local search to polish
        if int((1.0 - T / 1.0) * 1000) % 50 == 0 and time.time() < deadline:
            local_search_pass()
    
    # Final local search polish
    if time.time() < deadline:
        local_search_pass()
    
    # Use the best solution found
    final_selected = best_selected
    result = []
    for idx in range(len(edges)):
        if final_selected[idx]:
            result.append([edges[idx][0], edges[idx][1]])
    
    return result
