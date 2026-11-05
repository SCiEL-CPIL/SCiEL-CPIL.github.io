import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d
    
    # Generate all edges
    # Edge representation: (u, v, i) where u < v and v = u ^ (1 << i)
    # For fast set lookup, we use an integer ID: u * n + v
    all_edges = []
    edge_id_map = {} # (u,v) -> id
    id_to_edge = {} # id -> (u,v,i)
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                eid = x * n + y
                all_edges.append((x, y, i, eid))
                edge_id_map[(x, y)] = eid
                id_to_edge[eid] = (x, y, i)

    # Precompute conflict neighbors for each edge
    # For edge e with dim i, for each j != i:
    # The face is {u, v, u^2^j, v^2^j}
    # e is (u,v) dim i
    # Other edges:
    # (u^2^j, v^2^j) dim i
    # (u, u^2^j) dim j
    # (v, v^2^j) dim j
    
    # conflicts[eid] = list of lists of 3 eids
    conflicts = {eid: [] for _, _, _, eid in all_edges}
    
    for u, v, i, eid in all_edges:
        for j in range(d):
            if j == i:
                continue
            uj = u ^ (1 << j)
            vj = v ^ (1 << j)
            
            # Get IDs of the other 3 edges
            # Edge 1: (min(uj, vj), max(uj, vj)) dim i
            u1, v1 = (uj, vj) if uj < vj else (vj, uj)
            e1_id = u1 * n + v1
            
            # Edge 2: (min(u, uj), max(u, uj)) dim j
            u2, v2 = (u, uj) if u < uj else (uj, u)
            e2_id = u2 * n + v2
            
            # Edge 3: (min(v, vj), max(v, vj)) dim j
            u3, v3 = (v, vj) if v < vj else (vj, v)
            e3_id = u3 * n + v3
            
            conflicts[eid].append([e1_id, e2_id, e3_id])

    def can_add(eid, chosen):
        for trio in conflicts[eid]:
            if trio[0] in chosen and trio[1] in chosen and trio[2] in chosen:
                return False
        return True

    def add_all_possible(chosen, all_edges_list):
        """
        Iteratively add any valid edge until no more can be added.
        Returns number of edges added in this call.
        """
        added_count = 0
        changed = True
        # We need to be careful with iteration order or just repeat until stable.
        # Since adding an edge might block others, but not unblock, 
        # we can just iterate through all edges and add if valid.
        # If we add one, we might need to re-check others?
        # Actually, if we process in a fixed order, adding an edge only makes
        # future additions harder, not easier. 
        # So a single pass through all edges adding valid ones is sufficient 
        # to reach a "maximal" set for that order? 
        # No, because an edge skipped earlier might become valid? 
        # No, adding edges only removes options. 
        # So if an edge is invalid at step k, it will remain invalid at step k+1.
        # Thus, a single pass is sufficient to find a maximal subset 
        # consistent with the current set + the edges added in this pass.
        
        for u, v, i, eid in all_edges_list:
            if eid not in chosen:
                if can_add(eid, chosen):
                    chosen.add(eid)
                    added_count += 1
        return added_count

    best_count = 0
    best_chosen = set()
    
    # Randomized Greedy Restarts
    # We run multiple restarts to find a good starting point for local search.
    # The number of restarts depends on time.
    
    # Estimate time per greedy pass
    # 448 edges * 6 checks * 3 lookups = ~8000 ops. Very fast.
    # We can do many restarts.
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 30: # Leave 30s for local search
            break
            
        random.shuffle(all_edges)
        chosen = set()
        
        for u, v, i, eid in all_edges:
            if can_add(eid, chosen):
                chosen.add(eid)
                
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_chosen = set(chosen)
            
        if best_count >= 308:
            break

    # Local Search on the best result
    # Strategy: 
    # 1. Try to add any missing edge (should be none if greedy was maximal, but 
    #    after removals in local search, there might be).
    # 2. If stuck, remove a random edge, then try to add as many as possible.
    #    If net gain > 0, keep. Else revert.
    
    current_chosen = set(best_chosen)
    current_count = best_count
    
    no_improve_steps = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit - 5:
        # Phase 1: Add any possible edges
        added = add_all_possible(current_chosen, all_edges)
        if added > 0:
            current_count += added
            no_improve_steps = 0
            if current_count > best_count:
                best_count = current_count
                best_chosen = set(current_chosen)
                if best_count >= 308:
                    break
            continue # Restart loop with new state
        
        # Phase 2: Stuck. Try a swap.
        # Pick a random edge to remove
        if not current_chosen:
            break
            
        # Convert to list for random selection
        # To avoid creating a new list every time, we can pick an index?
        # Sets are unordered. We can maintain a list of chosen edges?
        # For simplicity, convert to list. 300 items is small.
        chosen_list = list(current_chosen)
        rem_eid = random.choice(chosen_list)
        
        # Remove it
        current_chosen.discard(rem_eid)
        temp_count = current_count - 1
        
        # Try to add back as many as possible
        added_back = add_all_possible(current_chosen, all_edges)
        temp_count += added_back
        
        if temp_count > current_count:
            # Improvement
            current_count = temp_count
            no_improve_steps = 0
            if current_count > best_count:
                best_count = current_count
                best_chosen = set(current_chosen)
                if best_count >= 308:
                    break
        else:
            # Revert
            current_chosen.add(rem_eid)
            current_count += 1
            no_improve_steps += 1
            
            if no_improve_steps > max_no_improve:
                # Try a different strategy or break?
                # Maybe shuffle the order of edges in add_all_possible?
                # Or just break if stuck for too long.
                # Let's try a few more random removals before giving up on this local optimum.
                # If we are stuck, we might need to remove 2 edges to add 3?
                # That's more complex. Let's stick to 1-remove for now.
                # If no improvement in 200 steps, we are likely in a deep local optimum.
                # We could restart the local search from a different greedy result?
                # For now, break to save time.
                break

    # Construct final output
    final_edges = []
    for eid in best_chosen:
        u, v, i = id_to_edge[eid]
        final_edges.append([u, v])
        
    return final_edges
