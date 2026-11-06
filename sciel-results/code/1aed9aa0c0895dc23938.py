import random
import time

def construct():
    d = 7
    n = 2 ** d
    
    # 1. Precompute edges
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_index[e] = len(edges)
                edges.append(e)
                
    E = len(edges) # 448
    
    # 2. Precompute blocking triples for each edge
    # For each edge e, find all 4-cycles (squares) containing e.
    # A square is defined by two vertices u, v that are distance 2 apart, and the two middle vertices.
    # Or simpler: For edge (u, v), the other dimension j defines the square.
    # Square vertices: u, v, u^j, v^j.
    # Edges in square: (u,v), (u, u^j), (v, v^j), (u^j, v^j).
    # If we pick (u,v), the other 3 are the blocking triple.
    
    blocking_triples = [[] for _ in range(E)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u_j = u ^ bit_j
            v_j = v ^ bit_j
            
            # The other 3 edges
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            # Get their indices
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            
            blocking_triples[idx].append((idx1, idx2, idx3))
            
    # 3. Helper to check if an edge can be added
    def can_add(edge_idx, chosen_set):
        for triple in blocking_triples[edge_idx]:
            if triple[0] in chosen_set and triple[1] in chosen_set and triple[2] in chosen_set:
                return False
        return True

    best_count = 0
    best_set = set()
    
    deadline = time.time() + 115.0 # safety margin before 120s kill
    
    # Local Search Improvement
    def local_search(chosen_set):
        count = len(chosen_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Strategy:
            # 1. Try to add any valid edge (free lunch)
            # 2. If no free edge, try 1-swap moves (remove 1, add 1) to reach a state where free edges exist
            # 3. Greedily fill with all possible free edges
            
            # Phase 1: Add all possible valid edges
            changed = True
            while changed and time.time() < deadline:
                changed = False
                non_chosen = [i for i in range(E) if i not in chosen_set]
                random.shuffle(non_chosen)
                for e_idx in non_chosen:
                    if time.time() >= deadline:
                        return len(chosen_set)
                    if can_add(e_idx, chosen_set):
                        chosen_set.add(e_idx)
                        changed = True
                        count += 1
            if changed:
                improved = True
                continue
                
            # Phase 2: If stuck, try swaps
            # We look for a swap (remove r, add a) such that after the swap, we can add at least 1 more edge.
            # To make this efficient, we just try random swaps and see if they unlock additions.
            
            non_chosen = [i for i in range(E) if i not in chosen_set]
            if not non_chosen:
                break
                
            # Sample a few candidate additions
            random.shuffle(non_chosen)
            candidates = non_chosen[:20]
            
            swap_found = False
            for a_idx in candidates:
                if time.time() >= deadline:
                    return len(chosen_set)
                
                # Check if a_idx is blocked
                if can_add(a_idx, chosen_set):
                    # Should have been caught in Phase 1, but just in case
                    chosen_set.add(a_idx)
                    count += 1
                    swap_found = True
                    improved = True
                    break
                
                # Find blocking triples for a_idx
                # A triple (t1, t2, t3) blocks a_idx if all are in chosen_set.
                # We need to remove one edge from EVERY blocking triple.
                # If there is exactly one blocking triple, we can remove any of its 3 edges.
                # If there are multiple, we need to find a common edge or remove multiple (not allowed in 1-swap).
                
                blocking_triples_present = []
                for triple in blocking_triples[a_idx]:
                    if triple[0] in chosen_set and triple[1] in chosen_set and triple[2] in chosen_set:
                        blocking_triples_present.append(triple)
                        
                if not blocking_triples_present:
                    # Should not happen
                    continue
                    
                if len(blocking_triples_present) == 1:
                    t = blocking_triples_present[0]
                    # Try removing each of the 3 edges in t
                    # We need to ensure that removing r and adding a_idx is valid (i.e., a_idx is no longer blocked)
                    # Since t was the only blocking triple, removing any edge from t unblocks a_idx.
                    # However, we must also ensure that removing r doesn't invalidate other edges? No, removal is always valid.
                    # The only constraint is that a_idx becomes addable.
                    
                    # Try each removal option
                    for r_idx in t:
                        # Tentative swap
                        chosen_set.remove(r_idx)
                        chosen_set.add(a_idx)
                        
                        # Check if this opens up more edges
                        # We do a quick greedy refill
                        added_more = 0
                        # Sample some non-chosen edges to see if any are now addable
                        # To be fast, we just check a random subset
                        refill_candidates = [i for i in range(E) if i not in chosen_set]
                        random.shuffle(refill_candidates)
                        
                        # Check first 50
                        for c_idx in refill_candidates[:50]:
                            if time.time() >= deadline:
                                # Revert and return
                                chosen_set.remove(a_idx)
                                chosen_set.add(r_idx)
                                return len(chosen_set)
                            if can_add(c_idx, chosen_set):
                                chosen_set.add(c_idx)
                                added_more += 1
                                if added_more > 1:
                                    break # Good enough, keep this swap
                        
                        if added_more > 0:
                            # Keep the swap and additions
                            count += 1 + added_more
                            improved = True
                            swap_found = True
                            break
                        else:
                            # Revert
                            chosen_set.remove(a_idx)
                            chosen_set.add(r_idx)
                        
                    if swap_found:
                        break
                        
                elif len(blocking_triples_present) > 1:
                    # More complex, skip for now to keep it fast
                    pass
                    
            if not swap_found:
                # No simple swap improved things
                break
                
        return count

    # Main Loop
    while time.time() < deadline:
        # Randomized Greedy
        chosen = set()
        edge_order = list(range(E))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if can_add(e_idx, chosen):
                chosen.add(e_idx)
                
        count = len(chosen)
        
        # Local Search
        if time.time() < deadline - 5: # Reserve time
            count = local_search(chosen)
            
        if count > best_count:
            best_count = count
            best_set = set(chosen)
            
    # Convert back to list of edges
    return [list(edges[i]) for i in best_set]
