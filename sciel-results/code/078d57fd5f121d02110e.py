"""Cochromatic partition solver for pinned random graph on 14 vertices."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for u, v in P['edges']:
        edges.add((min(u,v), max(u,v)))
    
    # Adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    def is_edge(u, v):
        if u == v: return False
        return adj[u][v]
    
    # Try k from 2 upward (k=1 means whole graph is clique or independent set)
    # For a given k, we assign each vertex to a slot 0..k-1
    # Each slot must be either all-clique or all-independent
    
    # Precompute for each pair whether they are adjacent
    # For a set of vertices assigned to the same slot, check if all pairs are edges or all non-edges
    
    # Use backtracking with constraint propagation
    # State: assignment of vertices to slots, and for each slot, whether it's forced to be clique/independent/mixed
    
    best_result = None
    deadline = time.time() + 50  # safety margin
    
    # Order vertices by degree (descending) for better pruning
    deg = [sum(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    inv_order = [0]*n
    for i, v in enumerate(order):
        inv_order[v] = i
    
    def solve(k):
        """Try to find a valid cochromatic partition with exactly k parts."""
        # assignment[v] = slot index for vertex v, -1 if unassigned
        assignment = [-1]*n
        # slot_type[s] = -1 (undetermined), 0 (clique), 1 (independent)
        slot_type = [-1]*k
        # slot_members[s] = set of vertices in slot s
        slot_members = [set() for _ in range(k)]
        
        # We process vertices in 'order' (by index in order list)
        # For symmetry breaking: first vertex goes to slot 0
        # Second vertex goes to slot 0 or 1 (if different from first)
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                # All vertices assigned, build the partition
                parts = []
                for s in range(k):
                    if slot_members[s]:
                        parts.append(sorted(slot_members[s]))
                return parts
            
            v = order[idx]
            
            # Try assigning v to each slot
            # Symmetry breaking: if this is the first vertex, only try slot 0
            # If this is the second vertex, try slot 0 and slot 1
            if idx == 0:
                slots_to_try = [0]
            elif idx == 1:
                slots_to_try = [0, 1] if k >= 2 else [0]
            else:
                slots_to_try = list(range(k))
            
            for s in slots_to_try:
                # Check if adding v to slot s is consistent
                st = slot_type[s]
                
                if st == 0:  # Must be clique
                    # v must be adjacent to all existing members
                    ok = True
                    for u in slot_members[s]:
                        if not is_edge(u, v):
                            ok = False
                            break
                    if not ok:
                        continue
                elif st == 1:  # Must be independent
                    # v must NOT be adjacent to any existing member
                    ok = True
                    for u in slot_members[s]:
                        if is_edge(u, v):
                            ok = False
                            break
                    if not ok:
                        continue
                else:  # Undetermined
                    # Check if adding v forces a type
                    # If slot has existing members, check consistency
                    if slot_members[s]:
                        # Check all pairs involving v
                        has_edge = False
                        has_nonedge = False
                        for u in slot_members[s]:
                            if is_edge(u, v):
                                has_edge = True
                            else:
                                has_nonedge = True
                        if has_edge and has_nonedge:
                            # Conflict: can't be all-clique or all-independent
                            continue
                        # Determine new type
                        if has_edge:
                            slot_type[s] = 0
                        elif has_nonedge:
                            slot_type[s] = 1
                        # If no members before, type remains undetermined
                    else:
                        # Empty slot, type remains undetermined
                        pass
                
                # Assign v to slot s
                assignment[v] = s
                slot_members[s].add(v)
                
                # Additional pruning: check that remaining unassigned vertices
                # can still be placed (basic check)
                
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                
                # Undo
                slot_members[s].discard(v)
                assignment[v] = -1
                if st == -1 and len(slot_members[s]) < 2:
                    # If we determined a type but now the slot has < 2 members, reset
                    # Actually, if slot had 1 member before and we determined type from that,
                    # we need to reset. Let's track this more carefully.
                    # Simpler: just reset if it was undetermined before
                    if not slot_members[s]:
                        slot_type[s] = -1
                    # If slot had exactly 1 member before, the type was determined by pair
                    # with that one member, so we need to reset
                    # But we don't have the old state easily. Let's use a different approach.
                    # Actually, if st was -1 and we set it, we should restore.
                    # Let me restructure to save/restore properly.
            
            return None
        
        # The above has issues with state restoration. Let me redo with proper save/restore.
        
        def backtrack2(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                parts = []
                for s in range(k):
                    if slot_members[s]:
                        parts.append(sorted(slot_members[s]))
                return parts
            
            v = order[idx]
            
            if idx == 0:
                slots_to_try = [0]
            elif idx == 1 and k >= 2:
                slots_to_try = [0, 1]
            else:
                slots_to_try = list(range(k))
            
            for s in slots_to_try:
                st_before = slot_type[s]
                
                # Check consistency
                valid = True
                new_type = st_before
                
                if slot_members[s]:
                    has_edge = False
                    has_nonedge = False
                    for u in slot_members[s]:
                        if is_edge(u, v):
                            has_edge = True
                        else:
                            has_nonedge = True
                    
                    if st_before == 0:
                        if has_nonedge:
                            valid = False
                    elif st_before == 1:
                        if has_edge:
                            valid = False
                    else:
                        if has_edge and has_nonedge:
                            valid = False
                        elif has_edge:
                            new_type = 0
                        elif has_nonedge:
                            new_type = 1
                
                if not valid:
                    continue
                
                # Apply assignment
                slot_type[s] = new_type
                slot_members[s].add(v)
                
                result = backtrack2(idx + 1)
                if result is not None:
                    return result
                
                # Undo
                slot_members[s].discard(v)
                slot_type[s] = st_before
            
            return None
        
        return backtrack2(0)
    
    # Try increasing k
    for k in range(2, n+1):
        if time.time() > deadline - 5:
            break
        result = solve(k)
        if result is not None:
            # Ensure types are list, not tuple
            return [list(part) for part in result]
    
    # Fallback: greedy (should not reach here)
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [is_edge(u, w) for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [list(p) for p in parts]

# EVOLVE-BLOCK-END
