import random
import time
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate all edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    face_masks = [] # Bitmask of edges in each face
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                mask_val = (1 << e1) | (1 << e2) | (1 << e3) | (1 << e4)
                face_masks.append(mask_val)
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    n_faces = len(face_edges)
    
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    def can_add(e, state):
        """Check if edge e can be added to state without creating a C4."""
        for f in edge_faces[e]:
            # Get mask of other 3 edges in face f
            others = face_masks[f] & ~(1 << e)
            if (state & others) == others:
                return False
        return True

    def greedy_construct():
        """Greedy construction using bitmasks."""
        state = 0
        # Randomize edge order for diversity
        edge_order = list(range(n_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if can_add(e, state):
                state |= (1 << e)
        
        return state

    def local_search(state, time_limit):
        """Local search with bitmasks."""
        best_state = state
        best_count = bin(state).count('1')
        
        while time.time() < time_limit:
            improved = False
            
            # Try single adds
            unselected = [e for e in range(n_edges) if not (state & (1 << e))]
            random.shuffle(unselected)
            for e in unselected:
                if can_add(e, state):
                    state |= (1 << e)
                    improved = True
                    break
            
            if improved:
                count = bin(state).count('1')
                if count > best_count:
                    best_count = count
                    best_state = state
                continue
            
            # Try 1-for-2 swap (remove 1, add 2)
            selected = [e for e in range(n_edges) if (state & (1 << e))]
            random.shuffle(selected)
            
            for e_remove in selected:
                # Temporarily remove
                state_temp = state & ~(1 << e_remove)
                
                # Try to add 2 edges
                added = []
                unselected_now = [e for e in range(n_edges) if not (state_temp & (1 << e))]
                random.shuffle(unselected_now)
                
                for e_add in unselected_now:
                    if len(added) >= 2:
                        break
                    if can_add(e_add, state_temp):
                        state_temp |= (1 << e_add)
                        added.append(e_add)
                
                if len(added) == 2:
                    state = state_temp
                    improved = True
                    count = bin(state).count('1')
                    if count > best_count:
                        best_count = count
                        best_state = state
                    break
            
            if improved:
                continue
                
            # Try 2-for-3 swap (remove 2, add 3)
            if time.time() > time_limit - 1:
                break
                
            selected = [e for e in range(n_edges) if (state & (1 << e))]
            random.shuffle(selected)
            
            # Limit pairs to check
            for i in range(min(20, len(selected))):
                if time.time() > time_limit:
                    break
                e1 = selected[i]
                for j in range(i + 1, min(20, len(selected))):
                    e2 = selected[j]
                    
                    state_temp = state & ~(1 << e1) & ~(1 << e2)
                    
                    added = []
                    unselected_now = [e for e in range(n_edges) if not (state_temp & (1 << e))]
                    random.shuffle(unselected_now)
                    
                    for e_add in unselected_now:
                        if len(added) >= 3:
                            break
                        if can_add(e_add, state_temp):
                            state_temp |= (1 << e_add)
                            added.append(e_add)
                    
                    if len(added) == 3:
                        state = state_temp
                        improved = True
                        count = bin(state).count('1')
                        if count > best_count:
                            best_count = count
                            best_state = state
                        break
                if improved:
                    break
            
        return best_state, best_count

    # Phase 1: Multiple greedy restarts to find a good starting point
    best_state = 0
    best_count = 0
    
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        state = greedy_construct()
        count = bin(state).count('1')
        if count > best_count:
            best_count = count
            best_state = state
            
    # Phase 2: Local search on the best state
    best_state, best_count = local_search(best_state, deadline - 5)
    
    # Phase 3: More restarts if time permits
    restarts = 0
    while time.time() < deadline - 5:
        state = greedy_construct()
        count = bin(state).count('1')
        if count > best_count - 2: # Only refine if close to best
            state, count = local_search(state, min(deadline - 1, time.time() + 2))
            if count > best_count:
                best_count = count
                best_state = state
        restarts += 1
        if restarts > 100:
            break

    # Build result
    result = []
    for i in range(n_edges):
        if best_state & (1 << i):
            result.append([edges[i][0], edges[i][1]])
            
    return result
