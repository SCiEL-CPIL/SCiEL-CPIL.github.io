"""Structural rewrite: Greedy removal (hitting set) with dynamic threat updates."""
import time
import random
import heapq

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit

    d = 7
    n = 1 << d  # 128 vertices
    total_edges = d * (1 << (d - 1))  # 448

    # Precompute all edges with unique indices
    edges = []  # list of (u, v) with u < v
    edge_index = {}  # (u,v) -> idx
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx

    # For each edge, find all faces that contain it
    # Face identified by (x, i, j) where x < x^(1<<i) < x^(1<<i|1<<j)
    # Edges in face: (x, x^i), (x, x^j), (x^i, x^i^j), (x^j, x^i^j)
    edge_to_faces = [[] for _ in range(total_edges)]
    face_edges = []  # list of 4 edge indices per face
    face_id = {}  # (x, i, j) -> face_idx

    for x in range(n):
        for i in range(d):
            xi = x ^ (1 << i)
            for j in range(i + 1, d):
                xj = x ^ (1 << j)
                xij = x ^ (1 << i) ^ (1 << j)
                # The 4 vertices of the face
                # Edges: (x,xi), (x,xj), (xi,xij), (xj,xij)
                e1 = edge_index[(min(x, xi), max(x, xi))]
                e2 = edge_index[(min(x, xj), max(x, xj))]
                e3 = edge_index[(min(xi, xij), max(xi, xij))]
                e4 = edge_index[(min(xj, xij), max(xj, xij))]
                face_idx = len(face_edges)
                face_edges.append((e1, e2, e3, e4))
                for e in (e1, e2, e3, e4):
                    edge_to_faces[e].append(face_idx)

    num_faces = len(face_edges)

    # Greedy Removal Strategy
    # 1. Start with all edges selected.
    # 2. Face count: number of selected edges in each face.
    # 3. An edge is "critical" if it belongs to a face with count == 3.
    # 4. Remove the edge that belongs to the most critical faces.
    
    selected = [True] * total_edges
    face_count = [0] * num_faces
    
    # Initialize face counts
    for e in range(total_edges):
        for f in edge_to_faces[e]:
            face_count[f] += 1
    
    # Count how many critical faces each edge belongs to
    # critical_face_count[f] = 1 if face_count[f] == 3 else 0
    # edge_threat[e] = sum of critical_face_count[f] for f in edge_to_faces[e]
    
    edge_threat = [0] * total_edges
    for e in range(total_edges):
        threat = 0
        for f in edge_to_faces[e]:
            if face_count[f] == 3:
                threat += 1
        edge_threat[e] = threat
        
    # Min-heap for edges with threat > 0? Or just scan?
    # Since we remove one edge at a time, we need to update threats.
    # Using a heap with lazy deletion is efficient.
    # Heap items: (-threat, edge_idx)
    
    heap = []
    for e in range(total_edges):
        if edge_threat[e] > 0:
            heapq.heappush(heap, (-edge_threat[e], e))
            
    removed_edges = set()
    
    while time.time() < deadline - 5:
        # Find the edge with the highest threat
        if not heap:
            break
            
        neg_threat, e = heapq.heappop(heap)
        
        # Check if this entry is still valid
        # An entry is invalid if:
        # 1. The edge has already been removed.
        # 2. The threat value has changed (we push new values when threats change).
        # To handle this, we can just check if the edge is still selected and 
        # if the threat matches the current calculated threat? 
        # Actually, simpler: just check if edge is selected. 
        # But threat might be 0 now. If threat is 0, we shouldn't remove it.
        
        if not selected[e]:
            continue
            
        # Recalculate current threat to be sure? 
        # Or trust the heap? 
        # If we update threats by pushing new heap entries, old entries become stale.
        # We need to check if the threat in the heap matches the current threat.
        
        current_threat = 0
        for f in edge_to_faces[e]:
            if face_count[f] == 3:
                current_threat += 1
                
        if current_threat == 0:
            # No critical faces, stop removing
            break
            
        if -neg_threat != current_threat:
            # Stale entry, push a fresh one and continue
            heapq.heappush(heap, (-current_threat, e))
            continue
            
        # Remove edge e
        selected[e] = False
        removed_edges.add(e)
        
        # Update face counts
        for f in edge_to_faces[e]:
            face_count[f] -= 1
            
            # Update threats for edges in this face
            # This face's status changed.
            # If face_count[f] was 3, it became 2. So this face is no longer critical.
            # If face_count[f] was 4, it became 3. So this face is now critical.
            
            old_count = face_count[f] + 1
            new_count = face_count[f]
            
            # For each edge in this face (except e, which is removed), 
            # its threat changes if the face status changed from critical to non-critical or vice versa.
            
            # Determine if face was critical before
            was_critical = (old_count == 3)
            is_critical = (new_count == 3)
            
            if was_critical == is_critical:
                # No change in criticality for this face
                # But wait, if old_count was 4, new is 3. 
                # If old_count was 3, new is 2.
                # If old_count was 2, new is 1.
                # If old_count was 1, new is 0.
                
                # The threat of an edge is the number of critical faces it belongs to.
                # So if the face was critical and is no longer, threat decreases by 1.
                # If the face was not critical and is now critical, threat increases by 1.
                
                pass # Logic below handles this generally
            
            # Let's just recalculate threat for all edges in the face?
            # That's expensive. 
            # Better: track delta.
            
            # If face was critical (count 3) and became non-critical (count 2):
            # All edges in this face (that are still selected) lose 1 threat.
            # If face was non-critical (count 4) and became critical (count 3):
            # All edges in this face (that are still selected) gain 1 threat.
            
            if was_critical and not is_critical:
                delta = -1
            elif not was_critical and is_critical:
                delta = 1
            else:
                delta = 0
                
            if delta != 0:
                for fe in face_edges[f]:
                    if fe == e:
                        continue
                    if not selected[fe]:
                        continue
                    
                    edge_threat[fe] += delta
                    if edge_threat[fe] > 0:
                        heapq.heappush(heap, (-edge_threat[fe], fe))

    # After greedy removal, try to add back edges
    # This is a local search step.
    # We can try to add edges that were removed.
    # If adding an edge doesn't create a face with count 4, add it.
    
    # Sort removed edges by some heuristic? 
    # Or just try all.
    
    # To speed up, we can check if adding e is possible.
    # It is possible if for all f in edge_to_faces[e], face_count[f] < 3.
    
    # We can iterate through removed edges and try to add them.
    # We might want to do this in multiple passes or use a queue.
    
    # Let's do a simple pass: try to add every removed edge if possible.
    # Repeat until no more can be added.
    
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for e in list(removed_edges):
            if selected[e]:
                continue
            
            can_add = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            
            if can_add:
                selected[e] = True
                removed_edges.discard(e)
                for f in edge_to_faces[e]:
                    face_count[f] += 1
                changed = True

    # Count final edges
    final_count = sum(selected)
    
    # Convert to list of [u, v]
    result = []
    for e in range(total_edges):
        if selected[e]:
            u, v = edges[e]
            result.append([u, v])

    return result
