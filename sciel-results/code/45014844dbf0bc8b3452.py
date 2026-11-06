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
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    n_faces = len(face_edges)
    
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety

    # Phase 1: Greedy - start with all edges, remove edges to break all faces
    # We want to minimize the number of removed edges (maximize kept edges)
    
    def greedy_min_removal():
        """Greedy: repeatedly remove the edge that's in the most 'at-risk' faces."""
        # A face is "at risk" if it has 4 edges selected (all present)
        # Initially all faces are at risk.
        # Remove an edge, which "fixes" all faces containing it.
        # After removal, a face is still at risk only if it still has all 4 edges.
        
        # Track which edges are present (not removed)
        present = [True] * n_edges
        # For each face, track how many of its 4 edges are present
        face_count = [4] * n_faces  # all start with 4
        
        removed = set()
        
        # Priority: remove edge that fixes the most faces
        # An edge "fixes" a face if that face currently has count 4 (all edges present)
        # and removing this edge drops it to 3.
        
        for _ in range(n_edges):
            if time.time() > deadline - 5:
                break
            
            # Find edge that maximizes number of faces with count==4 that contain it
            best_edge = -1
            best_score = 0
            for e in range(n_edges):
                if not present[e]:
                    continue
                score = 0
                for f in edge_faces[e]:
                    if face_count[f] == 4:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge == -1 or best_score == 0:
                break
            
            # Remove best_edge
            present[best_edge] = False
            removed.add(best_edge)
            for f in edge_faces[best_edge]:
                face_count[f] -= 1
        
        return removed, face_count, present
    
    def local_search(removed, face_count, present, time_limit):
        """Try to remove edges from 'removed' set (i.e., add them back) if possible.
        Also try 1-for-1 swaps: remove one edge from 'removed', add another edge to 'removed'
        that was previously present, if it allows net improvement."""
        
        best_removed = set(removed)
        best_size = len(removed)
        
        while time.time() < time_limit:
            improved = False
            
            # Try to add back each removed edge
            list_removed = list(removed)
            random.shuffle(list_removed)
            
            for e in list_removed:
                if e not in removed:
                    continue
                
                # Check if adding e back would violate any face
                # A face f is violated if all 4 edges of f are present after adding e
                # i.e., face_count[f] == 3 and e is the 4th edge
                safe = True
                for f in edge_faces[e]:
                    if face_count[f] == 3:
                        safe = False
                        break
                
                if safe:
                    # Add e back
                    removed.discard(e)
                    present[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    improved = True
                    if len(removed) < best_size:
                        best_size = len(removed)
                        best_removed = set(removed)
            
            if improved:
                continue
            
            # Try 1-for-1 swaps: remove edge e from 'removed', add edge e2 to 'removed'
            # This is useful if e2 covers more "critical" faces than e
            # Actually, this doesn't directly help minimize size. 
            # Instead, let's try: pick a random present edge e2, remove it (add to removed),
            # then see if we can add back some removed edges.
            
            if time.time() > time_limit - 2:
                break
            
            # Random perturbation: pick a random present edge, remove it,
            # then try to add back removed edges
            present_edges = [e for e in range(n_edges) if present[e]]
            if not present_edges:
                break
            
            e2 = random.choice(present_edges)
            
            # Temporarily remove e2
            present[e2] = False
            removed.add(e2)
            for f in edge_faces[e2]:
                face_count[f] -= 1
            
            # Now try to add back removed edges
            list_removed = list(removed)
            random.shuffle(list_removed)
            added_back = 0
            for e in list_removed:
                if e not in removed:
                    continue
                if e == e2 and added_back == 0:
                    # Don't add back e2 immediately; try others first
                    continue
                safe = True
                for f in edge_faces[e]:
                    if face_count[f] == 3:
                        safe = False
                        break
                if safe:
                    removed.discard(e)
                    present[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    added_back += 1
            
            # If we added back more than 1 edge, we improved
            if added_back >= 1:
                if len(removed) < best_size:
                    best_size = len(removed)
                    best_removed = set(removed)
                improved = True
            
            if not improved:
                # Try a few more random perturbations
                for _ in range(3):
                    if time.time() > time_limit - 1:
                        break
                    present_edges = [e for e in range(n_edges) if present[e]]
                    if not present_edges:
                        break
                    e2 = random.choice(present_edges)
                    present[e2] = False
                    removed.add(e2)
                    for f in edge_faces[e2]:
                        face_count[f] -= 1
                    
                    list_removed = list(removed)
                    random.shuffle(list_removed)
                    added_back = 0
                    for e in list_removed:
                        if e not in removed:
                            continue
                        if e == e2:
                            continue
                        safe = True
                        for f in edge_faces[e]:
                            if face_count[f] == 3:
                                safe = False
                                break
                        if safe:
                            removed.discard(e)
                            present[e] = True
                            for f in edge_faces[e]:
                                face_count[f] += 1
                            added_back += 1
                    
                    if added_back >= 1:
                        if len(removed) < best_size:
                            best_size = len(removed)
                            best_removed = set(removed)
                        improved = True
                        break
                    else:
                        # Revert: add e2 back to present
                        present[e2] = True
                        removed.discard(e2)
                        for f in edge_faces[e2]:
                            face_count[f] += 1
        
        return best_removed, best_size
    
    # Run multiple greedy + local search iterations
    best_removed = set(range(n_edges))  # worst case: remove all
    best_size = n_edges
    
    num_iterations = 0
    while time.time() < deadline - 5:
        removed, face_count, present = greedy_min_removal()
        removed, size = local_search(removed, face_count, present, deadline - 2)
        
        if size < best_size:
            best_size = size
            best_removed = set(removed)
        
        num_iterations += 1
        if num_iterations >= 100:
            break
    
    # Build result: selected edges = all edges - removed
    result = []
    for i in range(n_edges):
        if i not in best_removed:
            result.append([edges[i][0], edges[i][1]])
    
    return result
