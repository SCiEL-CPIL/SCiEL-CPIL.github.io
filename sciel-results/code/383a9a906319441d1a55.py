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
    deadline = start_time + 115 # 5s margin

    # Strategy: Minimize D (deleted edges) such that every face is hit by D.
    # This is equivalent to maximizing S (selected edges).
    # Initial D = empty.
    # Greedy: Add edge e that hits the most currently unhit faces.
    
    def greedy_cover():
        # count_unhit_faces[e] = number of faces containing e that are currently unhit
        count_unhit_faces = [0] * n_edges
        # is_hit[f] = whether face f is hit by D
        is_hit = [False] * n_faces
        # Initially all faces are unhit
        for f in range(n_faces):
            for e in face_edges[f]:
                count_unhit_faces[e] += 1
        
        D = set()
        # We need to pick edges. To make it randomized for diversity, we can shuffle or pick randomly among top candidates.
        # Standard greedy picks max. Let's pick max with some randomness or just max.
        
        for _ in range(n_edges):
            if time.time() > deadline - 10:
                break
            
            # Find edge with max count_unhit_faces
            max_count = -1
            candidates = []
            for e in range(n_edges):
                if e not in D and count_unhit_faces[e] > 0:
                    if count_unhit_faces[e] > max_count:
                        max_count = count_unhit_faces[e]
                        candidates = [e]
                    elif count_unhit_faces[e] == max_count:
                        candidates.append(e)
            
            if not candidates:
                break
                
            # Pick one randomly from candidates to allow different runs
            e = random.choice(candidates)
            D.add(e)
            
            # Update counts: for each face f containing e, if it was unhit, it becomes hit.
            # If it becomes hit, we decrement count_unhit_faces for all edges in f.
            for f in edge_faces[e]:
                if not is_hit[f]:
                    is_hit[f] = True
                    # Decrement count for all edges in this face
                    for e2 in face_edges[f]:
                        if e2 not in D: # We don't care about edges in D for future selection? 
                                         # Actually, if e2 is in D, it's already selected. 
                                         # But count_unhit_faces is used to decide what to add next.
                                         # If e2 is in D, we won't add it again.
                                         # However, the count should reflect how many *unhit* faces e2 covers.
                                         # If face f is now hit, e2 no longer "covers" an unhit face via f.
                                         # So we should decrement count_unhit_faces[e2] regardless of if e2 is in D?
                                         # Yes, because if we were to consider adding e2 (we won't, it's in D), 
                                         # its contribution to covering *new* faces is reduced.
                                         # But since e2 is in D, we don't care about its count for selection.
                                         # However, if we do local search later, we might remove e2.
                                         # For the greedy phase, we just need the counts to be correct for edges NOT in D.
                                         # But it's easier to just update all.
                                         pass
                    # Let's just update all edges in the face
                    for e2 in face_edges[f]:
                        count_unhit_faces[e2] -= 1
                        
        return D, is_hit

    def local_search_minimize_D(D, is_hit, time_limit):
        # D is a set of deleted edges
        # is_hit is a list of booleans for faces
        # We want to remove edges from D if possible.
        
        # Precompute for each edge e in D, the list of faces it hits that are *only* hit by e?
        # Or just check: if we remove e, does any face f containing e become unhit?
        # Face f becomes unhit if count_in_D[f] == 1 and e is the one.
        
        # Let's maintain count_in_D[f] for each face f.
        count_in_D = [0] * n_faces
        for f in range(n_faces):
            for e in face_edges[f]:
                if e in D:
                    count_in_D[f] += 1
        
        # Verify is_hit matches count_in_D > 0
        # for f in range(n_faces):
        #    if count_in_D[f] > 0 and not is_hit[f]: is_hit[f] = True
        #    if count_in_D[f] == 0 and is_hit[f]: is_hit[f] = False
            
        best_D = set(D)
        best_size = len(D)
        
        while time.time() < time_limit:
            improved = False
            
            # Try to remove each edge in D
            # Shuffle to avoid bias
            list_D = list(D)
            random.shuffle(list_D)
            
            for e in list_D:
                if e not in D:
                    continue
                
                # Check if e is critical
                critical = False
                for f in edge_faces[e]:
                    if count_in_D[f] == 1:
                        critical = True
                        break
                
                if not critical:
                    # Safe to remove
                    D.remove(e)
                    for f in edge_faces[e]:
                        count_in_D[f] -= 1
                    improved = True
                    if len(D) < best_size:
                        best_size = len(D)
                        best_D = set(D)
            
            if improved:
                continue
            
            # If no single removal works, try swaps?
            # Remove e, add e' that covers the faces e was uniquely covering?
            # This is complex. Let's try a random restart or more aggressive search.
            
            # Try 1-for-1 swap: Remove e, add e'
            # This only helps if e' covers the faces that e uniquely covered, 
            # and e' doesn't break anything else (it won't, since we are adding to D, 
            # making more faces hit).
            # Wait, if we remove e, some faces become unhit. We need e' to hit ALL of them.
            # And we want to keep |D| same or smaller.
            # If we remove e and add e', |D| stays same.
            # This doesn't minimize size directly, but might allow further removals.
            
            # Let's try: Pick a random e in D. Remove it. 
            # Identify faces that became unhit (count_in_D[f] became 0).
            # If none, we would have removed it in the previous step.
            # So there is at least one unhit face.
            # Find an edge e' that covers ALL currently unhit faces?
            # That's unlikely.
            # Instead, find e' that covers the *most* currently unhit faces.
            # Add e'.
            # This is a move.
            
            # Let's implement a simple swap heuristic:
            # 1. Pick random e in D.
            # 2. Remove e.
            # 3. Find unhit faces U.
            # 4. If U is empty, keep removed (shouldn't happen if previous step was thorough).
            # 5. Else, find e' not in D that maximizes |face_edges[e'] ∩ U|.
            # 6. If max coverage > 0, add e'.
            # 7. Repeat.
            
            if time.time() > time_limit - 1:
                break
                
            e_remove = random.choice(list(D))
            D.remove(e_remove)
            
            unhit_faces = []
            for f in edge_faces[e_remove]:
                count_in_D[f] -= 1
                if count_in_D[f] == 0:
                    unhit_faces.append(f)
            
            if not unhit_faces:
                # We just reduced size!
                if len(D) < best_size:
                    best_size = len(D)
                    best_D = set(D)
                improved = True
                continue
            
            # Find best e' to add
            best_e_add = -1
            best_cover = 0
            # Only consider edges that hit at least one unhit face
            # We can iterate all edges not in D
            # Optimization: precompute for each unhit face, the edges in it.
            # Union of these edges are candidates.
            candidates = set()
            for f in unhit_faces:
                for e in face_edges[f]:
                    if e not in D:
                        candidates.add(e)
                        
            for e_add in candidates:
                # Count how many unhit faces e_add covers
                cover = 0
                for f in edge_faces[e_add]:
                    if f in unhit_faces: # This check is O(1) if unhit_faces is a set? 
                                         # unhit_faces is a list. 
                                         # Let's make a set of unhit face IDs for quick lookup
                                         pass
                # This loop is slow. Let's use a set for unhit_faces.
                
            # Let's redo the swap logic with sets for speed
            unhit_set = set(unhit_faces)
            best_e_add = -1
            best_cover = 0
            for e_add in candidates:
                cover = 0
                for f in edge_faces[e_add]:
                    if f in unhit_set:
                        cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_e_add = e_add
            
            if best_e_add != -1 and best_cover > 0:
                D.add(best_e_add)
                for f in edge_faces[best_e_add]:
                    count_in_D[f] += 1
                # Note: Some faces might have become hit, some might still be unhit.
                # We don't update is_hit explicitly, we rely on count_in_D.
            else:
                # Revert
                D.add(e_remove)
                for f in edge_faces[e_remove]:
                    count_in_D[f] += 1

        return best_D, best_size

    # Phase 1: Greedy
    D, is_hit = greedy_cover()
    best_D = set(D)
    best_size = len(D)
    
    # Phase 2: Local Search
    # We need to pass count_in_D to local_search. 
    # Let's refactor local_search to compute count_in_D internally or take it.
    # For simplicity, let's just run local search on the set D.
    
    # Re-implementing local search to be self-contained and efficient
    def refine_D(D, time_limit):
        count_in_D = [0] * n_faces
        for f in range(n_faces):
            for e in face_edges[f]:
                if e in D:
                    count_in_D[f] += 1
        
        best_D = set(D)
        best_size = len(D)
        
        while time.time() < time_limit:
            improved = False
            list_D = list(D)
            random.shuffle(list_D)
            
            # 1. Removals
            for e in list_D:
                if e not in D: continue
                critical = False
                for f in edge_faces[e]:
                    if count_in_D[f] == 1:
                        critical = True
                        break
                if not critical:
                    D.remove(e)
                    for f in edge_faces[e]:
                        count_in_D[f] -= 1
                    improved = True
                    if len(D) < best_size:
                        best_size = len(D)
                        best_D = set(D)
            
            if improved:
                continue
                
            # 2. Swaps (1-for-1)
            if time.time() > time_limit - 2:
                break
                
            # Try to find a swap that allows a removal? 
            # Or just a swap that maintains size but changes structure?
            # Actually, if we can't remove, maybe we can swap e for e' such that e' is "better" 
            # (e.g., covers more faces that are currently hit only by e? No, that doesn't help removal).
            
            # Let's try a random restart of the greedy with a bias towards the current solution?
            # Or just break and do another greedy restart.
            
            break
            
        return best_D, best_size

    # Run Greedy + Refine
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        D, _ = greedy_cover()
        D, size = refine_D(D, deadline - 1)
        if size < best_size:
            best_size = size
            best_D = D

    # Build result: S = All Edges - D
    result = []
    for i in range(n_edges):
        if i not in best_D:
            result.append([edges[i][0], edges[i][1]])
            
    return result
