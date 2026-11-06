import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2
    
    # 1. Precomputation
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
                
    # Generate all 2-faces (4-cycles)
    # A face is defined by two distinct dimensions i, j and a base vertex x
    # where x has bits i and j set to 0 (canonical form)
    faces = [] # List of lists of 4 edge indices
    edge_to_faces = [[] for _ in range(n_edges_total)]
    
    for i in range(d):
        for j in range(i+1, d):
            # Iterate over all vertices x where bit i and j are 0
            # There are 2^(d-2) such vertices
            for x in range(n_vertices):
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    # The 4 vertices of the face
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    # The 4 edges
                    e01 = edge_to_idx[(min(v0, v1), max(v0, v1))]
                    e13 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                    e32 = edge_to_idx[(min(v3, v2), max(v3, v2))]
                    e20 = edge_to_idx[(min(v2, v0), max(v2, v0))]
                    
                    face = [e01, e13, e32, e20]
                    faces.append(face)
                    for e in face:
                        edge_to_faces[e].append(len(faces) - 1)
                        
    n_faces = len(faces)
    
    # State
    selected = [False] * n_edges_total
    face_counts = [0] * n_faces
    num_selected = 0
    
    def can_add(edge_idx):
        for f_idx in edge_to_faces[edge_idx]:
            if face_counts[f_idx] == 3:
                return False
        return True

    def add_edge(edge_idx):
        nonlocal num_selected
        if selected[edge_idx]:
            return
        if not can_add(edge_idx):
            return
        selected[edge_idx] = True
        num_selected += 1
        for f_idx in edge_to_faces[edge_idx]:
            face_counts[f_idx] += 1

    def remove_edge(edge_idx):
        nonlocal num_selected
        if not selected[edge_idx]:
            return
        selected[edge_idx] = False
        num_selected -= 1
        for f_idx in edge_to_faces[edge_idx]:
            face_counts[f_idx] -= 1

    def greedy_fill(randomize=True):
        nonlocal num_selected
        unselected = [i for i in range(n_edges_total) if not selected[i]]
        if randomize:
            random.shuffle(unselected)
        
        added = 0
        for e in unselected:
            if can_add(e):
                add_edge(e)
                added += 1
        return added

    # Initial Solution: Run greedy multiple times
    best_num = 0
    best_state = [False] * n_edges_total
    best_face_counts = [0] * n_faces
    
    for _ in range(10):
        # Reset
        selected = [False] * n_edges_total
        face_counts = [0] * n_faces
        num_selected = 0
        
        greedy_fill(True)
        
        if num_selected > best_num:
            best_num = num_selected
            best_state = selected[:]
            best_face_counts = face_counts[:]
            
    # Copy best state to current
    selected = best_state[:]
    face_counts = best_face_counts[:]
    num_selected = best_num
    
    # Local Search
    unselected_indices = list(range(n_edges_total))
    
    while time.time() - start_time < time_limit:
        # Try to add edges
        # Pick random unselected edges and try to add
        # We can maintain a list of "blocked" vs "unblocked" but simple random sampling is okay
        # To be faster, we can just iterate through a random subset of unselected edges
        
        # Get a random sample of unselected edges
        # Efficient way: just pick random indices and check if unselected
        improved = False
        for _ in range(50): # Try 50 random additions
            e = random.randint(0, n_edges_total - 1)
            if not selected[e]:
                if can_add(e):
                    add_edge(e)
                    improved = True
                    if num_selected > best_num:
                        best_num = num_selected
                        best_state = selected[:]
                        best_face_counts = face_counts[:]
        
        if not improved:
            # Local optimum reached (or stuck). Perform a kick.
            # Strategy: Remove K edges that are in faces with count 3
            # Identify candidate edges for removal
            candidates = []
            for e in range(n_edges_total):
                if selected[e]:
                    # Check if it's in any face with count 3
                    for f_idx in edge_to_faces[e]:
                        if face_counts[f_idx] == 3:
                            candidates.append(e)
                            break # Only need to know if it's a candidate
            
            if not candidates:
                # No candidates? Remove random edges
                candidates = [e for e in range(n_edges_total) if selected[e]]
            
            if not candidates:
                break # No edges left? Unlikely.
                
            # Pick K candidates to remove
            K = 5
            if len(candidates) < K:
                K = len(candidates)
            
            to_remove = random.sample(candidates, K)
            
            # Save current state to revert if kick fails
            old_selected = selected[:]
            old_face_counts = face_counts[:]
            old_num = num_selected
            
            for e in to_remove:
                remove_edge(e)
                
            # Greedy fill
            greedy_fill(True)
            
            if num_selected > best_num:
                best_num = num_selected
                best_state = selected[:]
                best_face_counts = face_counts[:]
            else:
                # Revert
                selected = old_selected
                face_counts = old_face_counts
                num_selected = old_num

    # Construct output
    out = []
    for i in range(n_edges_total):
        if best_state[i]:
            out.append(list(edges[i]))
            
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Edges found: {len(res)}")
