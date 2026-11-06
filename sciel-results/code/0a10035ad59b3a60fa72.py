import random
import time
import math

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2  # 448
    
    # Time limit guard
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit

    # 1. Precomputation
    # Generate edges
    edges = []
    edge_map = {} # (u, v) -> index
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
    
    n_edges = len(edges)
    
    # Generate faces (squares)
    # A face is defined by fixing d-2 bits and varying the other 2.
    # We represent a face by the set of 4 edge indices it contains.
    face_edge_lists = []
    seen_faces = set()
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n_vertices):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_map.get((min(v1, v2), max(v1, v2)))
                e2 = edge_map.get((min(v1, v3), max(v1, v3)))
                e3 = edge_map.get((min(v2, v4), max(v2, v4)))
                e4 = edge_map.get((min(v3, v4), max(v3, v4)))
                
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue
                
                face_key = frozenset([e1, e2, e3, e4])
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    face_edge_lists.append([e1, e2, e3, e4])
    
    n_faces = len(face_edge_lists)
    
    # Map edge to faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edge_lists):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
            
    # Helper: Check if adding edge e is valid given face_counts
    def can_add(e, face_counts):
        for f_idx in edge_faces[e]:
            if face_counts[f_idx] >= 3:
                return False
        return True

    # Helper: Add edge e to kept set and update counts
    def do_add(e, kept, face_counts):
        kept.add(e)
        for f_idx in edge_faces[e]:
            face_counts[f_idx] += 1

    # Helper: Remove edge e from kept set and update counts
    def do_remove(e, kept, face_counts):
        kept.discard(e)
        for f_idx in edge_faces[e]:
            face_counts[f_idx] -= 1

    # Helper: Greedy Maximal Extension
    # Adds any edge that can be added, in random order
    def maximal_extension(kept, face_counts):
        candidates = [e for e in range(n_edges) if e not in kept]
        random.shuffle(candidates)
        for e in candidates:
            if can_add(e, face_counts):
                do_add(e, kept, face_counts)

    # Helper: Local Search - try to swap edges to improve
    # For each edge in kept, try removing it and adding a different edge
    def local_search(kept, face_counts, max_iter=10000):
        current_size = len(kept)
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            improved = False
            # Try removing a random edge from kept
            kept_list = list(kept)
            random.shuffle(kept_list)
            for e_remove in kept_list:
                if time.time() - start_time > time_limit:
                    break
                # Try to add a random edge not in kept
                candidates = [e for e in range(n_edges) if e not in kept]
                random.shuffle(candidates)
                for e_add in candidates[:20]:  # Try at most 20 candidates
                    # Check if we can remove e_remove and add e_add
                    # First, check if adding e_add is valid after removing e_remove
                    # We need to check the faces of e_add
                    valid = True
                    for f_idx in edge_faces[e_add]:
                        # Count how many edges in this face are in kept, excluding e_remove
                        count = 0
                        for e in face_edge_lists[f_idx]:
                            if e == e_remove:
                                continue
                            if e in kept:
                                count += 1
                        if count >= 3:
                            valid = False
                            break
                    if valid:
                        # Perform the swap
                        do_remove(e_remove, kept, face_counts)
                        do_add(e_add, kept, face_counts)
                        current_size = len(kept)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return kept, face_counts

    # 2. Initialization
    # Start with a good hitting set.
    # Greedy Hitting Set:
    # We want to remove edges to hit all faces.
    # Equivalent: Start with all edges kept, remove edges until valid.
    removed = set()
    face_hits = [0] * n_faces
    uncovered_faces = list(range(n_faces))
    random.shuffle(uncovered_faces)
    
    while uncovered_faces:
        f_idx = uncovered_faces.pop()
        if face_hits[f_idx] > 0:
            continue
        
        # Find best edge in this face to remove
        best_e = -1
        max_cov = -1
        for e_idx in face_edge_lists[f_idx]:
            if e_idx in removed:
                continue
            cov = len(edge_faces[e_idx])
            if cov > max_cov:
                max_cov = cov
                best_e = e_idx
        
        if best_e != -1:
            removed.add(best_e)
            for ff in edge_faces[best_e]:
                face_hits[ff] += 1
            
            if all(face_hits[f] > 0 for f in range(n_faces)):
                uncovered_faces = []
                break

    # Convert to kept set
    kept = set(range(n_edges)) - removed
    face_counts = [0] * n_faces
    for e in kept:
        for f_idx in edge_faces[e]:
            face_counts[f_idx] += 1
            
    # Ensure validity (should be valid by construction)
    # Perform maximal extension to see if we can add back some removed edges
    maximal_extension(kept, face_counts)
    
    current_size = len(kept)
    best_size = current_size
    best_kept = set(kept)
    best_face_counts = face_counts[:]

    # 3. Simulated Annealing with Kicks
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # Kick parameters
    min_kick = 5
    max_kick = 20
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit:
            break
            
        # Determine kick size
        # Larger kicks help escape deep local optima
        kick_size = random.randint(min_kick, max_kick)
        if kick_size > current_size:
            kick_size = max(1, current_size // 2)
            
        # Select random edges to remove
        kept_list = list(kept)
        random.shuffle(kept_list)
        to_remove = set(kept_list[:kick_size])
        
        # Create a copy of the state to test
        next_kept = set(kept)
        next_face_counts = face_counts[:]
        
        # Apply removals
        for e in to_remove:
            do_remove(e, next_kept, next_face_counts)
            
        # Greedy Repair: Add back edges in random order
        # Candidates are all edges not in next_kept
        candidates = [e for e in range(n_edges) if e not in next_kept]
        random.shuffle(candidates)
        
        for e in candidates:
            if can_add(e, next_face_counts):
                do_add(e, next_kept, next_face_counts)
                
        next_size = len(next_kept)
        
        # Acceptance
        delta = next_size - current_size
        
        if delta >= 0:
            # Accept
            kept = next_kept
            face_counts = next_face_counts
            current_size = next_size
            
            if current_size > best_size:
                best_size = current_size
                best_kept = set(kept)
                best_face_counts = face_counts[:]
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = math.exp(delta / T)
            if random.random() < prob:
                kept = next_kept
                face_counts = next_face_counts
                current_size = next_size
                
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Reset to best and restart with higher T?
            # For now, just stay at T_min. The kick mechanism should handle exploration.
            
    # Final polish: Try to add any remaining edges
    maximal_extension(best_kept, best_face_counts)
    
    # Construct output
    out = []
    for e_idx in best_kept:
        u, v = edges[e_idx]
        out.append([u, v])
        
    return out

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
