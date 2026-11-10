import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115.0
    d = 7
    n_vertices = 1 << d
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Generate all edges
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = len(edges)
                edges.append((u, v))
    
    n_edges = len(edges)
    
    # Precompute face structure
    faces = []
    face_of_edge = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for mask in range(1 << d):
                if mask & mask_ij != 0:
                    continue
                v0 = mask
                v1 = mask | (1 << i)
                v2 = mask | (1 << j)
                v3 = mask | (1 << i) | (1 << j)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                face_edges = (edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4])
                face_idx = len(faces)
                faces.append(face_edges)
                for e_idx in face_edges:
                    face_of_edge[e_idx].append(face_idx)
    
    n_faces = len(faces)
    
    # Convert face_of_edge to list of lists for fast iteration
    # face_of_edge is already list of lists of ints
    
    def can_add(face_count, e_idx):
        """Check if edge e_idx can be added without forming a C4."""
        for f_idx in face_of_edge[e_idx]:
            if face_count[f_idx] == 3:
                return False
        return True

    def add_edge(selected, face_count, e_idx):
        selected[e_idx] = True
        for f_idx in face_of_edge[e_idx]:
            face_count[f_idx] += 1

    def remove_edge(selected, face_count, e_idx):
        selected[e_idx] = False
        for f_idx in face_of_edge[e_idx]:
            face_count[f_idx] -= 1

    def random_greedy_init(rng):
        """Generate an initial solution using randomized greedy."""
        selected = np.zeros(n_edges, dtype=bool)
        face_count = np.zeros(n_faces, dtype=np.int8)
        count = 0
        
        edge_indices = list(range(n_edges))
        rng.shuffle(edge_indices)
        
        for e_idx in edge_indices:
            if can_add(face_count, e_idx):
                selected[e_idx] = True
                for f_idx in face_of_edge[e_idx]:
                    face_count[f_idx] += 1
                count += 1
        
        return selected, face_count, count

    def local_search(selected, face_count, count, deadline):
        """
        Local search with:
        1. Add all possible edges.
        2. Try 1-for-1 swaps.
        """
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Pass 1: Add all possible edges
            for e_idx in range(n_edges):
                if not selected[e_idx]:
                    if can_add(face_count, e_idx):
                        selected[e_idx] = True
                        for f_idx in face_of_edge[e_idx]:
                            face_count[f_idx] += 1
                        count += 1
                        improved = True
            
            # Pass 2: Try 1-for-1 swaps
            # To make this faster, we only try swaps that might lead to improvement.
            # A swap is (remove e_out, add e_in).
            # If e_in was blocked by e_out, it might be addable now.
            # We can precompute for each e_in, which e_outs block it?
            # e_in is blocked if any face f containing e_in has count 3.
            # If we remove e_out, count[f] decreases for f in face_of_edge[e_out].
            # So e_in becomes unblocked if e_out is in face_of_edge[e_in] AND e_out was the 4th edge? No, just if it reduced count from 3 to 2.
            
            # Simple approach: Randomized swaps to avoid O(N^2) worst case if we do it smartly.
            # But N=448 is small. O(N^2) = 200k. 
            # Let's do a structured swap:
            
            # We want to find e_in such that it is blocked ONLY by e_out?
            # Or just try all pairs?
            # 448 * 448 = 200k iterations.
            # Each iteration: remove e_out (O(4)), check e_in (O(4)), add e_in (O(4)).
            # Total ~ 200k * 12 = 2.4M ops per pass.
            # If we do 100 passes, 240M ops. In Python, this is ~10-20s.
            # We have ~115s. So we can afford several rounds.
            
            if time.time() > deadline - 2.0:
                break
                
            # To speed up, we can iterate e_out, then e_in.
            # Optimization: Only consider e_in that are "almost" addable.
            # An edge is "almost" addable if it is blocked by exactly 1 face with count 3?
            # If it's blocked by 2 faces, removing one edge won't help unless that edge is in both faces.
            
            # Let's just do the full O(N^2) but break early if we find a good swap?
            # Actually, we want to maximize count. A swap keeps count same.
            # So why do swaps? To enable *future* additions.
            # So after a swap, we should re-run Pass 1.
            
            # Strategy:
            # 1. Try to add edges.
            # 2. If no edges can be added, try a swap.
            #    Randomly pick e_out from selected, e_in from unselected.
            #    If swap is valid (e_in addable after removing e_out), do it.
            #    Then re-run Pass 1.
            
            # To make it deterministic-ish but effective:
            # Try all e_out. For each, try all e_in.
            # If a swap allows adding *more* edges in Pass 1, it's good.
            # But we don't know that without doing it.
            
            # Heuristic: Try swaps that "unlock" many edges.
            
            # For now, let's do a randomized local search step for swaps.
            # Pick 100 random swaps.
            
            for _ in range(50): # Limit swaps per pass to avoid hanging
                if time.time() > deadline - 2.0:
                    break
                
                # Pick random selected edge to remove
                selected_indices = np.where(selected)[0]
                if len(selected_indices) == 0:
                    break
                e_out = selected_indices[np.random.randint(len(selected_indices))]
                
                # Pick random unselected edge to add
                unselected_indices = np.where(~selected)[0]
                if len(unselected_indices) == 0:
                    break
                e_in = unselected_indices[np.random.randint(len(unselected_indices))]
                
                # Try swap
                # Remove e_out
                selected[e_out] = False
                for f_idx in face_of_edge[e_out]:
                    face_count[f_idx] -= 1
                
                # Check if e_in can be added
                if can_add(face_count, e_in):
                    # Add e_in
                    selected[e_in] = True
                    for f_idx in face_of_edge[e_in]:
                        face_count[f_idx] += 1
                    # Count remains same, but structure changed.
                    # We mark improved to trigger Pass 1 again?
                    # Actually, count didn't increase.
                    # But it might allow increases.
                    # We should not mark improved=True based on count, but on state change.
                    # Let's force a re-evaluation.
                    improved = True 
                else:
                    # Revert
                    selected[e_out] = True
                    for f_idx in face_of_edge[e_out]:
                        face_count[f_idx] += 1
        
        return selected, face_count, count

    best_selected = None
    best_count = 0
    best_face_count = None
    
    rng = random.Random(42)
    
    # Iterated Local Search
    while time.time() < deadline - 10.0:
        # 1. Initialize
        selected, face_count, count = random_greedy_init(rng)
        
        # 2. Local Search
        selected, face_count, count = local_search(selected, face_count, count, deadline)
        
        if count > best_count:
            best_selected = selected.copy()
            best_count = count
            best_face_count = face_count.copy()
            
            # If we are close to the upper bound, we can tighten the deadline for subsequent loops
            # or just keep going.
            
    # Final Polish
    if best_selected is not None and time.time() < deadline - 2.0:
        best_selected, best_face_count, best_count = local_search(
            best_selected, best_face_count, best_count, deadline
        )
    
    # Convert to output format
    result = []
    for e_idx in range(n_edges):
        if best_selected[e_idx]:
            u, v = edges[e_idx]
            result.append([u, v])
    
    return result
