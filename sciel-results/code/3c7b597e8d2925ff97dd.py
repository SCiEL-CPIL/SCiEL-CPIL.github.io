import random
import time
import math
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2  # 448
    
    # Time limit guard
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit

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
            
    # Map face to edges (for repair, we need to check counts)
    # face_edge_lists already provides this.

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

    # 2. Initialization: Spectral Partitioning + Local Search
    # Build conflict graph: vertices are edges of Q7, edge between two Q7-edges if they share a face
    # We want to find a maximum cut in this conflict graph (edges in different partitions = kept)
    
    # Build adjacency matrix for conflict graph
    conflict_adj = np.zeros((n_edges, n_edges), dtype=np.int8)
    for f_idx, f_edges in enumerate(face_edge_lists):
        for i in range(4):
            for j in range(i+1, 4):
                e1, e2 = f_edges[i], f_edges[j]
                conflict_adj[e1, e2] = 1
                conflict_adj[e2, e1] = 1
    
    # Compute Laplacian: L = D - A
    deg = conflict_adj.sum(axis=1).astype(np.float64)
    L = np.diag(deg) - conflict_adj.astype(np.float64)
    
    # Compute Fiedler vector (second smallest eigenvector)
    # Use numpy's eigenvalue solver
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Find index of second smallest eigenvalue
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
    except:
        # Fallback: random initialization
        fiedler_vec = np.random.randn(n_edges)
    
    # Partition: edges with positive fiedler value in set A, negative in set B
    # We'll keep edges that are "in the cut" - but actually we want to keep edges such that no face has all 4
    # Let me reconsider: we want to remove edges to hit all faces
    # Spectral approach: minimize the number of faces with all 4 edges kept
    # This is equivalent to maximizing the number of faces hit
    # Let's use the fiedler vector to assign edges to two groups, then greedily remove edges to fix conflicts
    
    # Assign edges to groups based on fiedler vector sign
    group = np.sign(fiedler_vec)
    group[group == 0] = 1  # Treat zeros as positive
    
    # Start with all edges kept
    kept = set(range(n_edges))
    face_counts = [0] * n_faces
    for e in kept:
        for f_idx in edge_faces[e]:
            face_counts[f_idx] += 1
    
    # Greedy repair: remove edges from faces that have all 4 edges kept
    # Prioritize removal of edges that are in the "minority" group to respect spectral partition
    removed_count = 0
    max_removals = n_edges // 2  # Safety limit
    
    while removed_count < max_removals:
        # Find faces with all 4 edges kept
        bad_faces = [f_idx for f_idx in range(n_faces) if face_counts[f_idx] == 4]
        if not bad_faces:
            break
        
        # Pick a random bad face
        f_idx = random.choice(bad_faces)
        
        # Choose which edge to remove from this face
        # Prefer edges from the minority group
        face_edges = face_edge_lists[f_idx]
        minority_edges = [e for e in face_edges if group[e] == -1]
        majority_edges = [e for e in face_edges if group[e] == 1]
        
        if minority_edges:
            # Remove a random minority edge
            e_to_remove = random.choice(minority_edges)
        else:
            # All in majority, remove random
            e_to_remove = random.choice(face_edges)
        
        # Remove this edge
        do_remove(e_to_remove, kept, face_counts)
        removed_count += 1
    
    # Now do maximal extension to add back any edges that can be added
    maximal_extension(kept, face_counts)
            
    # Ensure validity (should be valid by construction)
    # Perform maximal extension to see if we can add back some removed edges
    # The hitting set might have removed more than necessary if the greedy choice was poor.
    maximal_extension(kept, face_counts)
    
    current_size = len(kept)
    best_size = current_size
    best_kept = set(kept)
    best_face_counts = face_counts[:]

    # 3. Simulated Annealing with Kicks
    T = 3.0
    T_min = 0.005
    cooling = 0.9995
    
    # Kick parameters
    min_kick = 3
    max_kick = 15
    
    # Track best and use it for restarts
    restart_count = 0
    max_restarts = 20
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit:
            break
            
        # Determine kick size
        kick_size = random.randint(min_kick, max_kick)
        if kick_size > current_size:
            kick_size = max(1, current_size // 4)
            
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
                restart_count = 0  # Reset restart counter on improvement
        else:
            # Accept with probability exp(delta / T)
            prob = math.exp(delta / T)
            if random.random() < prob:
                kept = next_kept
                face_counts = next_face_counts
                current_size = next_size
                
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            
            # If stuck for a while, restart from best solution
            restart_count += 1
            if restart_count >= max_restarts:
                # Reset to best and increase temperature
                kept = set(best_kept)
                face_counts = best_face_counts[:]
                current_size = best_size
                T = 2.0
                restart_count = 0
            
    # Final polish: Try to add any remaining edges
    maximal_extension(best_kept, best_face_counts)
    
    # Additional local search: try swapping edges
    for _ in range(100):
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        kept_list = list(best_kept)
        removed_list = [e for e in range(n_edges) if e not in best_kept]
        
        if not removed_list:
            break
        
        # Try random swaps
        for _ in range(10):
            e_in = random.choice(kept_list)
            e_out = random.choice(removed_list)
            
            # Check if we can remove e_in and add e_out
            test_face_counts = best_face_counts[:]
            for f_idx in edge_faces[e_in]:
                test_face_counts[f_idx] -= 1
            
            # Check if e_out can be added
            can_add_out = True
            for f_idx in edge_faces[e_out]:
                if test_face_counts[f_idx] >= 3:
                    can_add_out = False
                    break
            
            if can_add_out:
                # Perform the swap
                for f_idx in edge_faces[e_out]:
                    test_face_counts[f_idx] += 1
                
                # This swap maintains the same size, so we need a different criterion
                # Let's skip swaps for now and just do the maximal extension
                pass
        
        # Just do maximal extension again
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
