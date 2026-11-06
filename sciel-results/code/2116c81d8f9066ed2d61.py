import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 7 * 64 = 448
    
    # Precompute edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
                edge_index[(y, x)] = idx
                
    # Precompute 4-cycles (faces)
    # A face is defined by two distinct dimensions i, j and a vertex x 
    # where x has 0 in bits i and j (to avoid duplicates, or just any x and check uniqueness)
    # Actually, a face is a set of 4 vertices. 
    # Vertices: x, x^(1<<i), x^(1<<j), x^(1<<(i|j))
    # Edges: (x, x^i), (x^i, x^i^j), (x^j, x^i^j), (x, x^j)
    
    faces = []
    face_edge_indices = []
    edge_faces = [[] for _ in range(n_edges)]
    
    seen_faces = set()
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all "bases" x such that bit i and bit j are 0? 
            # No, just iterate all x, but ensure we don't duplicate the face.
            # A face is uniquely identified by the set of its 4 vertices.
            # To generate each face exactly once, we can require that the 
            # "lowest" vertex in the face (in some ordering) is x, and 
            # x has 0 in bits i and j.
            # Simpler: iterate all x. The set of 4 vertices is unique for the face.
            # We will use a set of frozensets of vertices to deduplicate.
            for x in range(n_vertices):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                # Check if this is a valid face (it always is geometrically)
                # But we need to avoid duplicates. 
                # The face is the same regardless of which vertex we start from.
                # We can canonicalize by taking the min vertex.
                min_v = min(v1, v2, v3, v4)
                # Also, the face is determined by the two dimensions and the "offset" 
                # in the other d-2 dimensions.
                # Let's just use a set of frozensets of the 4 edge indices.
                
                e1 = edge_index.get((min(v1, v2), max(v1, v2)))
                e2 = edge_index.get((min(v2, v4), max(v2, v4)))
                e3 = edge_index.get((min(v4, v3), max(v4, v3)))
                e4 = edge_index.get((min(v3, v1), max(v3, v1)))
                
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue # Should not happen
                
                face_key = frozenset([e1, e2, e3, e4])
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    faces.append(face_key)
                    face_edge_indices.append(list(face_key))
                    for e_idx in face_key:
                        edge_faces[e_idx].append(len(face_edge_indices) - 1)

    n_faces = len(face_edge_indices)
    
    # Helper to check if adding edge e is valid
    # It is valid if for all faces f containing e, not all other 3 edges are present.
    # We maintain a count of "filled" edges in each face? 
    # Or just check the 3 neighbors.
    # Since d=7, each edge is in d-1 = 6 faces. So checking is fast.
    
    def is_valid_add(e, selected):
        for f_idx in edge_faces[e]:
            f_edges = face_edge_indices[f_idx]
            # Check if the other 3 are in selected
            count = 0
            for ee in f_edges:
                if ee == e:
                    continue
                if ee in selected:
                    count += 1
                    if count == 3:
                        return False
            # If count < 3, it's fine for this face
        return True

    # Greedy Initialization
    def greedy_init():
        selected = set()
        random_edges = list(range(n_edges))
        random.shuffle(random_edges)
        for e in random_edges:
            if is_valid_add(e, selected):
                selected.add(e)
        return selected

    # Simulated Annealing / Local Search
    best_selected = greedy_init()
    best_size = len(best_selected)
    
    current_selected = set(best_selected)
    current_size = best_size
    
    # Precompute list of unselected edges for fast sampling? 
    # Just use random.randint and check membership.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9999
    
    steps = 0
    max_steps = 5000000 # Will be limited by time
    
    while time.time() < deadline:
        steps += 1
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge
            e = random.randint(0, n_edges - 1)
            if e in current_selected:
                continue
            
            if is_valid_add(e, current_selected):
                # Accept immediately (uphill)
                current_selected.add(e)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_selected = set(current_selected)
            else:
                # Try a swap: remove one conflicting edge
                # Find a face that is completed by e
                conflict_face = -1
                for f_idx in edge_faces[e]:
                    f_edges = face_edge_indices[f_idx]
                    other_edges = [ee for ee in f_edges if ee != e]
                    if all(ee in current_selected for ee in other_edges):
                        conflict_face = f_idx
                        break
                
                if conflict_face != -1:
                    f_edges = face_edge_indices[conflict_face]
                    other_edges = [ee for ee in f_edges if ee != e]
                    # Pick one to remove
                    e_remove = random.choice(other_edges)
                    # Check if removing e_remove allows adding e? Yes, by definition.
                    # But does removing e_remove break other faces? No, removing edges never creates 4-cycles.
                    # So this move is always valid.
                    # Size remains constant.
                    # Accept with some probability to allow exploration? 
                    # Since it doesn't improve size, we should be cautious.
                    # But it might lead to a state where we can add more edges later.
                    if random.random() < 0.1: # Low probability to accept neutral moves
                        current_selected.remove(e_remove)
                        current_selected.add(e)
                        # current_size unchanged
                        
        elif move_type < 0.9:
            # Try to remove an edge (downhill)
            if current_size > 0:
                # Pick a random edge to remove
                # To make it efficient, we can pick from a random sample
                # Or just pick a random edge and see if it's selected
                e = random.randint(0, n_edges - 1)
                if e in current_selected:
                    # Probability to accept removal: depends on T
                    # Delta E = -1
                    # P = exp(-Delta E / T) = exp(1/T) which is > 1, so always accept?
                    # No, we want to minimize energy (maximize edges). 
                    # If we define Energy = -Size, then Delta E = 1.
                    # P = exp(-1/T).
                    if random.random() < (2.71828 ** (-1.0 / T)):
                        current_selected.remove(e)
                        current_size -= 1
                        
        else:
            # Random restart / Perturbation
            # Remove k random edges to escape local minimum
            k = random.randint(5, 15)
            for _ in range(k):
                if current_size == 0:
                    break
                # Pick a random edge from current_selected
                # To avoid O(N) pick, we can just try random indices
                # But current_selected is a set, so we can convert to list if small, 
                # or just try random indices until we find one.
                # Better: maintain a list of selected edges? 
                # For now, just try random indices.
                removed = False
                for _ in range(10):
                    e = random.randint(0, n_edges - 1)
                    if e in current_selected:
                        current_selected.remove(e)
                        current_size -= 1
                        removed = True
                        break
                if not removed:
                    break
                    
        T *= alpha
        if T < T_min:
            T = T_min

    # Final greedy polish
    # Try to add any remaining valid edges
    changed = True
    while changed:
        changed = False
        for e in range(n_edges):
            if e not in best_selected:
                if is_valid_add(e, best_selected):
                    best_selected.add(e)
                    changed = True
    
    # Convert back to list of [u, v]
    result = []
    for e_idx in best_selected:
        u, v = edges[e_idx]
        result.append([u, v])
        
    return result
