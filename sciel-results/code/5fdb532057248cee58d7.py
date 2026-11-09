import time
import random
from collections import defaultdict

def construct():
    """
    Improved local search with edge-swap moves and better face tracking.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (kill at 150s)
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Build face list
    faces = []
    for base in range(n_vertices):
        for d1 in range(d):
            if (base >> d1) & 1:
                continue
            for d2 in range(d1 + 1, d):
                if (base >> d2) & 1:
                    continue
                e1 = (base, base ^ (1 << d1))
                e2 = (base, base ^ (1 << d2))
                e3 = (base ^ (1 << d1), base ^ (1 << d1) ^ (1 << d2))
                e4 = (base ^ (1 << d2), base ^ (1 << d1) ^ (1 << d2))
                
                e1 = (min(e1), max(e1))
                e2 = (min(e2), max(e2))
                e3 = (min(e3), max(e3))
                e4 = (min(e4), max(e4))
                
                face_edges = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                faces.append(face_edges)
    
    n_faces = len(faces)
    
    # For each edge, list of faces it belongs to
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    best_count = 0
    best_solution = None
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(sol):
        """Check if solution has no face with all 4 edges."""
        for face in faces:
            if all(sol[e] for e in face):
                return False
        return True
    
    def count_edges(sol):
        return sum(1 for e in range(n_edges) if sol[e])
    
    def greedy_initial(seed=None):
        """Generate a greedy initial solution with random order."""
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = [0] * n_faces
        
        order = list(range(n_edges))
        rng.shuffle(order)
        
        for ei in order:
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    can_add = False
                    break
            if can_add:
                sol[ei] = True
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        return sol
    
    def local_search(sol, seed=None):
        """Improve solution via local search with swap moves."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        # Track bad faces (all 4 edges present)
        bad_faces = set()
        for fi in range(n_faces):
            if face_counts[fi] == 4:
                bad_faces.add(fi)
        
        # Track addable edges (can be added without creating a bad face)
        def can_add_edge(ei):
            if sol[ei]:
                return False
            for fi in edge_faces[ei]:
                if face_counts[fi] >= 3:
                    return False
            return True
        
        improved = True
        stall_counter = 0
        
        while improved and time_left() > 10:
            improved = False
            
            if bad_faces:
                # Remove an edge from a bad face - pick edge in most bad faces
                bad_face_list = list(bad_faces)
                rng.shuffle(bad_face_list)
                
                best_removal = None
                best_score = -1
                for fi in bad_face_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            cnt = 0
                            for f2 in edge_faces[ei]:
                                if f2 in bad_faces:
                                    cnt += 1
                            if cnt > best_score:
                                best_score = cnt
                                best_removal = (fi, ei)
                
                if best_removal:
                    fi, ei = best_removal
                    sol[ei] = False
                    for f2 in edge_faces[ei]:
                        face_counts[f2] -= 1
                        if face_counts[f2] == 3 and f2 in bad_faces:
                            bad_faces.discard(f2)
                    if face_counts[fi] < 4:
                        bad_faces.discard(fi)
                    improved = True
                    stall_counter = 0
            else:
                # No bad faces - try to add edges
                addable = []
                for ei in range(n_edges):
                    if not sol[ei] and can_add_edge(ei):
                        addable.append(ei)
                
                if addable:
                    rng.shuffle(addable)
                    added_any = False
                    for ei in addable:
                        if time_left() < 10:
                            break
                        if can_add_edge(ei):
                            sol[ei] = True
                            for fi in edge_faces[ei]:
                                face_counts[fi] += 1
                            added_any = True
                            improved = True
                    if added_any:
                        stall_counter = 0
                    else:
                        stall_counter += 1
                else:
                    # Try swap moves: remove one edge, add another
                    # Find edges that if removed would allow adding others
                    swap_candidates = []
                    for ei in range(n_edges):
                        if sol[ei]:
                            # Check if removing this edge allows adding others
                            for fi in edge_faces[ei]:
                                if face_counts[fi] >= 3:
                                    # This edge is "blocking" face fi
                                    # Find edges that could be added if we remove ei
                                    for ej in range(n_edges):
                                        if not sol[ej] and ej != ei:
                                            # Check if ej can be added if ei is removed
                                            can_add_if_removed = True
                                            for f2 in edge_faces[ej]:
                                                if f2 == fi:
                                                    # This face would have count-1 after removing ei
                                                    if face_counts[f2] - 1 >= 3:
                                                        can_add_if_removed = False
                                                        break
                                                elif face_counts[f2] >= 3:
                                                    can_add_if_removed = False
                                                    break
                                            if can_add_if_removed:
                                                swap_candidates.append((ei, ej))
                                                break
                            if swap_candidates:
                                break
                        if len(swap_candidates) > 50:
                            break
                    
                    if swap_candidates:
                        rng.shuffle(swap_candidates)
                        for ei, ej in swap_candidates[:10]:  # Limit swaps per iteration
                            if time_left() < 10:
                                break
                            # Perform swap: remove ei, add ej
                            sol[ei] = False
                            for f2 in edge_faces[ei]:
                                face_counts[f2] -= 1
                                if face_counts[f2] == 3 and f2 in bad_faces:
                                    bad_faces.discard(f2)
                            
                            sol[ej] = True
                            for f2 in edge_faces[ej]:
                                face_counts[f2] += 1
                                if face_counts[f2] == 4:
                                    bad_faces.add(f2)
                            improved = True
                            stall_counter = 0
                            break
                    else:
                        stall_counter += 1
                
                # If stuck for too long, try a random perturbation
                if stall_counter > 5:
                    # Randomly remove a few edges and re-add greedily
                    num_remove = rng.randint(3, 10)
                    selected_edges = [ei for ei in range(n_edges) if sol[ei]]
                    rng.shuffle(selected_edges)
                    for ei in selected_edges[:num_remove]:
                        sol[ei] = False
                        for f2 in edge_faces[ei]:
                            face_counts[f2] -= 1
                            if face_counts[f2] == 3 and f2 in bad_faces:
                                bad_faces.discard(f2)
                    
                    # Re-add greedily
                    order = list(range(n_edges))
                    rng.shuffle(order)
                    for ei in order:
                        if not sol[ei] and can_add_edge(ei):
                            sol[ei] = True
                            for f2 in edge_faces[ei]:
                                face_counts[f2] += 1
                            if face_counts[f2] == 4:
                                bad_faces.add(f2)
                    
                    stall_counter = 0
                    improved = True
        
        return sol
    
    # Multiple restarts with different seeds
    attempts = 0
    while time_left() > 20:
        attempts += 1
        seed = attempts * 7919
        
        # Generate initial solution
        sol = greedy_initial(seed)
        
        # Local search
        sol = local_search(sol, seed)
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 30:
            break
    
    # Final fallback: if we haven't found anything good, use the original greedy
    if best_solution is None or best_count < 250:
        chosen = set()
        out = []
        for x in range(n_vertices):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c = x ^ (1 << j)
                            e = y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        out.append([x, y])
        return out
    
    # Convert solution to list of edges
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    
    return result
