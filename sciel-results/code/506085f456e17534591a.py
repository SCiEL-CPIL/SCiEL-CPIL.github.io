import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y, i))
                edge_to_idx[(x, y)] = idx
    
    num_edges = len(all_edges)  # 448

    # Precompute for each edge the list of faces (as tuples of 4 edge indices)
    # that contain this edge. A face is a 2D square in the hypercube.
    # Edge (x,y) with bit i: for each j != i, the C4 involves edges:
    #   (x,y), (x^j, y^j), (x, x^j), (y, y^j)
    # We store for each edge e, the list of (face_id, [other 3 edge indices])
    # where face_id is a canonical identifier.
    
    # Better: for each edge, store the list of faces it belongs to.
    # Each face is identified by its 4 edges. Two edges conflict if they share a face.
    # Actually, the constraint is: no face should have all 4 edges selected.
    # So for edge e, adding it is invalid if there exists a face containing e
    # where the other 3 edges are all already selected.
    
    # Precompute: for each edge index, list of (other_edge_idx1, other_edge_idx2, other_edge_idx3)
    # for each face containing this edge.
    
    edge_faces = [[] for _ in range(num_edges)]
    
    for idx, (x, y, bit_i) in enumerate(all_edges):
        for j in range(d):
            if j == bit_i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            # The four edges in this face:
            # e1 = (x, y) -> idx
            # e2 = (xj, yj)
            # e3 = (x, xj)
            # e4 = (y, yj)
            e2 = (min(xj, yj), max(xj, yj))
            e3 = (min(x, xj), max(x, xj))
            e4 = (min(y, yj), max(y, yj))
            
            idx2 = edge_to_idx[e2]
            idx3 = edge_to_idx[e3]
            idx4 = edge_to_idx[e4]
            
            # For edge idx, the other 3 edges are idx2, idx3, idx4
            edge_faces[idx].append((idx2, idx3, idx4))
    
    # Now is_valid_add can check: for edge idx, for each (a,b,c) in edge_faces[idx],
    # check if a, b, c are all in chosen_set (as indices)
    
    def is_valid_add_idx(chosen_set, idx):
        """Check if adding edge with index idx creates a C4. chosen_set is a set of edge indices."""
        for (a, b, c) in edge_faces[idx]:
            if a in chosen_set and b in chosen_set and c in chosen_set:
                return False
        return True

    def greedy_randomized(seed):
        """Run greedy with randomized edge order."""
        rng = random.Random(seed)
        edge_order = list(range(num_edges))
        rng.shuffle(edge_order)

        chosen_set = set()
        out = []

        for idx in edge_order:
            if is_valid_add_idx(chosen_set, idx):
                chosen_set.add(idx)
                x, y, _ = all_edges[idx]
                out.append([x, y])

        return out, chosen_set

    def simulated_annealing(out, chosen_set, rng, start_time, time_limit):
        """
        Simulated Annealing to improve the solution.
        Moves:
        1. Try to add a random edge. If valid, accept.
        2. If stuck (no add possible in a batch), remove a random edge.
        """
        current_score = len(chosen_set)
        best_score = current_score
        best_out = [list(e) for e in out]
        
        # Keep track of chosen edges as a list for removal
        chosen_list = list(chosen_set)
        
        # Annealing parameters
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() - start_time < time_limit - 2.0:
            # Try to add a random edge
            added = False
            # Try a few random candidates
            for _ in range(15):
                idx = rng.randrange(num_edges)
                if idx not in chosen_set:
                    if is_valid_add_idx(chosen_set, idx):
                        chosen_set.add(idx)
                        x, y, _ = all_edges[idx]
                        out.append([x, y])
                        chosen_list.append(idx)
                        current_score += 1
                        added = True
                        if current_score > best_score:
                            best_score = current_score
                            best_out = [list(e) for e in out]
                        break
            
            if not added:
                # Stuck. Try to remove an edge to escape local minimum.
                if chosen_list:
                    pos = rng.randrange(len(chosen_list))
                    remove_idx = chosen_list[pos]
                    
                    # Remove
                    chosen_set.discard(remove_idx)
                    out.pop(pos)
                    chosen_list.pop(pos)
                    current_score -= 1
                    
                    # Try to add back edges
                    added_count = 0
                    for _ in range(30):
                        idx = rng.randrange(num_edges)
                        if idx not in chosen_set:
                            if is_valid_add_idx(chosen_set, idx):
                                chosen_set.add(idx)
                                x, y, _ = all_edges[idx]
                                out.append([x, y])
                                chosen_list.append(idx)
                                added_count += 1
                                current_score += 1
                                if current_score > best_score:
                                    best_score = current_score
                                    best_out = [list(e) for e in out]
                                if added_count >= 3:
                                    break
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_out, best_score

    # Run multiple randomized greedy restarts
    best_result = None
    best_count = 0

    # Determine number of restarts based on available time
    # Each greedy pass is fast (~ms), SA takes most of the time
    # Do fewer restarts but longer SA for each
    
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 15.0:
            break

        rng = random.Random(restart * 12345 + 67890)
        out, chosen_set = greedy_randomized(restart)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement with time budget
        sa_time_limit = min(time.time() + 2.0, time_limit - 3.0)
        if time.time() < sa_time_limit:
            out, score = simulated_annealing(out, chosen_set, rng, start_time, sa_time_limit)
            if score > best_count:
                best_count = score
                best_result = out

    if best_result is None:
        # Fallback to deterministic greedy
        chosen_set = set()
        best_result = []
        for idx in range(num_edges):
            if is_valid_add_idx(chosen_set, idx):
                chosen_set.add(idx)
                x, y, _ = all_edges[idx]
                best_result.append([x, y])

    return best_result
