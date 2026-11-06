"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n_vertices = 2 ** d
    deadline = time.time() + 140  # safety margin before 150s kill

    # Precompute all edges and their face memberships
    all_edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx

    total_edges = len(all_edges)  # 448

    # For each edge, precompute the set of faces it belongs to
    # A face is identified by (v, i, j) where v is a corner, i<j are the two dimensions
    # Edge (x, y) with y = x ^ (1<<i) belongs to faces where x is a corner and dim i is one of the two dims
    # Face (x, i, j): edges are (x, x^(1<<i)), (x, x^(1<<j)), (x^(1<<i), x^(1<<i)^(1<<j)), (x^(1<<j), x^(1<<j)^(1<<i))

    # Build face -> list of edge indices
    face_to_edges = {}
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                # Face with corner x, dims i and j
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                face_key = (x, i, j)
                face_to_edges[face_key] = [edge_to_idx[e] for e in (e1, e2, e3, e4)]

    # For each edge, list of faces containing it
    edge_to_faces = [[] for _ in range(total_edges)]
    for face_key, edges in face_to_edges.items():
        for eidx in edges:
            edge_to_faces[eidx].append(face_key)

    # Fast validity check: an edge can be added if no face has all 4 edges selected
    # Use a counter per face: how many of its 4 edges are in the set
    def make_counter():
        return {fk: 0 for fk in face_to_edges}

    def try_add_edge(counter, eidx, selected):
        """Try to add edge eidx. Returns True if successful."""
        if selected[eidx]:
            return True
        # Check all faces containing this edge
        for fk in edge_to_faces[eidx]:
            if counter[fk] == 3:
                return False
        # All good, add it
        selected[eidx] = True
        for fk in edge_to_faces[eidx]:
            counter[fk] += 1
        return True

    def remove_edge(counter, eidx, selected):
        selected[eidx] = False
        for fk in edge_to_faces[eidx]:
            counter[fk] -= 1

    best_count = 0
    best_selected = None

    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(total_edges))
        rng.shuffle(order)

        selected = [False] * total_edges
        counter = make_counter()
        count = 0

        for eidx in order:
            if try_add_edge(counter, eidx, selected):
                count += 1

        return selected, counter, count

    def add_edge(counter, eidx, selected):
        selected[eidx] = True
        for fk in edge_to_faces[eidx]:
            counter[fk] += 1

    def local_search(selected, counter, count, max_iters=2000):
        """Try 2-opt swaps: remove one edge, try to add up to 2 rejected edges."""
        rng = random.Random(12345)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2:
            improved = False
            iters += 1
            
            selected_edges = [i for i in range(total_edges) if selected[i]]
            unselected_edges = [i for i in range(total_edges) if not selected[i]]
            
            rng.shuffle(selected_edges)
            rng.shuffle(unselected_edges)

            for e_remove in selected_edges:
                # Remove e_remove
                remove_edge(counter, e_remove, selected)
                count -= 1
                
                best_gain = 0
                best_adds = []
                
                # Try adding 1 edge
                for e_add in unselected_edges:
                    if selected[e_add]: 
                        continue
                    if try_add_edge(counter, e_add, selected):
                        best_gain = 1
                        best_adds = [e_add]
                        # Try adding a second edge
                        for e_add2 in unselected_edges:
                            if e_add2 == e_add or selected[e_add2]: 
                                continue
                            if try_add_edge(counter, e_add2, selected):
                                best_gain = 2
                                best_adds = [e_add, e_add2]
                                break
                            else:
                                # Undo e_add2
                                remove_edge(counter, e_add2, selected)
                        break # Found an improvement, stop trying other e_adds for this e_remove
                
                # If we found an improvement, it is already applied in 'selected' and 'counter'
                if best_gain > 0:
                    improved = True
                    break
                else:
                    # No improvement found, restore e_remove
                    add_edge(counter, e_remove, selected)
                    count += 1

        return selected, counter, count

    # Run multiple random greedy + local search
    num_restarts = 50
    for attempt in range(num_restarts):
        if time.time() > deadline - 10:
            break
        seed = 12345 + attempt * 7919
        selected, counter, count = greedy_random(seed)

        if count > best_count:
            best_count = count
            best_selected = selected.copy()

        # Local search
        if time.time() < deadline - 15:
            selected, counter, count = local_search(selected, counter, count, max_iters=500)

        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # If no local search was run, at least we have greedy results
    if best_selected is None:
        selected, counter, count = greedy_random(42)
        best_selected = selected
        best_count = count

    out = []
    for i in range(total_edges):
        if best_selected[i]:
            x, y = all_edges[i]
            out.append([x, y])

    return out
# EVOLVE-BLOCK-END
