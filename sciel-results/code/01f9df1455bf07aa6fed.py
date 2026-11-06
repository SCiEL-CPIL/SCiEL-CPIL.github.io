import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    edge_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_list.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx += 1
                
    assert len(edge_list) == 448

    faces = []
    face_to_edges = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            diff = u ^ v
            if bin(diff).count('1') == 2:
                bits = []
                for k in range(d):
                    if diff & (1 << k):
                        bits.append(k)
                i, j = bits[0], bits[1]
                
                w = u ^ (1 << i)
                z = u ^ (1 << j)
                
                e1 = (min(u, w), max(u, w))
                e2 = (min(w, v), max(w, v))
                e3 = (min(v, z), max(v, z))
                e4 = (min(z, u), max(z, u))
                
                edges = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                
                face_id = len(faces)
                faces.append((u, v, w, z))
                face_to_edges.append(edges)
                
                for e_idx in edges:
                    edge_to_faces[e_idx].append(face_id)

    n_faces = len(faces)
    
    def run_search(time_limit):
        start_time = time.time()
        
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        
        def is_valid_add(e_idx):
            for f in edge_to_faces[e_idx]:
                if face_counts[f] == 3:
                    return False
            return True

        def add_edge(e_idx):
            selected[e_idx] = True
            for f in edge_to_faces[e_idx]:
                face_counts[f] += 1

        def remove_edge(e_idx):
            selected[e_idx] = False
            for f in edge_to_faces[e_idx]:
                face_counts[f] -= 1

        # Greedy Init
        edge_indices = list(range(n_edges))
        random.shuffle(edge_indices)
        for e_idx in edge_indices:
            if is_valid_add(e_idx):
                add_edge(e_idx)
        
        current_score = sum(selected)
        best_score = current_score
        best_state = selected[:]
        
        unselected = [i for i in range(n_edges) if not selected[i]]
        selected_list = [i for i in range(n_edges) if selected[i]]
        
        # Helper to swap-remove from list
        def swap_remove(lst, idx):
            lst[idx] = lst[-1]
            lst.pop()

        while time.time() - start_time < time_limit:
            # 1. Try to add a random unselected edge
            added = False
            if unselected:
                # Try a few random picks
                for _ in range(5):
                    if not unselected:
                        break
                    k = random.randint(0, len(unselected) - 1)
                    e_idx = unselected[k]
                    if is_valid_add(e_idx):
                        add_edge(e_idx)
                        swap_remove(unselected, k)
                        selected_list.append(e_idx)
                        current_score += 1
                        added = True
                        if current_score > best_score:
                            best_score = current_score
                            best_state = selected[:]
                        break
            
            if not added:
                # 2. Stuck. Try to remove and re-add
                if not selected_list:
                    break
                
                e_out = random.choice(selected_list)
                
                # Save state
                old_selected = selected[:]
                old_face_counts = face_counts[:]
                old_unselected = unselected[:]
                old_selected_list = selected_list[:]
                old_score = current_score
                
                remove_edge(e_out)
                # Remove from selected_list
                # Find index of e_out in selected_list
                # Since it's random, we can just use list.remove which is O(N)
                # Or we can track indices. N=448 is small.
                try:
                    selected_list.remove(e_out)
                except ValueError:
                    pass # Should not happen
                unselected.append(e_out)
                current_score -= 1
                
                # Try to add edges
                # Loop until no more can be added or max attempts
                for _ in range(20):
                    if not unselected:
                        break
                    k = random.randint(0, len(unselected) - 1)
                    e_in = unselected[k]
                    if is_valid_add(e_in):
                        add_edge(e_in)
                        swap_remove(unselected, k)
                        selected_list.append(e_in)
                        current_score += 1
                        # Continue trying to add more
                    else:
                        # If we fail too many times, break?
                        # Just break to keep step short?
                        # No, we want to fill the hole.
                        # But if we scan all and find none, we are stuck again.
                        # Let's just break after first success to keep variance?
                        # Actually, adding multiple is better.
                        # But if we add one, the state changes, so other edges might become valid.
                        # So we should keep trying.
                        pass
                
                if current_score >= old_score:
                    if current_score > best_score:
                        best_score = current_score
                        best_state = selected[:]
                else:
                    # Revert
                    selected = old_selected
                    face_counts = old_face_counts
                    unselected = old_unselected
                    selected_list = old_selected_list
                    current_score = old_score

        result = []
        for i, is_sel in enumerate(best_state):
            if is_sel:
                u, v = edge_list[i]
                result.append([u, v])
        return result

    return run_search(115)
