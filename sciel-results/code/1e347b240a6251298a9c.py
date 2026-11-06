<<<<<<< SEARCH
    def swap_local_search(selected, face_count, time_limit):
        """Swap-based local search: 1-for-2 and 2-for-3 swaps."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-for-2 Swap
            selected_edges = [e for e in range(n_edges) if selected[e]]
            random.shuffle(selected_edges)
            
            for e_remove in selected_edges:
                if time.time() > time_limit:
                    break
                
                # Temporarily remove
                selected[e_remove] = False
                for f in edge_faces[e_remove]:
                    face_count[f] -= 1
                
                # Try to add back 2 edges
                added_back = []
                unselected_now = [e for e in range(n_edges) if not selected[e]]
                random.shuffle(unselected_now)
                
                for e_add in unselected_now:
                    if len(added_back) >= 2:
                        break
                    can_add = True
                    for f in edge_faces[e_add]:
                        if face_count[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e_add] = True
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        added_back.append(e_add)
                
                if len(added_back) < 2:
                    # Revert
                    selected[e_remove] = True
                    for f in edge_faces[e_remove]:
                        face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break  # Found an improvement, restart loop
            
            if improved:
                continue
                
            # 2-for-3 Swap
            if time.time() < time_limit:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                random.shuffle(selected_edges)
                
                # Try pairs of edges to remove
                # To keep it fast, we only try a limited number of pairs
                for i in range(min(50, len(selected_edges))):
                    if time.time() > time_limit:
                        break
                    e1 = selected_edges[i]
                    for j in range(i + 1, min(50, len(selected_edges))):
                        e2 = selected_edges[j]
                        
                        # Temporarily remove e1, e2
                        selected[e1] = False
                        selected[e2] = False
                        for f in edge_faces[e1]:
                            face_count[f] -= 1
                        for f in edge_faces[e2]:
                            face_count[f] -= 1
                        
                        # Try to add back 3 edges
                        added_back = []
                        unselected_now = [e for e in range(n_edges) if not selected[e]]
                        random.shuffle(unselected_now)
                        
                        for e_add in unselected_now:
                            if len(added_back) >= 3:
                                break
                            can_add = True
                            for f in edge_faces[e_add]:
                                if face_count[f] >= 3:
                                    can_add = False
                                    break
                            if can_add:
                                selected[e_add] = True
                                for f in edge_faces[e_add]:
                                    face_count[f] += 1
                                added_back.append(e_add)
                        
                        if len(added_back) < 3:
                            # Revert
                            selected[e1] = True
                            selected[e2] = True
                            for f in edge_faces[e1]:
                                face_count[f] += 1
                            for f in edge_faces[e2]:
                                face_count[f] += 1
                            for e_add in added_back:
                                selected[e_add] = False
                                for f in edge_faces[e_add]:
                                    face_count[f] -= 1
                        else:
                            improved = True
                            break
                    if improved:
                        break
        return improved
=======
    def get_addable_edges(selected, face_count):
        """Return list of unselected edges that can be added without violating any face."""
        addable = []
        for e in range(n_edges):
            if not selected[e]:
                can_add = True
                for f in edge_faces[e]:
                    if face_count[f] >= 3:
                        can_add = False
                        break
                if can_add:
                    addable.append(e)
        return addable

    def swap_local_search(selected, face_count, time_limit):
        """Swap-based local search: 1-for-2 and 2-for-3 swaps, optimized."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-for-2 Swap
            selected_edges = [e for e in range(n_edges) if selected[e]]
            random.shuffle(selected_edges)
            
            for e_remove in selected_edges:
                if time.time() > time_limit:
                    break
                
                # Temporarily remove
                selected[e_remove] = False
                for f in edge_faces[e_remove]:
                    face_count[f] -= 1
                
                # Try to add back 2 edges from addable set
                added_back = []
                addable = get_addable_edges(selected, face_count)
                random.shuffle(addable)
                
                for e_add in addable:
                    if len(added_back) >= 2:
                        break
                    selected[e_add] = True
                    for f in edge_faces[e_add]:
                        face_count[f] += 1
                    added_back.append(e_add)
                
                if len(added_back) < 2:
                    # Revert
                    selected[e_remove] = True
                    for f in edge_faces[e_remove]:
                        face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break  # Found an improvement, restart loop
            
            if improved:
                continue
                
            # 2-for-3 Swap
            if time.time() < time_limit:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                random.shuffle(selected_edges)
                
                # Try pairs of edges to remove
                n_try = min(80, len(selected_edges))
                for i in range(n_try):
                    if time.time() > time_limit:
                        break
                    e1 = selected_edges[i]
                    for j in range(i + 1, n_try):
                        e2 = selected_edges[j]
                        
                        # Check if removing e1,e2 could free up 3+ addable edges
                        # Quick heuristic: count faces affected
                        affected_faces = set(edge_faces[e1]) | set(edge_faces[e2])
                        if len(affected_faces) < 4:
                            continue
                        
                        # Temporarily remove e1, e2
                        selected[e1] = False
                        selected[e2] = False
                        for f in edge_faces[e1]:
                            face_count[f] -= 1
                        for f in edge_faces[e2]:
                            face_count[f] -= 1
                        
                        # Try to add back 3 edges
                        added_back = []
                        addable = get_addable_edges(selected, face_count)
                        random.shuffle(addable)
                        
                        for e_add in addable:
                            if len(added_back) >= 3:
                                break
                            selected[e_add] = True
                            for f in edge_faces[e_add]:
                                face_count[f] += 1
                            added_back.append(e_add)
                        
                        if len(added_back) < 3:
                            # Revert
                            selected[e1] = True
                            selected[e2] = True
                            for f in edge_faces[e1]:
                                face_count[f] += 1
                            for f in edge_faces[e2]:
                                face_count[f] += 1
                            for e_add in added_back:
                                selected[e_add] = False
                                for f in edge_faces[e_add]:
                                    face_count[f] -= 1
                        else:
                            improved = True
                            break
                    if improved:
                        break
        return improved
>>>>>>> REPLACE

