            improved = True
            while improved:
                improved = False
                # Shuffle indices to randomize greedy choice
                indices = list(range(total_edges))
                rng.shuffle(indices)
                for e in indices:
                    if not selected[e]:
                        valid = True
                        for f_idx in edge_to_faces[e]:
                            if face_counts[f_idx] > 2:
                                valid = False
                                break
                        if valid:
                            selected[e] = True
                            for f_idx in edge_to_faces[e]:
                                face_counts[f_idx] += 1
                            improved = True
