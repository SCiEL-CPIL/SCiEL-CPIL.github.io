                    # Get candidate set from R1 and R2 using rev_neighbors
                    cand_set = None
                    for p in r1_points:
                        ns = set(rev_neighbors[p])
                        cand_set = ns if cand_set is None else (cand_set & ns)
                        if not cand_set:
                            break
                            
                    if not cand_set:
                        continue
                        
                    for p in r2_points:
                        ns = set(rev_neighbors[p])
                        cand_set = cand_set & ns
                        if not cand_set:
                            break
