    def local_search_2opt(code, deadline):
        code = list(code)
        code_set = set(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            n_code = len(code)
            for i_idx in range(n_code):
                if time.time() >= deadline:
                    break
                if i_idx >= len(code):
                    continue
                for j_idx in range(i_idx + 1, n_code):
                    if time.time() >= deadline:
                        break
                    if j_idx >= len(code):
                        continue
                    
                    c_i = code[i_idx]
                    c_j = code[j_idx]
                    
                    # Temporarily remove c_i and c_j
                    counts[neighbors_arr[c_i, :]] -= 1
                    counts[neighbors_arr[c_j, :]] -= 1
                    
                    uncovered_mask = (counts == 0)
                    if not np.any(uncovered_mask):
                        # Both can be removed
                        code.pop(j_idx)
                        code.pop(i_idx)
                        code_set.remove(c_j)
                        code_set.remove(c_i)
                        improved = True
                        break
                    else:
                        uncovered_indices = np.where(uncovered_mask)[0]
                        
                        # A single word covers at most 22 neighbors. 
                        # If more than 22 are uncovered, 1 word cannot cover them all.
                        if len(uncovered_indices) > 22:
                            counts[neighbors_arr[c_i, :]] += 1
                            counts[neighbors_arr[c_j, :]] += 1
                            continue

                        u0 = uncovered_indices[0]
                        candidate_set = set(reverse_cover_arr[u0, :reverse_cover_counts[u0]].tolist())
                        
                        for u in uncovered_indices[1:]:
                            rc = reverse_cover_arr[u, :reverse_cover_counts[u]].tolist()
                            candidate_set &= set(rc)
                            if not candidate_set:
                                break
                            
                        if candidate_set:
                            candidate_set.discard(c_i)
                            candidate_set.discard(c_j)
                            
                            # Filter out candidates already in code
                            # Since code_set is maintained, we can check membership
                            valid_candidates = [w for w in candidate_set if w not in code_set]
                            
                            if valid_candidates:
                                w = random.choice(valid_candidates)
                                code.pop(j_idx)
                                code.pop(i_idx)
                                code.append(w)
                                code_set.remove(c_j)
                                code_set.remove(c_i)
                                code_set.add(w)
                                
                                counts[neighbors_arr[w, :]] += 1
                                
                                improved = True
                                break
                            else:
                                counts[neighbors_arr[c_i, :]] += 1
                                counts[neighbors_arr[c_j, :]] += 1
                                continue
                        else:
                            counts[neighbors_arr[c_i, :]] += 1
                            counts[neighbors_arr[c_j, :]] += 1
                            continue
                        
            if improved:
                continue
            else:
                break
                
        return code
