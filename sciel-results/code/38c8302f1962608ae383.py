<<<<<<< SEARCH
        # --- Phase 2c: Remove One, Add Two (Net +1) ---
        # This is more complex. If we have time, we can try it.
        # Given the deadline, and that swaps are already O(|S|) rebuilds, 
        # double adds are O(|S| * |outside|^2) checks if done naively.
        # We'll skip this for now to ensure stability and speed, 
        # as pure adds and swaps usually suffice for good B2[2] sizes.
=======
        # --- Phase 2c: Remove One, Add Two (Net +1) ---
        if outside and len(S) > 0 and time.time() < deadline - 5:
            remove_attempts = 0
            max_remove_attempts = 10
            
            while remove_attempts < max_remove_attempts and time.time() < deadline - 5:
                remove_attempts += 1
                
                rem_idx = rng.randrange(len(S))
                rem_val = S[rem_idx]
                
                S_arr = np.array(S)
                rem_sums = rem_val + S_arr
                counts[rem_sums] -= 1
                
                S_without_rem = [S[i] for i in range(len(S)) if i != rem_idx]
                S_wo_arr = np.array(S_without_rem)
                
                out_arr = np.array(outside)
                
                if len(S_without_rem) == 0:
                    all_new_sums = 2 * out_arr[:, None]
                else:
                    cross_sums = out_arr[:, None] + S_wo_arr[None, :]
                    diag_sums = 2 * out_arr[:, None]
                    all_new_sums = np.concatenate([cross_sums, diag_sums], axis=1)
                    
                gathered_counts = counts[all_new_sums]
                max_new_count = gathered_counts.max(axis=1) + 1
                feasible_indices = np.where(max_new_count <= MAX_REP)[0]
                
                if feasible_indices.size >= 2:
                    feasible_list = feasible_indices[:20]
                    found_pair = False
                    
                    for i in range(len(feasible_list)):
                        if found_pair:
                            break
                        for j in range(i + 1, len(feasible_list)):
                            if time.time() > deadline - 5:
                                break
                            
                            x1 = int(out_arr[feasible_list[i]])
                            x2 = int(out_arr[feasible_list[j]])
                            
                            sums_x1 = x1 + S_wo_arr
                            sums_x1 = np.append(sums_x1, 2*x1)
                            
                            S_wo_arr_with_x1 = np.append(S_wo_arr, x1)
                            sums_x2_after_x1 = x2 + S_wo_arr_with_x1
                            sums_x2_after_x1 = np.append(sums_x2_after_x1, 2*x2)
                            
                            combined = np.concatenate([sums_x1, sums_x2_after_x1])
                            
                            if len(combined) > 0:
                                sum_counts = np.bincount(combined, minlength=2*N+1)
                                max_violation = np.max(counts[combined] + sum_counts[combined])
                                if max_violation <= MAX_REP:
                                    S.pop(rem_idx)
                                    S_set.remove(rem_val)
                                    
                                    S.append(x1)
                                    S_set.add(x1)
                                    S.append(x2)
                                    S_set.add(x2)
                                    
                                    outside.remove(x1)
                                    outside.remove(x2)
                                    
                                    counts = rebuild_counts(S)
                                    
                                    improved = True
                                    found_pair = True
                                    break
                            
                            if time.time() > deadline - 5:
                                break
                
                if not improved:
                    counts[rem_sums] += 1
                
                if improved:
                    break
>>>>>>> REPLACE
