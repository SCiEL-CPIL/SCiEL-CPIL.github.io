    def local_search(selected_indices, deadline):
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        last_improve_time = time.time()
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            improved = False
            # Increase batch size to 100-200
            for _ in range(100):
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                c_prime = neighbors[c][rng.integers(1, ball_size)]
                
                if c_prime in selected_set:
                    continue
                
                # Tentative swap
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                if np.any(coverage_count[neighbors[c]] == 0):
                    # Revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    continue
                
                # Accept swap
                pos = selected_list.index(c)
                selected_list[pos] = c_prime
                selected_set.discard(c)
                selected_set.add(c_prime)
                
                # Try to prune
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    selected_set = set(selected_list)
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        improved = True
                        last_improve_time = time.time()
                else:
                    # Revert swap
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    pos = selected_list.index(c_prime)
                    selected_list[pos] = c
                    selected_set.discard(c_prime)
                    selected_set.add(c)
            
            # Restart if stuck
            if time.time() - last_improve_time > 3.0:
                selected_list = best_code[:]
                selected_set = set(selected_list)
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                last_improve_time = time.time()
                
        return best_code
