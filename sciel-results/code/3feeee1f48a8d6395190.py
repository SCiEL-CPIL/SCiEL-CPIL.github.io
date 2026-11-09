    def _extend_with_backtrack(path, used, end_index, bit_order, rng, max_depth=5, beam_width=2):
        # path is list, used is set
        # We want to extend path from end_index (0 or 1)
        
        def dfs(current_path, current_used, depth):
            if depth > max_depth:
                return current_path
            
            cur = current_path[end_index]
            
            candidates = []
            path_arr = np.array(current_path, dtype=np.int64)
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in current_used:
                    continue
                if not _is_valid_extension(v, path_arr, current_used, end_index):
                    continue
                candidates.append(v)
            
            if not candidates:
                return current_path
            
            scored = []
            for v in candidates:
                opts = _count_safe_options(v, path_arr, current_used)
                scored.append((opts, rng.random(), v))
            scored.sort(key=lambda x: (x[0], x[1]))
            
            best_path = current_path
            for i in range(min(beam_width, len(scored))):
                v = scored[i][2]
                
                new_path = current_path[:]
                new_used = current_used.copy()
                
                if end_index == 1:
                    new_path.append(v)
                else:
                    new_path.insert(0, v)
                new_used.add(v)
                
                res_path = dfs(new_path, new_used, depth + 1)
                
                if len(res_path) > len(best_path):
                    best_path = res_path
                    
            return best_path

        return dfs(path, used, 0)
    