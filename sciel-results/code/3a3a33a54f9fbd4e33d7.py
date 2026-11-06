        def find_indep6(adj):
            def dfs(candidates, size, path):
                if size == 6:
                    return tuple(path)
                if candidates == 0:
                    return None
                if candidates.bit_count() < size:
                    return None
                
                v_bit = candidates & -candidates
                v = v_bit.bit_length() - 1
                
                # Include v
                new_cand = candidates & ~(adj[v] | (1<<v))
                res = dfs(new_cand, size + 1, path + [v])
                if res:
                    return res
                    
                # Exclude v
                new_cand2 = candidates & ~v_bit
                res = dfs(new_cand2, size, path)
                if res:
                    return res
                    
                return None
            return dfs(full_mask, 0, [])
