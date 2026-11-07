            used_k20 = set()
            
            def dfs(t_node, parent_t, k_parent):
                # ...
                for k_v in candidates:
                    if k_v in used_k20:
                        continue
                    # ...
                    used_k20.add(k_v)
                    # ...
                    # backtrack:
                    used_k20.remove(k_v)
