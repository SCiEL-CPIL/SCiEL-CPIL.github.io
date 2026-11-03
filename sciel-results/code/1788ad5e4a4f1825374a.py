    def local_search_numpy(code, deadline):
        code = list(code)
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
            
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_count[neighbors[c]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Temporarily remove
                in_code[c_remove] = False
                cover_count[neighbors[c_remove]] -= 1
                uncovered = (cover_count == 0)
                
                if not uncovered.any():
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Need replacement
                    U_len = uncovered.sum()
                    
                    # We need w such that:
                    # 1. not in_code[w]
                    # 2. neighbors[w] covers all uncovered
                    #    i.e. sum(uncovered[neighbors[w]]) == U_len
                    
                    coverage = uncovered[neighbors].sum(axis=1)
                    
                    # Candidates: coverage == U_len and not in_code
                    candidates = np.where((coverage == U_len) & (~in_code))[0]
                    
                    if len(candidates) > 0:
                        chosen = candidates[random.randint(0, len(candidates) - 1)]
                        
                        # Perform swap
                        in_code[chosen] = True
                        cover_count[neighbors[chosen]] += 1
                        code[i] = chosen
                        improved = True
                        break
                    else:
                        # Cannot replace, restore
                        cover_count[neighbors[c_remove]] += 1
                        in_code[c_remove] = True
                        continue
                        
        return code
