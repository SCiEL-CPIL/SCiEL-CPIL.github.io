    def run_greedy_reduction(code_indices):
        in_code = [False] * num_words
        cover_count = [0] * num_words
        
        for c in code_indices:
            in_code[c] = True
            for w in neighbors[c]:
                cover_count[w] += 1
                
        current_code = code_indices[:]
        
        changed = True
        while changed:
            changed = False
            removable = []
            best_score = -1
            best_c = -1
            
            for c in current_code:
                min_count = num_words + 1
                for w in neighbors[c]:
                    if cover_count[w] < min_count:
                        min_count = cover_count[w]
                    if cover_count[w] == 1:
                        min_count = 0 # Not removable
                        break
                
                if min_count > 1:
                    # Removable
                    # Score: higher min_count is better (safer)
                    if min_count > best_score:
                        best_score = min_count
                        best_c = c
                        # If we find a very safe one, we can break? 
                        # No, let's check all to find the best.
                        
            if best_c != -1:
                # Remove best_c
                current_code.remove(best_c)
                in_code[best_c] = False
                for w in neighbors[best_c]:
                    cover_count[w] -= 1
                changed = True
                
                if time.time() - start_time > time_limit - 10:
                    break
                    
        return current_code
