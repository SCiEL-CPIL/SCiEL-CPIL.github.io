    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_i_idx = -1 # index in snapshot
        best_c = -1
        
        snapshot = family[:]
        
        for idx, s_i in enumerate(snapshot):
            # s_i should be in family
            if not in_family[s_i]: continue # safety
            
            remove_mask(s_i)
            
            # Find best candidate
            for c in range(1 << N):
                if not in_family[c]:
                    gain = cnt_sup[c] + cnt_sub[c]
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                        best_i_idx = idx # We need to know which s_i to replace
                        # Actually, we just need to replace s_i with best_c
                        # So we store s_i
                        best_s_i = s_i
                        
            add_mask(s_i)
            
        if best_gain > 0:
            # Replace best_s_i with best_c
            remove_mask(best_s_i)
            add_mask(best_c)
            improved = True
