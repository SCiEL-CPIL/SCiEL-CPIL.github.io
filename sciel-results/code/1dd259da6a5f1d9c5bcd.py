        temp = candidates
        verts = []
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
            
        if not verts:
            return
            
        verts.sort(key=lambda v: popcount(non_conflict_mask[v] & candidates), reverse=True)
        
        # Use this sorted list for bound
        color_bound = greedy_color_bound_sorted(verts, candidates)
        if current_size + color_bound <= best_size:
            return
            
        # ... rest of expand using verts[0]
