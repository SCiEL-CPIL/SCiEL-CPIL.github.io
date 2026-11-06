def construct():
    n = 128
    
    # Try increasing palette sizes
    for K in range(3, 10):
        palette = list(range(K))
        colors = [-1] * (n + 1)
        used_colors = set()
        success = True
        
        for x in range(1, n + 1):
            all_allowed_mask = (1 << K) - 1
            valid = True
            
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                
                triple_set = {c1, c2, c3}
                
                if len(triple_set) == 1:
                    valid = False
                    break
                elif len(triple_set) == 2:
                    mask = 0
                    for c in palette:
                        if c not in triple_set:
                            mask |= (1 << c)
                    all_allowed_mask &= mask
                else:
                    pass
            
            if not valid:
                break
            
            if all_allowed_mask == 0:
                valid = False
                break
            
            valid_colors = []
            for c in palette:
                if all_allowed_mask & (1 << c):
                    valid_colors.append(c)
            
            if not valid_colors:
                valid = False
                break
            
            # Prioritize colors already in used_colors
            used_valid = [c for c in valid_colors if c in used_colors]
            if used_valid:
                c_choice = min(used_valid)
            else:
                c_choice = min(valid_colors)
            
            colors[x] = c_choice
            used_colors.add(c_choice)
        
        if success:
            return colors[1:]
    
    raise Exception("Failed to construct coloring")
