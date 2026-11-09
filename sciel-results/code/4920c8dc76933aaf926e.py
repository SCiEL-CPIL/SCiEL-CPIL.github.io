<<<<<<< SEARCH
    def try_construct(max_c, seed_val=0):
        random.seed(seed_val)
        colors = [None] * 128
        used = set()
        for x in range(128):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(max_c):
                if check_color(x, c, colors):
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: prefer used colors to minimize count
            # But sometimes we need to pick a new color to avoid future dead ends?
            # The task says "Minimize the number of labels actually used".
            # Greedy "prefer used" is good for minimizing count, but might lead to dead ends.
            # If we hit a dead end, we return None and try higher max_c.
            
            # To be safe, we can randomize the choice among valid colors,
            # but bias towards used colors.
            
            used_valid = [c for c in valid_colors if c in used]
            new_valid = [c for c in valid_colors if c not in used]
            
            if used_valid:
                # Pick randomly from used valid colors? Or smallest?
                # Smallest is deterministic. Random might help escape local minima.
                # Let's pick randomly from used_valid to explore.
                chosen = random.choice(used_valid)
            else:
                # Must pick a new color. Pick smallest to keep labels low?
                # Or random?
                # Since we are minimizing count, we are forced to use a new one.
                # The label value doesn't matter for the count, only distinctness.
                # But "Minimize the number of labels actually used" is the objective.
                # The labels themselves can be anything in 0..127.
                # So picking the smallest new one is fine.
                chosen = new_valid[0]
            
            colors[x] = chosen
            used.add(chosen)
            
        return colors
=======
    def try_construct(max_c, seed_val=0):
        random.seed(seed_val)
        colors = [None] * 128
        used = set()
        for x in range(128):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(max_c):
                if check_color(x, c, colors):
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: prefer used colors to minimize count
            used_valid = [c for c in valid_colors if c in used]
            new_valid = [c for c in valid_colors if c not in used]
            
            if used_valid:
                # Pick randomly from used valid colors to explore
                chosen = random.choice(used_valid)
            else:
                chosen = new_valid[0]
            
            colors[x] = chosen
            used.add(chosen)
            
        return colors

    def try_reduce_colors(colors, max_retries=500):
        """Try to reduce the number of colors by reassigning positions."""
        if time.time() > deadline:
            return colors
        
        best_colors = list(colors)
        best_count = len(set(colors))
        
        for _ in range(max_retries):
            if time.time() > deadline:
                break
            
            # Try to remove one color at a time
            current_colors = list(best_colors)
            color_set = set(current_colors)
            
            # Try removing each color
            for c in sorted(color_set):
                # Try to reassign all positions with color c to other colors
                positions_with_c = [i for i in range(128) if current_colors[i] == c]
                
                if not positions_with_c:
                    continue
                
                # Try to reassign these positions
                success = True
                temp_colors = list(current_colors)
                
                for pos in positions_with_c:
                    # Find a valid color different from c
                    found = False
                    for new_c in range(max_c):
                        if new_c == c:
                            continue
                        if check_color(pos, new_c, temp_colors):
                            temp_colors[pos] = new_c
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                
                if success:
                    new_count = len(set(temp_colors))
                    if new_count < best_count:
                        best_colors = temp_colors
                        best_count = new_count
                        break
        
        return best_colors
>>>>>>> REPLACE

