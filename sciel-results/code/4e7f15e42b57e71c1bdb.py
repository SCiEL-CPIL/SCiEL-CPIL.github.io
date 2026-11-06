        old_count = current_count
        # ... update hist and colors ...
        
        if valid:
            if current_count < old_count:
                if random.random() > 0.1:
                    # Revert
                    current_colors[e_idx] = old_c
                    hist[new_c] -= 1
                    if hist[new_c] == 0:
                        current_count -= 1
                    hist[old_c] += 1
                    if hist[old_c] == 1:
                        current_count += 1
            if current_count > best_count:
                best_count = current_count
                best_colors = current_colors[:]
        else:
            # Revert
            current_colors[e_idx] = old_c
            hist[new_c] -= 1
            if hist[new_c] == 0:
                current_count -= 1
            hist[old_c] += 1
            if hist[old_c] == 1:
                current_count += 1
