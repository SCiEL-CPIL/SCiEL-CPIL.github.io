"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    colors = [-1] * 128
    
    for pos in range(128):
        if time.time() > deadline:
            break
        x = pos + 1
        
        # Collect all valid colors
        valid_colors = []
        for c in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3*d < 1:
                    break
                c1 = colors[x-1-3*d]
                c2 = colors[x-1-2*d]
                c3 = colors[x-1-d]
                if len({c, c1, c2, c3}) < 3:
                    valid = False
                    break
            if valid:
                valid_colors.append(c)
        
        if not valid_colors:
            # Should not happen; fallback
            colors[pos] = 0
            continue
        
        # Prefer already-used colors
        used_in_colors = set(colors[:pos])
        existing_valid = [c for c in valid_colors if c in used_in_colors]
        
        if existing_valid:
            # Pick the color that appears most often (to concentrate)
            best_color = max(existing_valid, key=lambda c: colors[:pos].count(c))
            colors[pos] = best_color
        else:
            # No existing color works; pick the smallest new color
            new_valid = [c for c in valid_colors if c not in used_in_colors]
            colors[pos] = min(new_valid)
    
    # Fill any remaining
    for i in range(128):
        if colors[i] == -1:
            colors[i] = 0
    
    return colors
# EVOLVE-BLOCK-END
