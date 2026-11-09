"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _validate(colors):
    """Check that every nonconstant 4-term progression uses at least 3 colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - a - 3) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            # Nonconstant progression requires d > 0
            colors_used = {colors[a], colors[b], colors[c], colors[e]}
            if len(colors_used) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_construct():
    """Greedy construction: assign each position the smallest color that satisfies constraints."""
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check all 4-term progressions ending at position x (index x-1)
            # A 4-term progression a, a+d, a+2d, a+3d where a+3d = x-1
            # So a = x-1-3d, and we need a >= 0
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                a = x - 1 - 3 * d
                if a < 0:
                    continue
                # The four positions are a, a+d, a+2d, a+3d = x-1
                c0 = colors[a]
                c1 = colors[a + d]
                c2 = colors[a + 2 * d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            return None
    return colors

def _local_search(colors, max_iters=50000, deadline=None):
    """Local search: try to reduce the number of colors used by recoloring positions."""
    n = len(colors)
    current_colors = list(colors)
    current_num_colors = _count_colors(current_colors)
    
    if deadline is None:
        deadline = time.time() + 50
    
    best_colors = list(current_colors)
    best_num_colors = current_num_colors
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random position and try to recolor it to a color already in use
        # that is not the most common color, to potentially reduce color count
        pos = random.randint(0, n - 1)
        old_color = current_colors[pos]
        
        # Get all colors currently in use
        in_use = set(current_colors)
        # Try colors that are in use but different from old_color
        # Prefer colors that appear less frequently
        color_counts = {}
        for c in current_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Sort colors by frequency, try less frequent ones first
        sorted_colors = sorted(in_use, key=lambda c: color_counts[c])
        
        improved = False
        for new_color in sorted_colors:
            if new_color == old_color:
                continue
            
            # Check if recoloring pos to new_color is valid
            current_colors[pos] = new_color
            if _validate(current_colors):
                # Check if this reduces the number of colors
                new_num_colors = _count_colors(current_colors)
                if new_num_colors < best_num_colors:
                    best_num_colors = new_num_colors
                    best_colors = list(current_colors)
                    improved = True
                else:
                    # Accept if same number of colors (for diversity)
                    pass
                break
            else:
                current_colors[pos] = old_color
        
        if not improved:
            # Try random recoloring to explore
            if random.random() < 0.1:
                new_color = random.choice(sorted_colors)
                current_colors[pos] = new_color
                if _validate(current_colors):
                    pass
                else:
                    current_colors[pos] = old_color
    
    return best_colors, best_num_colors

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    # Start with greedy construction
    colors = _greedy_construct()
    if colors is None:
        # Fallback: should not happen, but return something
        return list(range(128))
    
    best_colors = list(colors)
    best_num_colors = _count_colors(best_colors)
    
    # Try local search to reduce number of colors
    if time.time() < deadline:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Perturb the solution
            perturbed = list(best_colors)
            num_perturb = random.randint(1, 5)
            for _ in range(num_perturb):
                pos = random.randint(0, 127)
                new_color = random.randint(0, 127)
                perturbed[pos] = new_color
            
            if _validate(perturbed):
                result, num_colors = _local_search(perturbed, max_iters=20000, deadline=deadline - 5)
                if num_colors < best_num_colors:
                    best_colors = result
                    best_num_colors = num_colors
    
    return best_colors
# EVOLVE-BLOCK-END
