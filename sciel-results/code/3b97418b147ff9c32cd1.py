"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _validate(colors, n=128):
    """Check that every nonconstant 4-term progression uses >= 3 distinct colors."""
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j, k, l = i + d, i + 2*d, i + 3*d
            if l >= n:
                break
            c = {colors[i], colors[j], colors[k], colors[l]}
            if len(c) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_construct(palette_size, n=128, seed=0):
    """Greedy construction with a limited palette."""
    rng = random.Random(seed)
    palette = list(range(palette_size))
    colors = [None] * n
    
    for i in range(n):
        valid_colors = set(palette)
        # Check constraints with previous positions
        for d in range(1, (n - 1 - i) // 3 + 1):
            # This position is the end of a 4-term AP: i, i-d, i-2d, i-3d
            if i - 3*d < 0:
                break
            j, k, l = i - d, i - 2*d, i - 3*d
            if colors[j] is None or colors[k] is None or colors[l] is None:
                continue
            # We need at least 3 distinct colors in {colors[i], colors[j], colors[k], colors[l]}
            existing = {colors[j], colors[k], colors[l]}
            if len(existing) >= 3:
                # Any color works, but we want to minimize new colors
                pass
            elif len(existing) == 2:
                # We need colors[i] to be different from at least one of the two
                # Actually, we need the set {colors[i]} ∪ existing to have size >= 3
                # So colors[i] must NOT be in existing
                valid_colors = valid_colors - existing
            elif len(existing) == 1:
                # We need colors[i] such that {colors[i]} ∪ existing has size >= 3
                # So colors[i] must be different from the one existing color
                # Actually, if existing has 1 color, we need colors[i] to be different
                # to get size 2, but we need size 3. So colors[i] must be different
                # and we need another distinct color. But we only have one position.
                # Wait, the set is {colors[i], colors[j], colors[k], colors[l]}.
                # If existing has 1 color, say {c}, then we need {colors[i], c} to have size >= 3?
                # No, the set is {colors[i]} ∪ {c} = {colors[i], c} if colors[i] != c, size 2.
                # That's not >= 3. So we need colors[i] to be such that the total set has size >= 3.
                # But we only have one new element. So we can't get size 3 from size 1 + 1 new element.
                # This means the constraint is violated regardless? No, wait.
                # Let's re-read: "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
                # The progression is (a, a+d, a+2d, a+3d). The colors are (c_a, c_{a+d}, c_{a+2d}, c_{a+3d}).
                # We need |{c_a, c_{a+d}, c_{a+2d}, c_{a+3d}}| >= 3.
                # If we're assigning c_{a+3d}, and the other three are all the same color c,
                # then we need |{c_{a+3d}, c}| >= 3, which is impossible.
                # So if the previous three are all the same, we're stuck? 
                # No, that would mean no valid coloring exists, but the seed program works.
                # Ah, I see. The constraint is on the set of colors in the progression.
                # If previous three are {c, c, c}, then we need the new color to make the set size >= 3.
                # But {new, c} has size at most 2. So it's impossible.
                # This suggests that a valid coloring requires that no three consecutive terms in any AP
                # are all the same color. But that's not quite right either.
                # Let's just check: if existing has 1 color, we can't satisfy the constraint.
                # So valid_colors becomes empty, and we fail.
                valid_colors = set()  # Impossible to satisfy
            else:
                # len(existing) == 0, shouldn't happen since j,k,l < i and are assigned
                pass
        
        # Also need to check where this position is the start of a 4-term AP
        # i, i+d, i+2d, i+3d - but those are in the future, so no constraint yet.
        # And where this position is the 2nd or 3rd term:
        # For 2nd term: i-d, i, i+d, i+2d - future terms not assigned
        # For 3rd term: i-2d, i-d, i, i+d - future term not assigned
        # So only the constraint where this is the last term matters for greedy.
        
        if not valid_colors:
            return None
        
        # Choose a color, preferring ones already used
        used_colors = set(colors[:i])
        preferred = [c for c in valid_colors if c in used_colors]
        if preferred:
            colors[i] = rng.choice(preferred)
        else:
            colors[i] = rng.choice(list(valid_colors))
    
    return colors

def _local_search(colors, n=128, max_iter=1000, seed=0):
    """Local search to reduce the number of colors."""
    rng = random.Random(seed)
    current = colors[:]
    current_colors = set(current)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            old_color = current[i]
            # Try removing this color if it's not used elsewhere
            if current_colors.count(old_color) == 1:
                # Try changing to another existing color
                for new_color in current_colors - {old_color}:
                    candidate = current[:]
                    candidate[i] = new_color
                    if _validate(candidate, n):
                        current = candidate
                        current_colors = set(current)
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Try swapping two colors
            color_list = list(current_colors)
            if len(color_list) >= 2:
                for _ in range(10):  # Random swaps
                    c1, c2 = rng.sample(color_list, 2)
                    candidate = [c2 if c == c1 else c1 if c == c2 else c for c in current]
                    if _validate(candidate, n):
                        current = candidate
                        current_colors = set(current)
                        improved = True
                        break
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = None
    best_count = 128
    
    # Try different palette sizes and seeds
    for palette_size in [3, 4, 5, 6, 7, 8]:
        for seed in range(20):
            if time.time() > deadline:
                break
            
            colors = _greedy_construct(palette_size, 128, seed)
            if colors is None:
                continue
            
            if not _validate(colors, 128):
                continue
            
            count = _count_colors(colors)
            if count < best_count:
                best_count = count
                best = colors[:]
            
            # Try local search to improve
            if time.time() > deadline - 10:
                break
            
            improved = _local_search(colors, 128, 500, seed)
            if improved:
                imp_count = _count_colors(improved)
                if imp_count < best_count:
                    best_count = imp_count
                    best = improved[:]
        
        if time.time() > deadline:
            break
    
    if best is None:
        # Fallback: use the seed algorithm
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1, 2, 3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1))
                    and all(any(colors[x-1-j*d] != color for j in (1, 2)) 
                            for d in range(1, (x-1)//2 + 1))):
                    colors.append(color)
                    break
        return colors
    
    return best

# EVOLVE-BLOCK-END
