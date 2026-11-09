"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 128
    colors = [-1] * n
    
    # Precompute all 4-term progressions (a, b, c, d) with a < b < c < d, b-a = c-b = d-c
    # We need every non-constant 4-term progression to use at least 3 distinct colors.
    # A 4-term progression is (x, x+d, x+2d, x+3d) for x >= 1, d >= 1, x+3d <= 128.
    
    # For each position i (0-indexed, so position i+1), find all 4-term progressions that include it.
    # We'll use a greedy approach with look-ahead.
    
    def get_progressions_containing(i):
        """Get all 4-term progressions (as tuples of 0-indexed positions) that include position i."""
        progs = []
        # i can be at position 0, 1, 2, or 3 in the progression
        # Position 0: i = x, so x = i, need i+3d <= 127 (0-indexed), d >= 1
        #   i+3d <= 127 => d <= (127-i)//3
        for d in range(1, (127 - i) // 3 + 1):
            if i + 3 * d <= 127:
                progs.append((i, i+d, i+2*d, i+3*d))
        # Position 1: i = x+d, so x = i-d, need x >= 0, i+2d <= 127
        #   i-d >= 0 => d <= i
        #   i+2d <= 127 => d <= (127-i)//2
        for d in range(1, min(i, (127 - i) // 2) + 1):
            if i - d >= 0 and i + 2 * d <= 127:
                progs.append((i-d, i, i+d, i+2*d))
        # Position 2: i = x+2d, so x = i-2d, need x >= 0, i+d <= 127
        #   i-2d >= 0 => d <= i//2
        #   i+d <= 127 => d <= 127-i
        for d in range(1, min(i // 2, 127 - i) + 1):
            if i - 2 * d >= 0 and i + d <= 127:
                progs.append((i-2*d, i-d, i, i+d))
        # Position 3: i = x+3d, so x = i-3d, need x >= 0
        #   i-3d >= 0 => d <= i//3
        for d in range(1, i // 3 + 1):
            if i - 3 * d >= 0:
                progs.append((i-3*d, i-2*d, i-d, i))
        return progs
    
    # Precompute progressions for each position
    all_progs = [get_progressions_containing(i) for i in range(n)]
    
    # Greedy with look-ahead: assign smallest color that doesn't violate constraints
    # and doesn't lead to a dead-end in the next k steps
    
    def is_valid_assignment(colors, i, c):
        """Check if assigning color c to position i is valid given current colors."""
        for prog in all_progs[i]:
            if all(colors[j] != -1 for j in prog):
                # All four positions are assigned; check if we have < 3 distinct colors
                distinct = len({colors[j] for j in prog})
                if distinct < 3:
                    return False
        return True
    
    def count_future_conflicts(colors, i, c, k=4):
        """Count how many future positions (i+1 to i+k) would become problematic."""
        # This is a heuristic: check if assigning c to i creates situations
        # where future positions have very few valid colors
        temp_colors = colors[:]
        temp_colors[i] = c
        
        # For each future position j in i+1..i+k, count valid colors
        conflict_score = 0
        for j in range(i + 1, min(i + k + 1, n)):
            valid_colors = 0
            for cc in range(128):
                if is_valid_assignment(temp_colors, j, cc):
                    valid_colors += 1
                    if valid_colors > 5:  # Early exit: plenty of options
                        break
            # If very few valid colors, that's a potential problem
            if valid_colors <= 2:
                conflict_score += (3 - valid_colors)
        return conflict_score
    
    for i in range(n):
        best_color = None
        best_conflict = float('inf')
        
        for c in range(128):
            if not is_valid_assignment(colors, i, c):
                continue
            # Look-ahead: check future conflicts
            conflict = count_future_conflicts(colors, i, c, k=4)
            if conflict < best_conflict:
                best_conflict = conflict
                best_color = c
            elif conflict == best_conflict and best_color is not None and c < best_color:
                best_color = c
        
        if best_color is None:
            # Fallback: just pick the smallest valid color
            for c in range(128):
                if is_valid_assignment(colors, i, c):
                    best_color = c
                    break
            if best_color is None:
                # Should not happen, but just in case
                best_color = 0
        
        colors[i] = best_color
    
    return colors

# EVOLVE-BLOCK-END
