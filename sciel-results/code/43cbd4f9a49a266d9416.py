"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (i, i+d, i+2d, i+3d) within 0..n-1
    progressions = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            progressions.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_progs = len(progressions)
    
    # For each position, store indices of progressions it's in and the other 3 positions
    pos_progs = [[] for _ in range(n)]
    for pi, (a, b, c, e) in enumerate(progressions):
        for pos in (a, b, c, e):
            others = [p for p in (a, b, c, e) if p != pos]
            pos_progs[pos].append((pi, others))
    
    def count_colors(arr):
        return len(set(arr))
    
    def validate_full(arr):
        """Full validation - check all progressions."""
        for (a, b, c, e) in progressions:
            c0, c1, c2, c3 = arr[a], arr[b], arr[c], arr[e]
            # Fast path: if any two are equal, need third distinct
            if c0 == c1:
                if c2 == c0 and c3 == c0:
                    return False
            elif c0 == c2:
                if c1 == c0 and c3 == c0:
                    return False
            elif c0 == c3:
                if c1 == c0 and c2 == c0:
                    return False
            elif c1 == c2:
                if c0 == c1 and c3 == c1:
                    return False
            elif c1 == c3:
                if c0 == c1 and c2 == c1:
                    return False
            elif c2 == c3:
                if c0 == c2 and c1 == c2:
                    return False
            # All four different or three with one pair - check count
            # Actually simpler: just count distinct
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True
    
    def validate_change(arr, pos, old_color, new_color):
        """Check if changing arr[pos] from old_color to new_color maintains validity.
        Only checks progressions containing pos."""
        for (pi, others) in pos_progs[pos]:
            c0, c1, c2 = arr[others[0]], arr[others[1]], arr[others[2]]
            # Check if {new_color, c0, c1, c2} has at least 3 distinct values
            if new_color == c0:
                if new_color == c1 and new_color == c2:
                    return False
            elif new_color == c1:
                if new_color == c0 and new_color == c2:
                    return False
            elif new_color == c2:
                if new_color == c0 and new_color == c1:
                    return False
            else:
                # new_color is different from all three
                # Need at least one of c0,c1,c2 to differ from new_color (always true)
                # and at least 3 distinct total - since new_color is new, need c0,c1,c2 not all same
                if c0 == c1 and c1 == c2:
                    return False
        return True
    
    def greedy_construct():
        """Fast greedy construction of a valid coloring."""
        colors = [0] * n
        for pos in range(n):
            found = False
            for color in range(n):
                valid = True
                # Check all progressions containing pos where all other positions are assigned
                for (pi, others) in pos_progs[pos]:
                    # Check if all other positions are assigned
                    if all(o < pos for o in others):
                        c0, c1, c2 = colors[others[0]], colors[others[1]], colors[others[2]]
                        if len({color, c0, c1, c2}) < 3:
                            valid = False
                            break
                    elif all(o >= pos for o in others):
                        # All future - no constraint yet
                        pass
                    else:
                        # Mixed - some assigned, some not
                        # Only check if the assigned ones plus new_color already violate
                        assigned_colors = set()
                        assigned_colors.add(color)
                        for o in others:
                            if o < pos:
                                assigned_colors.add(colors[o])
                        if len(assigned_colors) < 2:
                            # At most 1 color among assigned+new, but we need >=3 total
                            # This is only a problem if it's impossible to add more colors
                            # For now, be conservative
                            if len(assigned_colors) == 1 and len(others) == 3:
                                # All three others are unassigned, we can still add colors
                                pass
                if valid:
                    colors[pos] = color
                    found = True
                    break
            if not found:
                colors[pos] = 0
        return colors
    
    # Start with a simple valid coloring: use 4 colors in a pattern
    # Pattern: color = (pos % 4) ensures any 4-term AP has at least 3 colors
    # Actually, let's use a smarter initial pattern
    colors = [0] * n
    for pos in range(n):
        # Use 4 colors: 0,1,2,3 in a pattern that avoids monochromatic or 2-color APs
        # Simple: color = pos % 4
        colors[pos] = pos % 4
    
    # Verify it's valid
    if not validate_full(colors):
        # Fallback: use even more colors
        colors = list(range(n))
    
    best_colors = colors[:]
    best_count = count_colors(colors)
    
    # Try greedy to see if we can do better
    if time.time() < deadline - 20:
        greedy_colors = greedy_construct()
        if validate_full(greedy_colors):
            greedy_count = count_colors(greedy_colors)
            if greedy_count < best_count:
                best_colors = greedy_colors[:]
                best_count = greedy_count
    
    # Local search to reduce color count
    def local_search(arr, max_iterations=200000):
        """Improve coloring by changing individual positions to reduce color count."""
        arr = list(arr)
        n_local = len(arr)
        current_colors = count_colors(arr)
        
        # Build color frequency
        color_freq = {}
        for c in arr:
            color_freq[c] = color_freq.get(c, 0) + 1
        
        used_colors = list(color_freq.keys())
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try to reduce color count by reassigning positions
            # Strategy: for each color, try to reassign all its positions to other colors
            # If successful, we reduce the color count by 1
            
            # First, try simple single-position changes that might enable color reduction
            for pos in range(n_local):
                if time.time() > deadline - 3:
                    break
                
                current_color = arr[pos]
                
                # Try reassigning to each other used color
                for new_color in used_colors:
                    if new_color == current_color:
                        continue
                    
                    if validate_change(arr, pos, current_color, new_color):
                        # Check if this reduces color count
                        old_freq = color_freq[current_color]
                        if old_freq == 1:
                            # This is the last position with this color - removing it reduces count
                            arr[pos] = new_color
                            color_freq[current_color] -= 1
                            del color_freq[current_color]
                            color_freq[new_color] = color_freq.get(new_color, 0) + 1
                            used_colors.remove(current_color)
                            current_colors -= 1
                            improved = True
                            break
                        else:
                            # Just a reassignment, doesn't reduce count directly
                            # But might enable future reductions - do it anyway if it doesn't hurt
                            arr[pos] = new_color
                            color_freq[current_color] -= 1
                            color_freq[new_color] = color_freq.get(new_color, 0) + 1
                            if color_freq[current_color] == 0:
                                del color_freq[current_color]
                                used_colors.remove(current_color)
                                current_colors -= 1
                                improved = True
                                break
                            else:
                                # Revert - this doesn't help immediately
                                arr[pos] = current_color
                                color_freq[current_color] += 1
                                color_freq[new_color] -= 1
                                if color_freq[new_color] == 0:
                                    del color_freq[new_color]
                                    used_colors.append(new_color)
                
                if improved:
                    break
            
            if not improved:
                # Try more aggressive: pick a color with fewest occurrences and try to eliminate it
                if len(used_colors) > 1:
                    # Find color with minimum frequency
                    min_color = min(used_colors, key=lambda c: color_freq[c])
                    min_freq = color_freq[min_color]
                    
                    # Try to reassign all positions with min_color to other colors
                    positions_with_min = [p for p in range(n_local) if arr[p] == min_color]
                    
                    # Try different orderings
                    random.shuffle(positions_with_min)
                    
                    success = True
                    new_arr = list(arr)
                    new_color_freq = dict(color_freq)
                    new_used = list(used_colors)
                    
                    for pos in positions_with_min:
                        placed = False
                        for new_color in new_used:
                            if new_color == min_color:
                                continue
                            if validate_change(new_arr, pos, min_color, new_color):
                                new_arr[pos] = new_color
                                new_color_freq[min_color] -= 1
                                new_color_freq[new_color] = new_color_freq.get(new_color, 0) + 1
                                placed = True
                                break
                        if not placed:
                            success = False
                            break
                    
                    if success and new_color_freq[min_color] == 0:
                        arr = new_arr
                        color_freq = new_color_freq
                        used_colors = new_used
                        used_colors.remove(min_color)
                        current_colors -= 1
                        improved = True
            
            if not improved:
                break
        
        return arr, current_colors
    
    # Run local search with multiple restarts
    random.seed(42)
    
    # First, try to improve the current best
    if time.time() < deadline - 15:
        improved_arr, improved_count = local_search(best_colors, max_iterations=50000)
        if validate_full(improved_arr) and improved_count < best_count:
            best_colors = improved_arr
            best_count = improved_count
    
    # Try random restarts with local search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed version of the best solution
        trial = list(best_colors)
        
        # Randomly perturb some positions
        num_perturb = random.randint(3, 10)
        for _ in range(num_perturb):
            pos = random.randint(0, n - 1)
            trial[pos] = random.randint(0, best_count)
        
        # Try to fix and improve
        trial_arr, trial_count = local_search(trial, max_iterations=30000)
        
        if validate_full(trial_arr) and trial_count < best_count:
            best_colors = trial_arr
            best_count = trial_count
    
    # Final validation
    if not validate_full(best_colors):
        # Should not happen, but fallback
        best_colors = list(range(n))
    
    return best_colors

# EVOLVE-BLOCK-END
