import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (0-indexed)
    progressions = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j3 = i + 3 * d
            if j3 < n:
                progressions.append((i, i + d, i + 2 * d, j3))
    
    num_progressions = len(progressions)
    
    # Build position -> progression index mapping
    pos_progressions = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        pos_progressions[a].append(idx)
        pos_progressions[b].append(idx)
        pos_progressions[c].append(idx)
        pos_progressions[d].append(idx)
    
    def validate(arr):
        """Check if every nonconstant 4-term progression uses at least 3 distinct colors."""
        for (a, b, c, d) in progressions:
            colors = {arr[a], arr[b], arr[c], arr[d]}
            if len(colors) < 3:
                return False
        return True
    
    def count_colors(arr):
        return len(set(arr))
    
    def is_valid_assignment(arr, pos, color):
        """Check if assigning 'color' to position 'pos' keeps all progressions valid."""
        for pidx in pos_progressions[pos]:
            a, b, c, d = progressions[pidx]
            # Get colors of all 4 positions, with 'color' for pos
            ca = color if a == pos else arr[a]
            cb = color if b == pos else arr[b]
            cc = color if c == pos else arr[c]
            cd = color if d == pos else arr[d]
            if len({ca, cb, cc, cd}) < 3:
                return False
        return True
    
    def greedy_construct(max_colors=128):
        """Build a valid coloring greedily."""
        colors = [-1] * n
        
        for x in range(n):
            # Try colors in order of least use first
            color_counts = {}
            for c in range(max_colors):
                color_counts[c] = sum(1 for i in range(x) if colors[i] == c)
            
            sorted_colors = sorted(range(max_colors), key=lambda c: color_counts[c])
            
            for color in sorted_colors:
                if is_valid_assignment(colors, x, color):
                    colors[x] = color
                    break
            
            if colors[x] == -1:
                # Fallback: use a new color
                colors[x] = max_colors
                max_colors += 1
        
        return colors
    
    def local_search_eliminate(arr, max_iterations=50000, deadline=deadline):
        """Try to eliminate colors one at a time via local search."""
        arr = list(arr)
        current_colors = count_colors(arr)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            used_colors = sorted(set(arr))
            
            for color_to_remove in used_colors:
                if time.time() > deadline - 3:
                    break
                
                positions_with_color = [i for i in range(n) if arr[i] == color_to_remove]
                if not positions_with_color:
                    continue
                
                # Try to reassign each position with this color to another existing color
                trial = list(arr)
                success = True
                
                for pos in positions_with_color:
                    found_valid = False
                    # Try other colors, preferring less frequent ones
                    other_colors = [c for c in used_colors if c != color_to_remove]
                    # Sort by frequency (ascending) to spread out
                    freq = {c: sum(1 for i in range(n) if arr[i] == c) for c in other_colors}
                    other_colors.sort(key=lambda c: freq[c])
                    
                    for new_color in other_colors:
                        trial[pos] = new_color
                        if is_valid_assignment(trial, pos, new_color):
                            found_valid = True
                            break
                    
                    if not found_valid:
                        success = False
                        break
                
                if success and validate(trial):
                    arr = trial
                    current_colors -= 1
                    improved = True
                    break
            
            if not improved:
                break
        
        return arr, current_colors
    
    def random_repair(trial, max_iter=10000, deadline=deadline):
        """Try to fix a random assignment into a valid one."""
        trial = list(trial)
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Find first invalid progression
            invalid_idx = -1
            for idx, (a, b, c, d) in enumerate(progressions):
                if len({trial[a], trial[b], trial[c], trial[d]}) < 3:
                    invalid_idx = idx
                    break
            
            if invalid_idx == -1:
                return trial, True
            
            a, b, c, d = progressions[invalid_idx]
            # Pick a random position in this progression to change
            pos = random.choice([a, b, c, d])
            # Try random colors
            for _ in range(20):
                new_color = random.randint(0, 7)
                if is_valid_assignment(trial, pos, new_color):
                    trial[pos] = new_color
                    break
        
        return trial, validate(trial)
    
    # Phase 1: Greedy construction
    best_arr = greedy_construct()
    best_count = count_colors(best_arr)
    
    if not validate(best_arr):
        # Shouldn't happen, but fallback
        best_arr = list(range(n))
        best_count = n
    
    # Phase 2: Local search to eliminate colors
    if time.time() < deadline - 10:
        improved_arr, improved_count = local_search_eliminate(best_arr, max_iterations=30000)
        if improved_count < best_count:
            best_arr = improved_arr
            best_count = improved_count
    
    # Phase 3: Random restarts with repair
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random starting point with few colors
        num_colors = random.randint(3, max(3, best_count - 1))
        trial = [random.randint(0, num_colors - 1) for _ in range(n)]
        
        # Try to repair
        repaired, valid = random_repair(trial, max_iter=5000, deadline=deadline)
        
        if valid:
            cnt = count_colors(repaired)
            if cnt < best_count:
                best_arr = repaired
                best_count = cnt
                # Try to eliminate more colors
                if time.time() < deadline - 5:
                    improved_arr, improved_count = local_search_eliminate(best_arr, max_iterations=10000)
                    if improved_count < best_count:
                        best_arr = improved_arr
                        best_count = improved_count
    
    # Phase 4: Final local search on best
    if time.time() < deadline - 3:
        improved_arr, improved_count = local_search_eliminate(best_arr, max_iterations=20000)
        if improved_count < best_count:
            best_arr = improved_arr
            best_count = improved_count
    
    # Ensure validity
    if not validate(best_arr):
        # Last resort fallback
        best_arr = list(range(n))
        best_count = n
    
    return best_arr

