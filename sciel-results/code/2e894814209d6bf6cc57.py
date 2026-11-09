"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    def count_colors(arr):
        return len(set(arr))
    
    def validate(arr):
        """Check if every nonconstant 4-term progression uses at least 3 distinct colors."""
        n = len(arr)
        for i in range(n):
            for d in range(1, (n - 1 - i) // 3 + 1):
                j1 = i + d
                j2 = i + 2*d
                j3 = i + 3*d
                if j3 < n:
                    colors_in_prog = {arr[i], arr[j1], arr[j2], arr[j3]}
                    if len(colors_in_prog) < 3:
                        return False
        return True
    
    def validate_periodic_pattern(pattern, period, n=128):
        """Check if a periodic pattern satisfies the 3-color constraint for all positions 1..n."""
        arr = [pattern[i % period] for i in range(n)]
        return validate(arr)
    
    def search_periodic_patterns(max_period=12, max_colors=3):
        """Exhaustively search over small-period patterns with few colors."""
        best_arr = None
        best_count = max_colors + 1
        
        # Try 2 colors first
        for period in range(2, max_period + 1):
            if time.time() > deadline - 10:
                break
            
            # All patterns with 2 colors (0 and 1)
            for pattern in product([0, 1], repeat=period):
                if time.time() > deadline - 10:
                    break
                
                # Quick check: pattern must not be constant
                if len(set(pattern)) < 2:
                    continue
                
                if validate_periodic_pattern(pattern, period):
                    arr = [pattern[i % period] for i in range(128)]
                    c = count_colors(arr)
                    if c < best_count:
                        best_count = c
                        best_arr = arr
                        if best_count == 2:
                            return best_arr, best_count
        
        # Try 3 colors
        for period in range(2, max_period + 1):
            if time.time() > deadline - 5:
                break
            
            # All patterns with 3 colors (0, 1, 2)
            for pattern in product([0, 1, 2], repeat=period):
                if time.time() > deadline - 5:
                    break
                
                # Must use at least 2 distinct colors in the pattern
                if len(set(pattern)) < 2:
                    continue
                
                if validate_periodic_pattern(pattern, period):
                    arr = [pattern[i % period] for i in range(128)]
                    c = count_colors(arr)
                    if c < best_count:
                        best_count = c
                        best_arr = arr
                        if best_count == 2:
                            return best_arr, best_count
        
        return best_arr, best_count
    
    def local_search(arr, max_iterations=50000):
        """Improve coloring by changing individual positions to reduce color count."""
        arr = list(arr)
        n = len(arr)
        current_colors = count_colors(arr)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            improved = False
            used_colors = list(set(arr))
            
            for pos in range(n):
                if time.time() > deadline - 2:
                    break
                
                for new_color in used_colors:
                    if new_color == arr[pos]:
                        continue
                    
                    arr[pos] = new_color
                    if validate(arr):
                        new_colors = count_colors(arr)
                        if new_colors < current_colors:
                            current_colors = new_colors
                            improved = True
                            break
                        else:
                            arr[pos] = arr[pos]  # revert
                    else:
                        arr[pos] = arr[pos]  # revert
                
                if improved:
                    break
            
            if not improved:
                break
        
        return arr, current_colors
    
    # Step 1: Search for periodic patterns
    best_colors, best_count = search_periodic_patterns(max_period=12, max_colors=3)
    
    # If no periodic pattern found, fall back to greedy construction
    if best_colors is None:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color)
                    break
        best_colors = colors
        best_count = count_colors(colors)
    
    # Step 2: Try local search to further reduce color count
    if time.time() < deadline - 10 and best_count > 2:
        improved_arr, improved_count = local_search(best_colors, max_iterations=100000)
        if improved_count < best_count:
            best_colors = improved_arr
            best_count = improved_count
    
    # Step 3: Try random restarts with local search
    if time.time() < deadline - 5 and best_count > 2:
        random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            trial = list(best_colors)
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                pos = random.randint(0, 127)
                trial[pos] = random.randint(0, 127)
            
            trial_arr, trial_count = local_search(trial, max_iterations=50000)
            
            if trial_count < best_count:
                best_colors = trial_arr
                best_count = trial_count
    
    return best_colors
# EVOLVE-BLOCK-END
