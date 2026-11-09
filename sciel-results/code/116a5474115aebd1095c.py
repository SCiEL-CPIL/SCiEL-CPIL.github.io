"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

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
    
    def local_search(arr, max_iterations=50000):
        """Improve coloring by changing individual positions to reduce color count."""
        arr = list(arr)
        n = len(arr)
        current_colors = count_colors(arr)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try changing each position to a color already in use (to potentially reduce total colors)
            used_colors = list(set(arr))
            
            for pos in range(n):
                if time.time() > deadline - 2:
                    break
                
                for new_color in used_colors:
                    if new_color == arr[pos]:
                        continue
                    
                    # Check if this change is valid
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
    
    # Start with the greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    
    best_colors = colors
    best_count = count_colors(colors)
    
    # Try local search to reduce color count
    if time.time() < deadline - 10:
        improved_arr, improved_count = local_search(colors, max_iterations=100000)
        if improved_count < best_count:
            best_colors = improved_arr
            best_count = improved_count
    
    # Try multiple random restarts with local search
    if time.time() < deadline - 5:
        random.seed(42)
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Create a random valid starting point by modifying the greedy solution
            trial = list(colors)
            
            # Randomly perturb some positions
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                pos = random.randint(0, 127)
                trial[pos] = random.randint(0, 127)
            
            # Try to fix invalidities by local search
            trial_arr, trial_count = local_search(trial, max_iterations=50000)
            
            if trial_count < best_count:
                best_colors = trial_arr
                best_count = trial_count
    
    return best_colors
# EVOLVE-BLOCK-END
