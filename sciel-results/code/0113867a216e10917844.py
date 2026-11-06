import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Seed solution
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    best_n = n_seed
    best_colors = colors_seed
    
    # Try to find larger n
    for target_n in range(10, 17):
        if time.time() > deadline - 10:
            break
            
        # Backtracking search for target_n
        # ... (implementation from above) ...
        
        # If found, update best
        # ...
        
    return {'n': best_n, 'colors': best_colors}
