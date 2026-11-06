<<<<<<< SEARCH
import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Fallback: n=11 construction
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb
    
    # Try to find valid colorings for larger n
    # Prioritize 27 (new result), then 26 (known Lazebnik-Woldar), then others
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        # Allocate time budget
        if n == 27:
            t_limit = 25
        elif n == 26:
            t_limit = 35
        elif n == 25:
            t_limit = 15
        elif n <= 23:
            t_limit = 10
        else:
            t_limit = 5
            
        actual_limit = min(deadline - time.time() - 3, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n that works
            
    return {'n': best_n, 'colors': best_colors}
=======
import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Fallback: n=11 construction
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb
    
    # Try to find valid colorings for larger n
    # Prioritize 27 (new result), then 26 (known Lazebnik-Woldar), then others
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        # Allocate time budget
        if n == 27:
            t_limit = 25
        elif n == 26:
            t_limit = 35
        elif n == 25:
            t_limit = 15
        elif n <= 23:
            t_limit = 10
        else:
            t_limit = 5
            
        actual_limit = min(deadline - time.time() - 3, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n that works
            
    return {'n': best_n, 'colors': best_colors}
>>>>>>> REPLACE

