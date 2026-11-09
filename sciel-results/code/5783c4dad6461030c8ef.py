import time
import random
import numpy as np

def _validate(colors):
    """Check that every nonconstant 4-term progression uses at least 3 colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            colors_set = {colors[a], colors[b], colors[c], colors[e]}
            if len(colors_set) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 128
    best_colors = None
    best_count = 128
    
    # Precompute all 4-term progressions
    progressions = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                progressions.append((a, b, c, e))
    
    num_progs = len(progressions)
    
    # Try to find solutions with k colors for k = 2, 3, 4, ...
    # k=2 is impossible (any 4-term progression with 2 colors has at most 2 distinct)
    # So we start with k=3
    
    for k in range(3, 128):
        if time.time() > deadline:
            break
        
        # Multiple restarts for each k
        max_restarts = 50 if k <= 4 else 20
        restart_time_limit = (deadline - time.time()) / max_restarts
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            restart_start = time.time()
            restart_deadline = min(deadline, restart_start + max(1.0, restart_time_limit))
            
            # Initialize with random assignment of k colors
            colors = np.random.randint(0, k, size=n)
            
            # Local search: try to fix violations
            improved = True
            no_improve_count = 0
            
            while improved and time.time() < restart_deadline:
                improved = False
                no_improve_count += 1
                if no_improve_count > 200:
                    break
                
                # Find a violated progression
                # Check all progressions, pick a random violated one
                violated = []
                for idx, (a, b, c, e) in enumerate(progressions):
                    cs = {colors[a], colors[b], colors[c], colors[e]}
                    if len(cs) < 3:
                        violated.append(idx)
                
                if not violated:
                    # Valid solution found!
                    actual_colors = set(colors)
                    count = len(actual_colors)
                    if count < best_count:
                        best_count = count
                        best_colors = list(colors)
                    # If we used exactly k colors, we're done for this k
                    if count <= k:
                        break
                    improved = True
                    continue
                
                # Pick a random violated progression
                idx = random.choice(violated)
                a, b, c, e = progressions[idx]
                
                # Try changing one of the four positions to a different color
                positions = [a, b, c, e]
                random.shuffle(positions)
                
                for pos in positions:
                    old_color = colors[pos]
                    # Try each other color
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        colors[pos] = new_color
                        # Quick check: does this fix the violation without creating new ones?
                        # Just check this progression
                        cs = {colors[a], colors[b], colors[c], colors[e]}
                        if len(cs) >= 3:
                            # Check if this creates new violations (limited check)
                            # Count how many progressions involving pos are now violated
                            new_violations = 0
                            for idx2, (a2, b2, c2, e2) in enumerate(progressions):
                                if pos in (a2, b2, c2, e2):
                                    cs2 = {colors[a2], colors[b2], colors[c2], colors[e2]}
                                    if len(cs2) < 3:
                                        new_violations += 1
                            # Accept if we didn't increase violations too much
                            # Simple heuristic: accept if new_violations is small
                            if new_violations <= 3:
                                improved = True
                                break
                            else:
                                colors[pos] = old_color
                    
                    if improved:
                        break
            
            # End of local search for this restart
    
    # Fallback: use the seed program's greedy approach if we didn't find anything better
    if best_colors is None:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                           for d in range(1, (x-1)//2 + 1))):
                    colors.append(color)
                    break
        best_colors = colors
        best_count = len(set(colors))
    
    return best_colors
