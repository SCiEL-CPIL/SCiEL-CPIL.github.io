"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_4tps(n=128):
    """Generate all non-constant 4-term arithmetic progressions in 0..n-1."""
    tps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                tps.append((a, b, c, e))
    return tps

def _count_distinct_colors(tp, colors):
    """Count distinct colors in a 4-tuple of positions."""
    a, b, c, e = tp
    vals = {colors[a], colors[b], colors[c], colors[e]}
    return len(vals)

def _is_valid_assignment(colors, tps):
    """Check if every 4-TP has at least 3 distinct colors."""
    for tp in tps:
        if _count_distinct_colors(tp, colors) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    tps = _all_4tps(n)
    
    # Pre-compute for each position, which 4-TPs it participates in
    pos_tps = [[] for _ in range(n)]
    for i, tp in enumerate(tps):
        for pos in tp:
            pos_tps[pos].append(i)
    
    # We'll use a local search / simulated annealing approach
    # Start with a random 2-coloring and try to fix violations by switching colors
    
    best_colors = None
    best_score = float('inf')  # score = number of colors used
    
    # First, try to find a valid coloring with few colors
    # Start with k=2, then k=3, etc.
    
    for k in range(2, 10):  # try 2 to 9 colors
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts for each k
        max_restarts = 50 if k <= 3 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3.0:
                break
            
            # Initialize with random assignment from k colors
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search: repeatedly find a violated 4-TP and fix it
            # by changing one of its positions to a new color
            improved = True
            iteration = 0
            max_iterations = 5000
            
            while improved and iteration < max_iterations:
                if time.time() > deadline - 2.0:
                    break
                
                iteration += 1
                improved = False
                
                # Find a violated 4-TP
                violated = None
                for tp in tps:
                    if _count_distinct_colors(tp, colors) < 3:
                        violated = tp
                        break
                
                if violated is None:
                    # No violations! Check color count
                    used = _count_colors(colors)
                    if used < best_score:
                        best_score = used
                        best_colors = list(colors)
                    break
                
                # Try to fix the violation by changing one position in the 4-TP
                # to a color that resolves the violation without creating new ones
                a, b, c, e = violated
                positions = [a, b, c, e]
                
                fixed = False
                for pos in positions:
                    for new_color in range(k):
                        if new_color == colors[pos]:
                            continue
                        
                        # Try this change
                        old_color = colors[pos]
                        colors[pos] = new_color
                        
                        # Check if this creates new violations
                        new_violations = 0
                        for tp_idx in pos_tps[pos]:
                            tp = tps[tp_idx]
                            if _count_distinct_colors(tp, colors) < 3:
                                new_violations += 1
                            if new_violations > 2:  # early exit
                                break
                        
                        if new_violations == 0:
                            fixed = True
                            improved = True
                            break
                        else:
                            colors[pos] = old_color
                    
                    if fixed:
                        break
                
                if not fixed:
                    # Try a random change
                    pos = random.choice(positions)
                    old_color = colors[pos]
                    colors[pos] = random.randint(0, k - 1)
                    if colors[pos] != old_color:
                        improved = True
            
            if time.time() > deadline - 2.0:
                break
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        # Verify validity
        if _is_valid_assignment(best_colors, tps):
            return best_colors
    
    # Fallback: return a random valid coloring (shouldn't happen)
    # Use the seed program's approach as fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    
    return colors

# EVOLVE-BLOCK-END
