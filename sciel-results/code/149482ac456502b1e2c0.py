"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all 4-term APs (0-based)
    aps = []
    for a in range(n):
        for d in range(1, n):
            if a + 3*d < n:
                aps.append((a, a+d, a+2*d, a+3*d))
    
    # For each position, list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for i in range(4):
            pos_aps[ap[i]].append((idx, i))
    
    def check_validity(colors, num_colors):
        """Check if coloring is valid."""
        for ap in aps:
            vals = [colors[p] for p in ap]
            if len(set(vals)) < 3:
                return False
        return True
    
    def count_violations(colors, num_colors):
        """Count number of violated APs."""
        count = 0
        for ap in aps:
            vals = [colors[p] for p in ap]
            if len(set(vals)) < 3:
                count += 1
        return count
    
    # Try 3 colors with randomized local search
    best_result = None
    best_violations = float('inf')
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initial assignment with 3 colors
        colors = [random.randint(0, 2) for _ in range(n)]
        
        # Local search: try to fix violations
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            violations = count_violations(colors, 3)
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_result = colors[:]
            
            # Pick a random violated AP and change one of its positions
            violated_aps = []
            for idx, ap in enumerate(aps):
                vals = [colors[p] for p in ap]
                if len(set(vals)) < 3:
                    violated_aps.append(ap)
            
            if not violated_aps:
                break
            
            ap = random.choice(violated_aps)
            pos = random.choice(ap)
            old_color = colors[pos]
            new_color = random.randint(0, 2)
            if new_color == old_color:
                continue
            colors[pos] = new_color
    
    if best_result is not None and check_validity(best_result, 3):
        return best_result
    
    # If 3 colors didn't work, try 4 colors
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        colors = [random.randint(0, 3) for _ in range(n)]
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            violations = count_violations(colors, 4)
            if violations == 0:
                return colors
            
            violated_aps = []
            for ap in aps:
                vals = [colors[p] for p in ap]
                if len(set(vals)) < 3:
                    violated_aps.append(ap)
            
            if not violated_aps:
                break
            
            ap = random.choice(violated_aps)
            pos = random.choice(ap)
            new_color = random.randint(0, 3)
            colors[pos] = new_color
    
    # Fallback: use the seed approach
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
