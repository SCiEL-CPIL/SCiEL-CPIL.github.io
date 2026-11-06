"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 128
    
    # Precompute all 4-term progressions
    progressions = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            if i + 3*d < n:
                progressions.append((i, i+d, i+2*d, i+3*d))
    
    def count_violations(colors):
        count = 0
        for i, j, k, l in progressions:
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                count += 1
        return count
    
    # Try to find a valid coloring with k colors
    for k in range(3, 128):
        best = None
        best_viol = float('inf')
        
        for restart in range(100):
            colors = [random.randint(0, k-1) for _ in range(n)]
            viol = count_violations(colors)
            
            if viol == 0:
                return colors
            
            # Simulated annealing
            temp = 10.0
            for step in range(20000):
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                new_viol = count_violations(colors)
                
                if new_viol < viol:
                    viol = new_viol
                    if viol == 0:
                        return colors
                else:
                    delta = new_viol - viol
                    if random.random() < (2.718281828 ** (-delta / temp)):
                        viol = new_viol
                    else:
                        colors[pos] = old_color
                
                temp *= 0.9995
                
                if viol < best_viol:
                    best_viol = viol
                    best = colors[:]
            
            if best_viol == 0:
                return best
        
        if best is not None and best_viol == 0:
            return best
    
    # Fallback: return a random coloring (may be invalid, but better than nothing)
    return [random.randint(0, 127) for _ in range(n)]

# EVOLVE-BLOCK-END

