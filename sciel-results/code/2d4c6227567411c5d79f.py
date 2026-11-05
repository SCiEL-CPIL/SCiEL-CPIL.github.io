"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Minimize the number of distinct color labels used on positions 1..128
    such that every non-constant 4-term arithmetic progression (a, a+d, a+2d, a+3d)
    uses at least 3 distinct colors.
    
    Strategy:
    1. Try to find a valid coloring with 3 colors (labels 0, 1, 2).
    2. If 3 colors is not possible (or too slow), try 4 colors.
    3. Use a local search / backtracking hybrid approach.
    
    The seed program uses a greedy approach which likely uses many colors.
    We know that 2 colors is impossible for sufficiently long sequences.
    Let's try to construct a 3-coloring or 4-coloring.
    
    We will use a simulated annealing or local search approach to minimize the number of colors used,
    while satisfying the constraint that every 4-AP has >= 3 distinct colors.
    Actually, the objective is to minimize the number of distinct labels used.
    This is equivalent to finding the smallest k such that a valid k-coloring exists.
    
    Let's first try a direct backtracking for k=3, then k=4, etc.
    If backtracking is too slow, we can fall back to a greedy approach that tries to reuse colors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_distinct_colors(aps, colors):
        """Count the number of APs that violate the constraint."""
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            distinct = len({c0, c1, c2, c3})
            if distinct < 3:
                bad += 1
        return bad
    
    def is_valid_coloring(colors):
        """Check if the coloring is valid."""
        return count_distinct_colors(ap_list, colors) == 0
    
    def num_distinct_colors(colors):
        """Count the number of distinct colors used."""
        return len(set(colors))
    
    # Try to find a valid coloring with k colors for k = 3, 4, 5, ...
    for k in range(3, 10):
        if time.time() > deadline:
            break
        
        # Initialize with random colors from 0 to k-1
        colors = np.random.randint(0, k, n)
        
        # Local search to minimize violations
        best_colors = colors.copy()
        best_bad = count_distinct_colors(ap_list, best_colors)
        
        # Simulated annealing / hill climbing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Pick a random position
                pos = np.random.randint(0, n)
                old_color = colors[pos]
                
                # Try a new color
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_bad = count_distinct_colors(ap_list, colors)
                
                delta = new_bad - best_bad
                
                if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                    if new_bad < best_bad:
                        best_bad = new_bad
                        best_colors = colors.copy()
                        improved = True
                    # Accept the move
                else:
                    # Reject the move
                    colors[pos] = old_color
            
            T *= cooling
            
            if best_bad == 0:
                break
        
        if best_bad == 0:
            # Found a valid coloring with k colors
            return best_colors.tolist()
    
    # Fallback: greedy approach from the seed program
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
# EVOLVE-BLOCK-END
