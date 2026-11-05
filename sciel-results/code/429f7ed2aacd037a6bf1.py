import time
import numpy as np

def construct():
    """
    Minimize the number of distinct color labels used on positions 1..128
    such that every non-constant 4-term arithmetic progression (a, a+d, a+2d, a+3d)
    uses at least 3 distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    def count_violations(colors):
        """Count the number of APs that violate the constraint (distinct colors < 3)."""
        bad = 0
        for ap in ap_list:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if distinct count < 3
            # This is equivalent to: not (len(set) >= 3)
            # We can check: if (c0==c1 and c0==c2 and c0==c3) -> 1 color (bad)
            # or if the set size is 2.
            # A faster check:
            # If c0==c1 and c2==c3 and c0!=c2 -> 2 colors (bad)
            # If c0==c2 and c1==c3 and c0!=c1 -> 2 colors (bad)
            # If c0==c3 and c1==c2 and c0!=c1 -> 2 colors (bad)
            # If all same -> 1 color (bad)
            # If any other combination, it's 3 or 4 colors (good)
            
            # Let's just use set for clarity, it's fast enough for 2600 APs
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad

    # Try to find a valid coloring with k colors for k = 3, 4, 5...
    best_colors = None
    best_bad = float('inf')
    
    for k in range(3, 6):
        if time.time() > deadline:
            break
            
        # Multiple restarts for local search
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with random colors from 0 to k-1
            colors = np.random.randint(0, k, n)
            current_bad = count_violations(colors)
            
            if current_bad < best_bad:
                best_bad = current_bad
                best_colors = colors.copy()
                
            if current_bad == 0:
                break
                
            # Simulated annealing / hill climbing
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            while time.time() < deadline and T > min_T:
                improved = False
                for _ in range(50):
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
                    new_bad = count_violations(colors)
                    
                    delta = new_bad - current_bad
                    
                    if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                        if new_bad < best_bad:
                            best_bad = new_bad
                            best_colors = colors.copy()
                            if new_bad == 0:
                                return best_colors.tolist()
                        current_bad = new_bad
                        improved = True
                    else:
                        # Reject the move
                        colors[pos] = old_color
                
                T *= cooling
                
                if current_bad == 0:
                    break
    
    if best_colors is not None:
        return best_colors.tolist()
    
    # Fallback: greedy approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
