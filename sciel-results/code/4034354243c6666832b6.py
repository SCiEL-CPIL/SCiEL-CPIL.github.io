import time
import random
import numpy as np

def construct():
    """Return 256 colours in {0,1} for integers 1..256 minimizing monochromatic 4-APs."""
    n = 256
    deadline = time.time() + 110  # 10s buffer for 120s limit
    
    # Precompute all 4-term APs: a, a+d, a+2d, a+3d with 0-based indexing
    # a >= 0, d >= 1, a + 3d <= n-1
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each position, the list of AP indices containing it
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d_pos) in enumerate(ap_list):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d_pos].append(idx)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for a, b, c, d_pos in ap_list:
            if coloring[a] == coloring[b] == coloring[c] == coloring[d_pos]:
                count += 1
        return count
    
    def thue_morse():
        """Generate Thue-Morse sequence."""
        tm = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    def alternating():
        """Generate alternating pattern."""
        return np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
    
    def random_coloring():
        return np.random.randint(0, 2, size=n).astype(np.int8)
    
    # Initialize with several strategies
    candidates = []
    candidates.append(thue_morse())
    candidates.append(alternating())
    candidates.append(random_coloring())
    candidates.append(random_coloring())
    
    # Local search with simulated annealing
    def local_search(initial, max_steps=50000):
        coloring = initial.copy()
        current_cost = count_mono(coloring)
        best_coloring = coloring.copy()
        best_cost = current_cost
        
        # Precompute neighbor gains for efficiency
        # For each position, store the APs it belongs to
        # When flipping position p, the change in cost depends on the other 3 positions in each AP
        
        temp = 1.0
        cooling = 0.9995
        step = 0
        start_time = time.time()
        
        while step < max_steps and (time.time() - start_time) < 105:
            if time.time() > deadline:
                break
            
            p = random.randint(0, n - 1)
            
            # Calculate cost change if we flip position p
            delta = 0
            for ap_idx in pos_aps[p]:
                a, b, c, d_pos = ap_list[ap_idx]
                # Get the other three positions
                others = [x for x in (a, b, c, d_pos) if x != p]
                other_colors = [coloring[x] for x in others]
                
                # Before flip: all four same?
                before_mono = (coloring[p] == other_colors[0] == other_colors[1] == other_colors[2])
                # After flip: all four same?
                new_color = 1 - coloring[p]
                after_mono = (new_color == other_colors[0] == other_colors[1] == other_colors[2])
                
                if before_mono and not after_mono:
                    delta -= 1
                elif not before_mono and after_mono:
                    delta += 1
            
            new_cost = current_cost + delta
            
            if new_cost < current_cost or random.random() < np.exp(-delta / max(temp, 1e-10)):
                coloring[p] = 1 - coloring[p]
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_coloring = coloring.copy()
            
            step += 1
            temp *= cooling
        
        return best_coloring, best_cost
    
    # Run local search from each candidate
    best_overall = None
    best_overall_cost = float('inf')
    
    for candidate in candidates:
        result, cost = local_search(candidate)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall = result
        
        if time.time() > deadline - 5:
            break
    
    # If we have time, try more random restarts
    if best_overall is not None and time.time() < deadline - 10:
        num_restarts = 0
        while time.time() < deadline - 5:
            restart = random_coloring()
            result, cost = local_search(restart, max_steps=20000)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall = result
            num_restarts += 1
    
    # Convert to list of ints
    return [int(c) for c in best_overall]
