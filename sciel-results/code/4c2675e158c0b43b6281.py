import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 110
    
    # Precompute all 4-term APs as index tuples (0-based)
    # For each AP, store the 4 indices
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    
    # For each position, precompute which APs it participates in
    # ap_indices[pos] = list of AP indices where pos is one of the 4 elements
    # Also, for each AP, we need to know which position is which
    # Let's store for each position, a list of (ap_idx, role) where role is 0,1,2,3
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_to_aps[i0].append((ap_idx, 0))
        pos_to_aps[i1].append((ap_idx, 1))
        pos_to_aps[i2].append((ap_idx, 2))
        pos_to_aps[i3].append((ap_idx, 3))
    
    def count_mono(colors):
        """Count monochromatic 4-term APs. colors is a numpy array of 0s and 1s."""
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        # Mono if all same: all 0 or all 1
        mono0 = (c0 == 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
        mono1 = (c0 == 1) & (c1 == 1) & (c2 == 1) & (c3 == 1)
        return int(np.sum(mono0) + np.sum(mono1))
    
    def greedy_polish(colors, max_iters=5000):
        """Local search: flip bits to reduce mono count."""
        colors = colors.copy()
        current = count_mono(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping each position
            for pos in range(N):
                old_c = colors[pos]
                new_c = 1 - old_c
                colors[pos] = new_c
                new_count = count_mono(colors)
                if new_count < current:
                    current = new_count
                    improved = True
                else:
                    colors[pos] = old_c
            
            if not improved:
                break
        
        return colors, current
    
    def simulated_annealing(initial_colors, start_temp=1.0, end_temp=0.01, 
                            cooling_rate=0.999, max_steps=50000):
        """Simulated annealing to escape local optima."""
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        temp = start_temp
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random position to flip
            pos = random.randint(0, N - 1)
            old_c = colors[pos]
            new_c = 1 - old_c
            colors[pos] = new_c
            new_count = count_mono(colors)
            
            delta = new_count - current
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current = new_count
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            else:
                colors[pos] = old_c
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
        
        return best_colors, best_count
    
    # Strategy: Multiple restarts with random initial colorings, 
    # each followed by greedy polish, and occasionally simulated annealing.
    
    best_colors = None
    best_score = float('inf')
    
    # First, try the original greedy approach as a baseline
    colors_greedy = np.zeros(N, dtype=np.int8)
    for k in range(N):
        # Simple heuristic: balance colors
        c0_count = np.sum(colors_greedy[:k] == 0)
        c1_count = np.sum(colors_greedy[:k] == 1)
        if c0_count <= c1_count:
            colors_greedy[k] = 0
        else:
            colors_greedy[k] = 1
    
    polished, score = greedy_polish(colors_greedy)
    if score < best_score:
        best_score = score
        best_colors = polished.copy()
    
    # Now do multiple random restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring
        random_colors = np.random.randint(0, 2, N, dtype=np.int8)
        
        # Greedy polish first
        polished, score = greedy_polish(random_colors, max_iters=3000)
        
        if score < best_score:
            best_score = score
            best_colors = polished.copy()
        
        # If we have time, try simulated annealing from this state
        if time.time() < deadline - 10:
            sa_colors, sa_score = simulated_annealing(
                polished, start_temp=0.5, end_temp=0.01, 
                cooling_rate=0.9995, max_steps=20000
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors.copy()
    
    # Final polish on the best solution
    if time.time() < deadline - 2:
        final_colors, final_score = greedy_polish(best_colors, max_iters=10000)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors.copy()
    
    # Convert to list of ints
    return [int(c) for c in best_colors]
