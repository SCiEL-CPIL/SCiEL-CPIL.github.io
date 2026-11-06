"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    Uses NumPy-vectorized validation with simulated annealing to maximize
    distinct colors while maintaining the C4-rainbow-free property.
    """
    deadline = time.time() + 55  # Safety margin for 90s limit
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_edge(i, j):
        if i > j:
            i, j = j, i
        return edge_idx(i, j)
    
    # Precompute all 4-cycles as numpy arrays for vectorized validation
    # Each cycle is 4 edge indices; we need 3 cycles per 4-vertex subset
    all_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    all_cycles.append([get_edge(a,b), get_edge(b,c), get_edge(c,d), get_edge(d,a)])
                    # Cycle 2: a-b-d-c-a
                    all_cycles.append([get_edge(a,b), get_edge(b,d), get_edge(d,c), get_edge(c,a)])
                    # Cycle 3: a-c-b-d-a
                    all_cycles.append([get_edge(a,c), get_edge(c,b), get_edge(b,d), get_edge(d,a)])
    
    cycles_arr = np.array(all_cycles, dtype=np.int32)  # shape (num_cycles, 4)
    num_cycles = len(cycles_arr)
    
    def is_valid_numpy(colors_np):
        """Vectorized check: no cycle has 4 distinct colors."""
        # Get colors for all edges in all cycles: shape (num_cycles, 4)
        cycle_colors = colors_np[cycles_arr]  # shape (num_cycles, 4)
        # A cycle is rainbow iff all 4 colors are distinct
        # Check: sort the 4 colors, if all different then rainbow
        sorted_colors = np.sort(cycle_colors, axis=1)
        # Rainbow iff all adjacent pairs differ
        diffs = np.abs(np.diff(sorted_colors, axis=1))
        rainbow_mask = np.all(diffs > 0, axis=1)
        return not np.any(rainbow_mask)
    
    def count_distinct(colors_list):
        return len(set(colors_list))
    
    # Build a good initial coloring: use a structured pattern
    # Color based on (i+j) mod k for some k, then refine
    def initial_coloring():
        """Create a valid starting coloring with multiple colors."""
        # Use a simple valid pattern: color by min(i,j) gives 15 colors
        # But let's try something that might allow more
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                # Use a pattern that avoids rainbow C4s
                # Color = (i + j) % 15 works? Let's use min(i,j) as base
                col[get_edge(i, j)] = min(i, j)
        return col
    
    # Start with a structured coloring
    colors = initial_coloring()
    colors_np = np.array(colors, dtype=np.int32)
    
    # Verify initial coloring is valid
    if not is_valid_numpy(colors_np):
        # Fallback to all zeros
        colors = [0] * 120
        colors_np = np.array(colors, dtype=np.int32)
    
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    random.seed(42)
    np.random.seed(42)
    
    # Simulated annealing
    temperature = 10.0
    cooling_rate = 0.9995
    min_temperature = 0.1
    
    while time.time() < deadline and temperature > min_temperature:
        # Randomly pick an edge to change
        ei = random.randint(0, 119)
        old_color = colors[ei]
        
        # Choose a new color: bias toward introducing new colors
        # with some probability, or pick from existing/new colors
        if random.random() < 0.3:
            # Try a new color (possibly one not yet used)
            used_colors = set(colors)
            available = [c for c in range(120) if c not in used_colors]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, 119)
        else:
            new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        colors[ei] = new_color
        colors_np[ei] = new_color
        
        if is_valid_numpy(colors_np):
            new_count = count_distinct(colors)
            if new_count >= best_count:
                best_colors = colors[:]
                best_count = new_count
            # Accept if same or better, or with probability based on temperature
            # For simplicity, accept all valid moves
        else:
            # Reject invalid move
            colors[ei] = old_color
            colors_np[ei] = old_color
        
        temperature *= cooling_rate
    
    # After SA, do a local search to try to add more colors
    # Try each edge, see if we can assign a new color
    for ei in range(120):
        if time.time() > deadline - 2:
            break
        old_color = colors[ei]
        # Try colors from high to low to find new ones
        for new_color in range(119, -1, -1):
            if new_color == old_color:
                continue
            colors[ei] = new_color
            colors_np[ei] = new_color
            if is_valid_numpy(colors_np):
                new_count = count_distinct(colors)
                if new_count > best_count:
                    best_colors = colors[:]
                    best_count = new_count
                    break
        colors[ei] = old_color
        colors_np[ei] = old_color
    
    # Multiple restarts if time permits
    while time.time() < deadline - 5:
        # New random restart
        colors = [random.randint(0, 119) for _ in range(120)]
        colors_np = np.array(colors, dtype=np.int32)
        
        # Try to fix up the random coloring
        valid = False
        for attempt in range(100):
            if is_valid_numpy(colors_np):
                valid = True
                break
            # Find a violating cycle and fix it
            cycle_colors = colors_np[cycles_arr]
            sorted_colors = np.sort(cycle_colors, axis=1)
            diffs = np.abs(np.diff(sorted_colors, axis=1))
            rainbow_mask = np.all(diffs > 0, axis=1)
            if np.any(rainbow_mask):
                # Pick a random violating cycle
                violating_indices = np.where(rainbow_mask)[0]
                idx = random.choice(violating_indices)
                cycle = cycles_arr[idx]
                # Change one edge in this cycle to match another
                edge_to_change = random.choice(cycle.tolist())
                other_edges = [e for e in cycle if e != edge_to_change]
                colors[edge_to_change] = colors[random.choice(other_edges)]
                colors_np[edge_to_change] = colors_np[random.choice(other_edges)]
        
        if valid:
            cnt = count_distinct(colors)
            if cnt > best_count:
                best_colors = colors[:]
                best_count = cnt
            
            # Local search on this coloring
            for ei in range(120):
                if time.time() > deadline - 2:
                    break
                old_color = colors[ei]
                for new_color in range(119, -1, -1):
                    if new_color == old_color:
                        continue
                    colors[ei] = new_color
                    colors_np[ei] = new_color
                    if is_valid_numpy(colors_np):
                        new_count = count_distinct(colors)
                        if new_count > best_count:
                            best_colors = colors[:]
                            best_count = new_count
                            break
                colors[ei] = old_color
                colors_np[ei] = old_color
    
    return best_colors
# EVOLVE-BLOCK-END
