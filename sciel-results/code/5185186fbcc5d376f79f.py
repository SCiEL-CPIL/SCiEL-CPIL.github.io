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
    all_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    all_cycles.append([get_edge(a,b), get_edge(b,c), get_edge(c,d), get_edge(d,a)])
                    all_cycles.append([get_edge(a,b), get_edge(b,d), get_edge(d,c), get_edge(c,a)])
                    all_cycles.append([get_edge(a,c), get_edge(c,b), get_edge(b,d), get_edge(d,a)])
    
    cycles_arr = np.array(all_cycles, dtype=np.int32)
    num_cycles = len(cycles_arr)
    
    # Precompute for each edge, which cycles it belongs to (for fast incremental checks)
    edge_to_cycles = [[] for _ in range(120)]
    for ci in range(num_cycles):
        for ei in cycles_arr[ci]:
            edge_to_cycles[ei].append(ci)
    
    def is_valid_numpy(colors_np):
        """Vectorized check: no cycle has 4 distinct colors."""
        cycle_colors = colors_np[cycles_arr]
        sorted_colors = np.sort(cycle_colors, axis=1)
        diffs = np.abs(np.diff(sorted_colors, axis=1))
        rainbow_mask = np.all(diffs > 0, axis=1)
        return not np.any(rainbow_mask)
    
    def is_valid_fast(colors_np, changed_edge):
        """Check only cycles affected by changing one edge."""
        affected = edge_to_cycles[changed_edge]
        if not affected:
            return True
        cycle_colors = colors_np[cycles_arr[affected]]
        sorted_colors = np.sort(cycle_colors, axis=1)
        diffs = np.abs(np.diff(sorted_colors, axis=1))
        rainbow_mask = np.all(diffs > 0, axis=1)
        return not np.any(rainbow_mask)
    
    def count_distinct(colors_list):
        return len(set(colors_list))
    
    # Build structured initial colorings
    def initial_coloring_min(i):
        """Color by min(i,j) - gives 15 colors, always valid."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = i
        return col
    
    def initial_coloring_sum_mod(k):
        """Color by (i+j) mod k."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = (i + j) % k
        return col
    
    def initial_coloring_diff_mod(k):
        """Color by (j-i) mod k."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = (j - i) % k
        return col
    
    def initial_coloring_row_col(k1, k2):
        """Color by i mod k1, combined with j mod k2."""
        col = [0] * 120
        for i in range(N):
            for j in range(i+1, N):
                col[get_edge(i, j)] = (i % k1) * k2 + (j % k2)
        return col
    
    # Try multiple structured initializations
    best_colors = None
    best_count = 0
    
    initializations = [
        initial_coloring_min,
        lambda: initial_coloring_sum_mod(4),
        lambda: initial_coloring_sum_mod(5),
        lambda: initial_coloring_sum_mod(6),
        lambda: initial_coloring_diff_mod(4),
        lambda: initial_coloring_diff_mod(5),
        lambda: initial_coloring_diff_mod(6),
        lambda: initial_coloring_diff_mod(8),
        lambda: initial_coloring_row_col(4, 4),
        lambda: initial_coloring_row_col(4, 5),
        lambda: initial_coloring_row_col(5, 4),
    ]
    
    for init_fn in initializations:
        if time.time() > deadline - 10:
            break
        try:
            col = init_fn()
        except Exception:
            continue
        col_np = np.array(col, dtype=np.int32)
        if is_valid_numpy(col_np):
            cnt = count_distinct(col)
            if cnt > best_count:
                best_count = cnt
                best_colors = col[:]
    
    # If no valid initialization found, use min(i,j)
    if best_colors is None:
        best_colors = initial_coloring_min()
        best_count = count_distinct(best_colors)
    
    colors = best_colors[:]
    colors_np = np.array(colors, dtype=np.int32)
    
    random.seed(42)
    np.random.seed(42)
    
    # Simulated annealing with focus on introducing new colors
    temperature = 10.0
    cooling_rate = 0.9997
    min_temperature = 0.05
    
    while time.time() < deadline - 8 and temperature > min_temperature:
        ei = random.randint(0, 119)
        old_color = colors[ei]
        
        # Bias toward new colors
        used_colors = set(colors)
        num_used = len(used_colors)
        
        if random.random() < 0.4 and num_used < 120:
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
        
        if is_valid_fast(colors_np, ei):
            new_count = count_distinct(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = colors[:]
        else:
            # Reject
            colors[ei] = old_color
            colors_np[ei] = old_color
        
        temperature *= cooling_rate
    
    # Local search: for each edge, try ALL possible new colors to maximize count
    for pass_num in range(3):
        if time.time() > deadline - 3:
            break
        for ei in range(120):
            if time.time() > deadline - 2:
                break
            old_color = colors[ei]
            best_new_color = old_color
            best_new_count = count_distinct(colors)
            
            # Try all 120 colors
            for new_color in range(120):
                if new_color == old_color:
                    continue
                colors[ei] = new_color
                colors_np[ei] = new_color
                if is_valid_fast(colors_np, ei):
                    new_count = count_distinct(colors)
                    if new_count > best_new_count:
                        best_new_count = new_count
                        best_new_color = new_color
                colors[ei] = old_color
                colors_np[ei] = old_color
            
            if best_new_color != old_color:
                colors[ei] = best_new_color
                colors_np[ei] = best_new_color
                if best_new_count > best_count:
                    best_count = best_new_count
                    best_colors = colors[:]
    
    # Final polish: try to add colors that are currently unused
    if time.time() < deadline - 1:
        used = set(colors)
        unused = [c for c in range(120) if c not in used]
        random.shuffle(unused)
        
        for target_color in unused[:30]:  # Try up to 30 unused colors
            if time.time() > deadline - 1:
                break
            found = False
            for ei in range(120):
                if time.time() > deadline - 1:
                    break
                old_color = colors[ei]
                colors[ei] = target_color
                colors_np[ei] = target_color
                if is_valid_fast(colors_np, ei):
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                        found = True
                        break
                colors[ei] = old_color
                colors_np[ei] = old_color
            if found:
                # Update colors to reflect the change
                # Re-run local search briefly
                for ei in range(120):
                    if time.time() > deadline - 1:
                        break
                    old_color = colors[ei]
                    for new_color in range(120):
                        if new_color == old_color:
                            continue
                        colors[ei] = new_color
                        colors_np[ei] = new_color
                        if is_valid_fast(colors_np, ei):
                            new_count = count_distinct(colors)
                            if new_count > best_count:
                                best_count = new_count
                                best_colors = colors[:]
                                break
                        colors[ei] = old_color
                        colors_np[ei] = old_color
    
    return best_colors
# EVOLVE-BLOCK-END
