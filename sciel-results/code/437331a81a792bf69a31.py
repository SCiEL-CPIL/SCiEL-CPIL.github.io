import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
                    cycles.append(e1)
                    e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
                    cycles.append(e2)
                    e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
                    cycles.append(e3)
    return cycles

def construct():
    deadline = time.time() + 55
    
    cycles = build_four_cycles()
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)
    
    # Greedy initialization: assign colors to maximize distinct colors
    colors = [0] * NUM_EDGES
    used_colors = set([0])
    
    # Process edges in order of constraint tightness (most constrained first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(cycles_by_edge[e]), reverse=True)
    
    for e in edge_order:
        # Try to find a new color that is valid
        found = False
        for c in range(1, NUM_EDGES):
            if c in used_colors:
                continue
            valid = True
            for cycle in cycles_by_edge[e]:
                other_colors = [colors[oe] for oe in cycle if oe != e]
                other_set = set(other_colors)
                if len(other_set) == 3 and c not in other_set:
                    valid = False
                    break
            if valid:
                colors[e] = c
                used_colors.add(c)
                found = True
                break
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search with simulated annealing
    rng = random.Random(42)
    
    # Precompute edge pairs that don't share a cycle (for swap moves)
    # Two edges can be swapped if they don't appear together in any 4-cycle
    # Actually, for swap to be safe, we just need to check validity after swap
    
    # Track color counts for fast distinct color updates
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    def get_distinct_count(color_counts):
        return len([c for c, cnt in color_counts.items() if cnt > 0])
    
    def try_recolor(colors, e, nc, cycles_by_edge, color_counts):
        """Try to recolor edge e to color nc. Returns True if valid."""
        cur = colors[e]
        if cur == nc:
            return True
        
        # Check all cycles containing this edge
        for cycle in cycles_by_edge[e]:
            # Get colors of other edges in cycle
            other_colors = []
            for oe in cycle:
                if oe != e:
                    other_colors.append(colors[oe])
            other_set = set(other_colors)
            if len(other_set) == 3 and nc not in other_set:
                return False
        return True
    
    def apply_recolor(colors, e, nc, color_counts):
        """Apply recoloring and update color counts."""
        cur = colors[e]
        if cur == nc:
            return
        colors[e] = nc
        color_counts[cur] -= 1
        if color_counts[cur] == 0:
            del color_counts[cur]
        color_counts[nc] = color_counts.get(nc, 0) + 1
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    current_count = get_distinct_count(color_counts)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to increase distinct colors by recoloring
        for e in edge_order:
            if time.time() > deadline:
                break
            cur = colors[e]
            
            # Try new colors (not currently used)
            for nc in range(NUM_EDGES):
                if nc == cur:
                    continue
                if nc in color_counts:
                    continue  # Skip already-used colors for count-increasing moves
                
                if try_recolor(colors, e, nc, cycles_by_edge, color_counts):
                    apply_recolor(colors, e, nc, color_counts)
                    current_count += 1
                    improved = True
                    break
            
            if time.time() > deadline:
                break
        
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if not improved:
            # Phase 2: Simulated annealing moves to escape local optima
            T = max(T * cooling, T_min)
            
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Random move: recolor a random edge to a random color
                e = rng.randint(0, NUM_EDGES - 1)
                nc = rng.randint(0, NUM_EDGES - 1)
                cur = colors[e]
                
                if nc == cur:
                    continue
                
                if not try_recolor(colors, e, nc, cycles_by_edge, color_counts):
                    continue
                
                # Compute delta in distinct color count
                old_count = current_count
                apply_recolor(colors, e, nc, color_counts)
                new_count = get_distinct_count(color_counts)
                delta = new_count - old_count
                
                # Accept with probability based on temperature
                if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                    current_count = new_count
                    # Keep the move
                else:
                    # Reject: revert
                    apply_recolor(colors, e, cur, color_counts)
                    current_count = old_count
            
            # Update best after SA phase
            if current_count > best_count:
                best_count = current_count
                best_colors = colors[:]
            
            # If stuck, do a stronger perturbation
            if T < 1.0:
                # Random restart of a subset
                num_perturb = rng.randint(3, 8)
                for _ in range(num_perturb):
                    if time.time() > deadline:
                        break
                    e = rng.randint(0, NUM_EDGES - 1)
                    nc = rng.randint(0, NUM_EDGES - 1)
                    if nc == colors[e]:
                        continue
                    if try_recolor(colors, e, nc, cycles_by_edge, color_counts):
                        old_count = current_count
                        apply_recolor(colors, e, nc, color_counts)
                        current_count = get_distinct_count(color_counts)
                
                if current_count > best_count:
                    best_count = current_count
                    best_colors = colors[:]
        
        # Periodic full restart from best if no progress for a while
        if current_count < best_count - 2 and T < 1.0:
            # Reset to best and reheat
            colors = best_colors[:]
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            current_count = best_count
            T = 5.0
    
    return best_colors
