"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i, j) with i < j."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles():
    """Build all 3 four-cycles for each 4-vertex subset."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on vertices a,b,c,d
                    # Cycle 1: a-b-c-d-a
                    cycles.append((
                        _edge_idx(a, b),
                        _edge_idx(b, c),
                        _edge_idx(c, d),
                        _edge_idx(d, a)
                    ))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((
                        _edge_idx(a, b),
                        _edge_idx(b, d),
                        _edge_idx(d, c),
                        _edge_idx(c, a)
                    ))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((
                        _edge_idx(a, c),
                        _edge_idx(c, b),
                        _edge_idx(b, d),
                        _edge_idx(d, a)
                    ))
    return cycles

def _precompute_cycle_data(cycles):
    """Precompute numpy arrays for fast vectorized cycle checking."""
    num_cycles = len(cycles)
    cycle_edges = np.array(cycles, dtype=np.int32)  # shape (num_cycles, 4)
    
    # For each edge, which cycles does it appear in?
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for pos in range(4):
            e = cycle_edges[ci, pos]
            edge_to_cycles[e].append(ci)
    
    # Convert to arrays
    edge_cycle_lists = []
    for e in range(NUM_EDGES):
        edge_cycle_lists.append(np.array(edge_to_cycles[e], dtype=np.int32))
    
    return cycle_edges, edge_cycle_lists

def _is_valid_fast(colors, cycle_edges, edge_cycle_lists):
    """Check validity using numpy for speed."""
    colors_arr = np.array(colors, dtype=np.int32)
    cycle_colors = colors_arr[cycle_edges]  # shape (num_cycles, 4)
    # A cycle is rainbow if all 4 colors are distinct
    # Count unique colors per row
    for ci in range(len(cycle_edges)):
        row = cycle_colors[ci]
        if len(set(row.tolist())) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _algebraic_init(a, b, c, M):
    """Generate algebraic coloring (a*i + b*j + c) % M."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            colors[idx] = (a * i + b * j + c) % M
    return colors

def _fast_local_search(colors, cycle_edges, edge_cycle_lists, deadline, rng, max_iter=5000):
    """Fast local search with numpy-optimized checks."""
    best = list(colors)
    best_count = _count_colors(best)
    colors_arr = np.array(best, dtype=np.int32)
    
    # Build a faster check: for a single edge change, check only affected cycles
    for iteration in range(max_iter):
        if time.time() >= deadline:
            break
        
        # Pick random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_c = best[e]
        
        # Try a random new color (not the old one)
        new_c = rng.randint(0, MAX_COLORS - 1)
        if new_c == old_c:
            continue
        
        # Check only cycles containing edge e
        affected = edge_cycle_lists[e]
        if len(affected) == 0:
            # No constraints, just take it
            best[e] = new_c
            colors_arr[e] = new_c
            new_count = _count_colors(best)
            if new_count > best_count:
                best_count = new_count
            else:
                best[e] = old_c
                colors_arr[e] = old_c
            continue
        
        valid = True
        for ci in affected:
            e1, e2, e3, e4 = cycle_edges[ci]
            c1 = colors_arr[e1] if e1 != e else new_c
            c2 = colors_arr[e2] if e2 != e else new_c
            c3 = colors_arr[e3] if e3 != e else new_c
            c4 = colors_arr[e4] if e4 != e else new_c
            if len({c1, c2, c3, c4}) == 4:
                valid = False
                break
        
        if valid:
            best[e] = new_c
            colors_arr[e] = new_c
            new_count = _count_colors(best)
            if new_count > best_count:
                best_count = new_count
            else:
                # Revert
                best[e] = old_c
                colors_arr[e] = old_c
    
    return best, best_count

def _systematic_local_search(colors, cycle_edges, edge_cycle_lists, deadline, rng):
    """Systematic local search: try each edge with multiple candidate colors."""
    best = list(colors)
    best_count = _count_colors(best)
    colors_arr = np.array(best, dtype=np.int32)
    
    edges = list(range(NUM_EDGES))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(edges)
        for e in edges:
            if time.time() >= deadline:
                break
            old_c = best[e]
            affected = edge_cycle_lists[e]
            
            # Try a few random candidate colors
            candidates = rng.sample(range(MAX_COLORS), min(10, MAX_COLORS))
            for new_c in candidates:
                if new_c == old_c:
                    continue
                
                valid = True
                for ci in affected:
                    e1, e2, e3, e4 = cycle_edges[ci]
                    c1 = colors_arr[e1] if e1 != e else new_c
                    c2 = colors_arr[e2] if e2 != e else new_c
                    c3 = colors_arr[e3] if e3 != e else new_c
                    c4 = colors_arr[e4] if e4 != e else new_c
                    if len({c1, c2, c3, c4}) == 4:
                        valid = False
                        break
                
                if valid:
                    best[e] = new_c
                    colors_arr[e] = new_c
                    new_count = _count_colors(best)
                    if new_count > best_count:
                        best_count = new_count
                        improved = True
                    else:
                        best[e] = old_c
                        colors_arr[e] = old_c
                    break  # Only one change per edge per pass
    
    return best, best_count

def _random_valid_init(rng, cycle_edges):
    """Generate a random valid coloring by greedy assignment."""
    colors = [0] * NUM_EDGES
    # Start with all zeros (valid)
    # Then randomly assign colors while maintaining validity
    order = list(range(NUM_EDGES))
    rng.shuffle(order)
    
    for e in order:
        # Try random colors
        for _ in range(50):
            new_c = rng.randint(0, MAX_COLORS - 1)
            colors[e] = new_c
            # Check if still valid (only need to check cycles containing e)
            # Quick check: just verify the whole thing is valid
            # This is expensive, so do it less frequently
            break  # For now just assign randomly
    
    # Verify and fix
    colors_arr = np.array(colors, dtype=np.int32)
    # Check validity
    for ci in range(len(cycle_edges)):
        e1, e2, e3, e4 = cycle_edges[ci]
        c1, c2, c3, c4 = colors_arr[e1], colors_arr[e2], colors_arr[e3], colors_arr[e4]
        if len({c1, c2, c3, c4}) == 4:
            # Fix: set one edge to match another
            colors[e4] = colors[e1]
            colors_arr[e4] = colors_arr[e1]
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    cycles = _build_cycles()
    cycle_edges, edge_cycle_lists = _precompute_cycle_data(cycles)
    
    best_colors = None
    best_count = 0
    
    # Generate diverse initial colorings
    rng = random.Random(42)
    
    # Algebraic initializations
    init_colors = []
    for M in [10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80, 100, 120]:
        for a in range(1, min(M, 16) + 1):
            for b in range(1, min(M, 16) + 1):
                if a == b and M <= 20:
                    continue
                colors = _algebraic_init(a, b, 0, M)
                # Quick validity check
                colors_arr = np.array(colors, dtype=np.int32)
                valid = True
                for ci in range(len(cycle_edges)):
                    e1, e2, e3, e4 = cycle_edges[ci]
                    c1, c2, c3, c4 = colors_arr[e1], colors_arr[e2], colors_arr[e3], colors_arr[e4]
                    if len({c1, c2, c3, c4}) == 4:
                        valid = False
                        break
                if valid:
                    count = _count_colors(colors)
                    init_colors.append((colors, count))
                    if count > best_count:
                        best_count = count
                        best_colors = list(colors)
                if len(init_colors) > 50:
                    break
            if len(init_colors) > 50:
                break
        if len(init_colors) > 50:
            break
    
    # Add some random valid inits
    for _ in range(10):
        colors = _random_valid_init(rng, cycle_edges)
        colors_arr = np.array(colors, dtype=np.int32)
        valid = True
        for ci in range(len(cycle_edges)):
            e1, e2, e3, e4 = cycle_edges[ci]
            c1, c2, c3, c4 = colors_arr[e1], colors_arr[e2], colors_arr[e3], colors_arr[e4]
            if len({c1, c2, c3, c4}) == 4:
                valid = False
                break
        if valid:
            count = _count_colors(colors)
            init_colors.append((colors, count))
            if count > best_count:
                best_count = count
                best_colors = list(colors)
    
    # Population-based local search from multiple starting points
    if time.time() < deadline - 2:
        # Take top initial colorings and run local search on each
        init_colors.sort(key=lambda x: -x[1])
        num_starts = min(len(init_colors), 15)
        
        for si in range(num_starts):
            if time.time() >= deadline - 1:
                break
            
            start_colors, _ = init_colors[si]
            sub_deadline = min(deadline - 1, start_time + 55 - (si * 2))
            
            # Fast random local search
            ls_colors, ls_count = _fast_local_search(
                start_colors, cycle_edges, edge_cycle_lists, 
                sub_deadline, rng, max_iter=3000
            )
            
            if ls_count > best_count:
                best_count = ls_count
                best_colors = ls_colors
            
            # Systematic local search on the best result
            if time.time() < deadline - 1:
                sys_colors, sys_count = _systematic_local_search(
                    ls_colors, cycle_edges, edge_cycle_lists,
                    deadline - 1, rng
                )
                if sys_count > best_count:
                    best_count = sys_count
                    best_colors = sys_colors
    
    # Final verification
    if best_colors is not None:
        colors_arr = np.array(best_colors, dtype=np.int32)
        for ci in range(len(cycle_edges)):
            e1, e2, e3, e4 = cycle_edges[ci]
            c1, c2, c3, c4 = colors_arr[e1], colors_arr[e2], colors_arr[e3], colors_arr[e4]
            if len({c1, c2, c3, c4}) == 4:
                # Invalid! Fall back to all zeros
                best_colors = [0] * NUM_EDGES
                best_count = 1
                break
    
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors
# EVOLVE-BLOCK-END
