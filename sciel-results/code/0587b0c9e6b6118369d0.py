import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try n from 17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                return result
        except Exception as e:
            print(f"Error for n={n}: {e}")
            continue
            
    # Fallback to seed n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    # Precompute edge indices
    edge_idx_map = {}
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edges_list)
            edge_idx_map[(i, j)] = idx
            edges_list.append((i, j))
            
    num_edges = len(edges_list)
    
    # Precompute cycles
    # A cycle is a set of 6 edges
    cycle_set = set()
    # To speed up, we can just iterate combinations of 6 vertices
    for verts in itertools.combinations(range(n), 6):
        # Generate all 60 cycles for this set of vertices
        # Fix first vertex, permute remaining 5
        # Divide by 2 for direction
        seen_cycles = set()
        v0 = verts[0]
        others = verts[1:]
        for perm in itertools.permutations(others):
            # Cycle: v0 - p0 - p1 - p2 - p3 - p4 - v0
            cycle_edges = []
            prev = v0
            for v in perm:
                u, v_sorted = sorted((prev, v))
                e_idx = edge_idx_map[(u, v_sorted)]
                cycle_edges.append(e_idx)
                prev = v
            # Close the cycle
            u, v_sorted = sorted((prev, v0))
            e_idx = edge_idx_map[(u, v_sorted)]
            cycle_edges.append(e_idx)
            
            # Canonical form: frozenset of edge indices
            # But we need to avoid duplicates from rotation/reversal
            # Using frozenset of edges is sufficient to identify the cycle
            # However, different permutations might generate the same set of edges
            # We just add to a global set
            cycle_set.add(frozenset(cycle_edges))
            
    cycles = list(cycle_set)
    num_cycles = len(cycles)
    
    # Build edge_to_cycles
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)
            
    # Convert to numpy arrays for speed
    edge_to_cycles_np = [np.array(lst) if lst else np.array([], dtype=int) for lst in edge_to_cycles]
    
    # Initialize random coloring
    colors = np.random.randint(0, 4, size=num_edges)
    
    # Initialize cycle color counts
    # cycle_color_counts[c_idx, color] = number of edges in cycle c_idx with that color
    cycle_color_counts = np.zeros((num_cycles, 4), dtype=np.int8)
    
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            cycle_color_counts[c_idx, colors[e_idx]] += 1
            
    # Calculate initial cost
    cost = (cycle_color_counts == 6).sum()
    
    if cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    best_cost = cost
    best_colors = colors.copy()
    
    # We need to be careful with time
    # Estimate steps per second.
    # Each step involves:
    # - Random choice
    # - NumPy update on ~30k elements
    # - Acceptance check
    # Should be fast.
    
    while time.time() < deadline - 5.0:
        # Pick random edge
        e_idx = np.random.randint(num_edges)
        c_old = colors[e_idx]
        
        # Pick new color
        # Choose from 0,1,2,3 excluding c_old
        c_new = np.random.randint(0, 4)
        while c_new == c_old:
            c_new = np.random.randint(0, 4)
            
        # Get cycles containing this edge
        cycle_indices = edge_to_cycles_np[e_idx]
        
        if len(cycle_indices) == 0:
            # No cycles, just change
            colors[e_idx] = c_new
            continue
            
        # Update counts
        old_counts = cycle_color_counts[cycle_indices, c_old]
        new_counts = cycle_color_counts[cycle_indices, c_new]
        
        # Calculate delta cost
        # Cost decreases if old_counts was 6
        # Cost increases if new_counts becomes 6
        # Note: new_counts is BEFORE increment
        # So if new_counts == 5, it becomes 6 -> cost +1
        # If old_counts == 6, it becomes 5 -> cost -1
        
        delta_cost = (old_counts == 6).sum() - (new_counts == 5).sum()
        
        # Acceptance
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            # Apply change
            colors[e_idx] = c_new
            cycle_color_counts[cycle_indices, c_old] -= 1
            cycle_color_counts[cycle_indices, c_new] += 1
            cost += delta_cost
            
            if cost < best_cost:
                best_cost = cost
                best_colors = colors.copy()
                
                if cost == 0:
                    return {'n': n, 'colors': colors.tolist()}
        else:
            # Reject, do nothing to colors or counts
            pass
            
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    # Return best found if valid, else None
    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    else:
        return None
