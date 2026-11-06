import numpy as np
import time
import random

# Precomputation for n=17
N = 17
NUM_EDGES = N * (N - 1) // 2

# Map (i, j) to edge index
def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Generate all 6-cycles
# A 6-cycle is a set of 6 vertices.
# We generate all combinations of 6 vertices.
# For each combination, there are 60 distinct cycles (Hamiltonian cycles in K6).
# Actually, we just need the set of edges.
# For a set of 6 vertices, how many $C_6$ subgraphs?
# The number of $C_6$ in $K_6$ is 60.
# So for each 6-subset, we generate the 60 edge-sets.

def generate_cycles(n):
    cycles = []
    # Precompute all 6-subsets
    from itertools import combinations
    for comb in combinations(range(n), 6):
        # For each 6-subset, generate all 60 cycles
        # A cycle is a permutation of the 6 vertices, modulo rotation and reflection.
        # 6! / (6 * 2) = 60.
        # We can generate them by fixing one vertex and permuting the other 5, then dividing by 2.
        # Or just iterate all permutations and deduplicate edge sets.
        # Since 60 is small, we can just use a set of frozensets of edges.
        seen = set()
        v = comb
        # Generate all permutations of v
        from itertools import permutations
        for p in permutations(v):
            # p is a tuple of 6 vertices
            # Form edges
            edges = []
            for i in range(6):
                u = p[i]
                w = p[(i + 1) % 6]
                e = edge_index(u, w)
                edges.append(e)
            # Deduplicate
            edge_set = frozenset(edges)
            if edge_set not in seen:
                seen.add(edge_set)
                cycles.append(list(edge_set))
    return cycles

# Precompute edge_to_cycles
# edge_to_cycles[e] = list of cycle indices containing e

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try to solve for n=17
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Precomputation
    cycles = generate_cycles(n)
    num_cycles = len(cycles)
    
    # Map edge to list of cycle indices
    edge_to_cycles = [[] for _ in range(num_edges)]
    for i, c in enumerate(cycles):
        for e in c:
            edge_to_cycles[e].append(i)
    
    # Initialize colors
    # Random start
    colors = np.random.randint(0, 4, size=num_edges)
    
    def get_cost():
        cost = 0
        for c in cycles:
            # Check if all edges in c have same color
            col = colors[c[0]]
            # Optimization: break early if mismatch
            for e in c[1:]:
                if colors[e] != col:
                    break
            else:
                cost += 1
        return cost
    
    current_cost = get_cost()
    best_cost = current_cost
    best_colors = colors.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick random edge
        e = np.random.randint(0, num_edges)
        old_color = colors[e]
        
        # Pick new color
        new_color = np.random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta cost
        # We need to check cycles containing e
        delta = 0
        
        # Cycles that might become inactive (were active in old_color)
        # Cycles that might become active (will be active in new_color)
        
        # We iterate through cycles containing e
        for c_idx in edge_to_cycles[e]:
            c = cycles[c_idx]
            
            # Check if this cycle is currently monochromatic in old_color
            # We know edge e is old_color. Check others.
            is_old_mono = True
            for ee in c:
                if ee == e:
                    continue
                if colors[ee] != old_color:
                    is_old_mono = False
                    break
            
            # Check if this cycle will be monochromatic in new_color
            # We know edge e will be new_color. Check others.
            is_new_mono = True
            for ee in c:
                if ee == e:
                    continue
                if colors[ee] != new_color:
                    is_new_mono = False
                    break
            
            if is_old_mono:
                delta -= 1
            if is_new_mono:
                delta += 1
        
        # Apply change
        colors[e] = new_color
        current_cost += delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()
            
            if best_cost == 0:
                # Found a solution!
                # Validate
                if get_cost() == 0:
                    # Convert to list format for return
                    res_colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            res_colors.append(int(best_colors[edge_index(i, j)]))
                    return {'n': n, 'colors': res_colors}
        
        # Simulated Annealing acceptance
        if delta > 0:
            if random.random() < np.exp(-delta / T):
                pass # Accept
            else:
                # Reject
                colors[e] = old_color
                current_cost -= delta
        
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # If we didn't find a solution for n=17, try to salvage
    # Use the best coloring found, remove a vertex, and check
    # We'll try n=16, 15...
    # For simplicity, if we didn't find 17, we return a known valid smaller one?
    # Or we can try to repair.
    # Given the time, let's just return the best valid one we can quickly generate.
    # Actually, if we didn't find 17, we might have found a valid 16 during the process? No, we only checked 17.
    
    # Fallback: Return a valid n=16 or lower.
    # We can quickly find a valid n=16 by running a shorter SA or using a known one.
    # Let's just run a quick SA for n=16 with remaining time.
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        n_new = 16
        num_edges_new = n_new * (n_new - 1) // 2
        cycles_new = generate_cycles(n_new)
        edge_to_cycles_new = [[] for _ in range(num_edges_new)]
        for i, c in enumerate(cycles_new):
            for e in c:
                edge_to_cycles_new[e].append(i)
                
        colors_new = np.random.randint(0, 4, size=num_edges_new)
        
        def get_cost_new():
            cost = 0
            for c in cycles_new:
                col = colors_new[c[0]]
                for e in c[1:]:
                    if colors_new[e] != col:
                        break
                else:
                    cost += 1
            return cost
            
        current_cost_new = get_cost_new()
        best_cost_new = current_cost_new
        best_colors_new = colors_new.copy()
        
        T_new = 1.0
        cooling_new = 0.999
        
        while time.time() < deadline - 1.0:
            e = np.random.randint(0, num_edges_new)
            old_color = colors_new[e]
            new_color = np.random.randint(0, 4)
            if new_color == old_color:
                continue
                
            delta = 0
            for c_idx in edge_to_cycles_new[e]:
                c = cycles_new[c_idx]
                is_old_mono = True
                for ee in c:
                    if ee == e: continue
                    if colors_new[ee] != old_color:
                        is_old_mono = False
                        break
                is_new_mono = True
                for ee in c:
                    if ee == e: continue
                    if colors_new[ee] != new_color:
                        is_new_mono = False
                        break
                if is_old_mono: delta -= 1
                if is_new_mono: delta += 1
                
            colors_new[e] = new_color
            current_cost_new += delta
            
            if current_cost_new < best_cost_new:
                best_cost_new = current_cost_new
                best_colors_new = colors_new.copy()
                if best_cost_new == 0:
                    if get_cost_new() == 0:
                        res_colors = []
                        for i in range(n_new):
                            for j in range(i + 1, n_new):
                                res_colors.append(int(best_colors_new[edge_index(i, j)]))
                        return {'n': n_new, 'colors': res_colors}
                        
            if delta > 0:
                if random.random() < np.exp(-delta / T_new):
                    pass
                else:
                    colors_new[e] = old_color
                    current_cost_new -= delta
                    
            T_new *= cooling_new
            if T_new < 0.01: T_new = 0.01

    # Final fallback: Return the seed solution (n=9) to ensure score > 0
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
