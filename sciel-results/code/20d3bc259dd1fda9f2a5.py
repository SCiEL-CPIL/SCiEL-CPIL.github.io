import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles
    four_cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # Three 4-cycles:
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]

        for cycle_edges in [e1, e2, e3]:
            four_cycles.append(cycle_edges)
            for e in cycle_edges:
                others = [x for x in cycle_edges if x != e]
                edge_to_cycles[e].append(others)

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        for cycle in four_cycles:
            if len(set(colors[e] for e in cycle)) == 4:
                return False
        return True

    def try_change(colors, edge, new_color):
        old_color = colors[edge]
        if old_color == new_color:
            return True
        for others in edge_to_cycles[edge]:
            cycle_colors = set()
            cycle_colors.add(new_color)
            for e in others:
                cycle_colors.add(colors[e])
            if len(cycle_colors) == 4:
                return False
        return True

    # Recursive Bisection Construction (Inspiration 1)
    def build_coloring(n):
        # Returns a dictionary mapping (local_i, local_j) to color
        # and the max color value used
        if n == 1:
            return {}, 0
        if n == 2:
            return {(0, 1): 1}, 1
        if n == 3:
            # K3. Any 3 edges. No C4. Can use 3 colors.
            return {(0, 1): 1, (0, 2): 2, (1, 2): 3}, 3
        if n == 4:
            # K4 Base Case
            # Edges:
            # (0,1): 1, (0,2): 4, (0,3): 1
            # (1,2): 2, (1,3): 4
            # (2,3): 3
            # Colors used: {1, 2, 3, 4}
            # Check cycles:
            # 0-1-2-3-0: 1, 2, 3, 1 -> {1,2,3} OK
            # 0-1-3-2-0: 1, 4, 3, 4 -> {1,3,4} OK
            # 0-2-1-3-0: 4, 2, 4, 1 -> {1,2,4} OK
            base = {
                (0, 1): 1,
                (0, 2): 4,
                (0, 3): 1,
                (1, 2): 2,
                (1, 3): 4,
                (2, 3): 3
            }
            return base, 4
        
        n1 = n // 2
        n2 = n - n1
        
        col1, max1 = build_coloring(n1)
        col2, max2 = build_coloring(n2)
        
        # Shift colors of col2 to be above max1
        shifted_col2 = {}
        for (i, j), c in col2.items():
            shifted_col2[(i, j)] = c + max1
            
        bridge_color = max1 + max2 + 1
        
        new_col = {}
        
        # Edges in first half
        for (i, j), c in col1.items():
            new_col[(i, j)] = c
            
        # Edges in second half (shift indices)
        for (i, j), c in shifted_col2.items():
            new_col[(i + n1, j + n1)] = c
            
        # Edges between halves
        for i in range(n1):
            for j in range(n1, n):
                new_col[(i, j)] = bridge_color
                
        return new_col, bridge_color

    # Generate the initial 19-coloring
    coloring_dict, max_color_used = build_coloring(N)
    
    initial_colors = [0] * NUM_EDGES
    for (i, j), c in coloring_dict.items():
        initial_colors[edge_idx(i, j)] = c
        
    best_colors = initial_colors[:]
    best_count = count_distinct(best_colors)
    
    # Local Search Refinement (Current Program Strategy)
    # Try to introduce more colors
    random.seed(42)
    
    current = best_colors[:]
    current_count = best_count
    
    # We will try to improve by changing edges to unused colors
    # The recursive coloring is likely tight, but let's check.
    
    for _ in range(100): # 100 restarts/iterations
        if time.time() > deadline - 5:
            break
            
        used_colors = set(current)
        unused_colors = [c for c in range(MAX_COLOR) if c not in used_colors]
        
        if not unused_colors:
            break
        
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        improved = False
        for edge in edges:
            if time.time() > deadline - 5:
                break
                
            old_color = current[edge]
            random.shuffle(unused_colors)
            
            for new_color in unused_colors:
                if try_change(current, edge, new_color):
                    current[edge] = new_color
                    new_count = current_count + 1
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current[:]
                    current_count = new_count
                    improved = True
                    break
        
        if not improved:
            # If no improvement in a full pass, try random perturbations
            # to escape local optima, though unlikely to help if tight.
            any_improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                edge = random.randint(0, NUM_EDGES - 1)
                new_color = random.randint(0, MAX_COLOR - 1)
                if try_change(current, edge, new_color):
                    old = current[edge]
                    current[edge] = new_color
                    new_count = len(set(current))
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current[:]
                    current_count = new_count
                    any_improved = True
                    break
            if not any_improved:
                # Stuck
                # Reset to best
                current = best_colors[:]
                current_count = best_count

    # Final Verification
    if not is_valid(best_colors):
        # Fallback to recursive construction
        best_colors = initial_colors[:]
        
    return best_colors
