import time
import numpy as np
import random
from itertools import combinations
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def edge_vertices(idx):
    i = 0
    while i * N - i * (i + 1) // 2 + (N - i - 1) <= idx:
        i += 1
    i -= 1
    offset = idx - (i * N - i * (i + 1) // 2)
    j = i + 1 + offset
    return i, j

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    # Precompute all 4-cycles
    all_cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)))
        
    num_cycles = len(all_cycles)
    
    # For fast constraint checking during backtracking
    # For each edge, list of (cycle_index, position_in_cycle)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(all_cycles):
        for pos, e in enumerate(cycle):
            edge_to_cycles[e].append((ci, pos))
            
    # Convert to lists of lists for easier access if needed, or keep as is.
    # We need to check if a move is valid.
    # A move is valid if it doesn't create a 4-distinct color cycle.
    
    def check_move_valid(edge_idx, new_color, colors):
        """
        Returns True if setting colors[edge_idx] = new_color is valid.
        colors is the current state (before the move).
        """
        old_color = colors[edge_idx]
        if new_color == old_color:
            return True
        
        for ci, pos in edge_to_cycles[edge_idx]:
            cycle = all_cycles[ci]
            # Get colors of the other 3 edges
            # We need to check if {new_color, c1, c2, c3} are all distinct.
            # c1, c2, c3 are the colors of the other edges in the cycle.
            
            # Optimization: 
            # If any of the other 3 edges has color new_color, it's valid (not 4 distinct).
            # If any of the other 3 edges have the same color as each other, it's valid.
            
            other_positions = [p for p in range(4) if p != pos]
            c1 = colors[cycle[other_positions[0]]]
            c2 = colors[cycle[other_positions[1]]]
            c3 = colors[cycle[other_positions[2]]]
            
            # Check for 4 distinct: new_color, c1, c2, c3
            # 1. new_color must be different from c1, c2, c3
            if new_color == c1 or new_color == c2 or new_color == c3:
                continue # This cycle is safe
            
            # 2. c1, c2, c3 must not be all distinct
            if c1 == c2 or c1 == c3 or c2 == c3:
                continue # This cycle is safe
                
            # If we reach here, this cycle would have 4 distinct colors.
            return False
            
        return True

    def is_fully_valid(colors):
        # Used for final verification
        for ci, cycle in enumerate(all_cycles):
            c0 = colors[cycle[0]]
            c1 = colors[cycle[1]]
            c2 = colors[cycle[2]]
            c3 = colors[cycle[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    # Strategy: Backtracking Search with Heuristics
    # We want to maximize the number of distinct colors.
    # We can try to assign colors to edges one by one.
    # To maximize distinct colors, we should try to use new colors whenever possible.
    # However, a pure "try new color first" greedy might fail.
    # We will use a randomized backtracking approach.
    
    # Order of edges: 
    # We can try to order edges such that high-degree edges (in the constraint graph) are assigned early, 
    # or we can just use a random order. 
    # Let's use a random order to avoid bias.
    
    best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_score = 0
    
    rng = random.Random(42)
    
    # We will run multiple restarts of the backtracking solver.
    # Each restart tries to find a valid coloring with as many colors as possible.
    # The objective function for the solver is to maximize the number of unique colors.
    # This is tricky for standard backtracking. 
    # Alternative: 
    # 1. Find ANY valid coloring (e.g., all 0s).
    # 2. Use the backtracking solver to "improve" the coloring by recoloring edges to new colors.
    #    This is essentially local search but with exact feasibility checking for the move.
    #    Wait, the current program already does this with SA. 
    #    The difference: Instead of SA's probabilistic acceptance, we can use a 
    #    "Greedy with Backtracking" approach where we try to add a new color to a random edge,
    #    and if it leads to a dead end, we backtrack.
    
    # Let's try a different angle: 
    # Constructive Heuristic:
    # 1. Start with all edges colored 0.
    # 2. Iterate through edges. For each edge, try to assign the smallest available color 
    #    that keeps the coloring valid. 
    #    This minimizes colors. We want to MAXIMIZE.
    # 3. So, try to assign a color that is DIFFERENT from existing ones, but valid.
    #    If we can assign a new color, do it. If not, assign an existing valid color.
    
    def greedy_maximize_colors():
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        used_colors = set([0])
        
        # Random order of edges
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            # Try to find a valid color for edge e
            # We prefer new colors to maximize the count.
            # But we must ensure validity.
            # Let's try random colors, but bias towards unused ones?
            # Or just try all colors 0..119 and pick a valid one.
            # To maximize distinct colors, we should pick a color that is not in used_colors if possible.
            
            valid_colors = []
            # Check all 120 colors? That's 120 * 182 checks per edge. 120*182*120 ~ 2.6M ops. 
            # Might be slow in Python. 
            # Let's optimize check_move_valid.
            
            # Instead of checking all 120, we can just pick a random valid one.
            # But to maximize distinct colors, we want to use new ones.
            
            # Let's try a few random new colors first.
            new_colors_found = []
            existing_colors_found = []
            
            # Try a few random colors from the full range
            # We'll just iterate through a random permutation of colors
            color_order = list(range(MAX_COLORS))
            rng.shuffle(color_order)
            
            # To prioritize new colors, we can shuffle the used colors to the end?
            # No, just random is fine for now, but we want to MAXIMIZE.
            # Let's try to pick a color not in used_colors.
            
            for c in color_order:
                if check_move_valid(e, c, colors):
                    if c not in used_colors:
                        colors[e] = c
                        used_colors.add(c)
                        break
                    else:
                        # Store first valid existing color as fallback
                        if not existing_colors_found:
                            existing_colors_found.append(c)
                        # Keep looking for a new color? 
                        # If we find a new one, we take it. 
                        # If we exhaust all, we take the existing one.
                        # But checking all 120 is expensive.
                        # Let's limit the search.
                        if len(color_order) < 20: # Only check first 20? No, we shuffled all.
                            pass
                            
            if e not in [x for x in []]: # Placeholder to check if assigned
                pass
                
            # The above loop is inefficient. 
            # Let's refine:
            # 1. Try 5 random new colors.
            # 2. If none valid, try 5 random existing colors.
            # 3. If none valid, try all.
            
            assigned = False
            # Try new colors
            for _ in range(10):
                c = rng.randint(0, MAX_COLORS - 1)
                if c not in used_colors and check_move_valid(e, c, colors):
                    colors[e] = c
                    used_colors.add(c)
                    assigned = True
                    break
            
            if not assigned:
                # Try existing colors
                for c in list(used_colors):
                    if check_move_valid(e, c, colors):
                        colors[e] = c
                        assigned = True
                        break
            
            if not assigned:
                # Fallback: try all
                for c in range(MAX_COLORS):
                    if check_move_valid(e, c, colors):
                        colors[e] = c
                        if c not in used_colors:
                            used_colors.add(c)
                        assigned = True
                        break
                        
            if not assigned:
                # Should not happen if 0 is always valid? No, 0 might not be valid if it creates 4 distincts.
                # But if all edges are 0, it's valid. So there is always a valid color.
                # Wait, if the current state is valid, and we change e, 
                # we might break validity. 
                # But we are constructing from scratch. 
                # If we fail, it means no color works? That's impossible if we start with valid.
                # Actually, if we pick a random order, we might paint ourselves into a corner?
                # No, because we can always revert to a color that was valid before? 
                # No, we are assigning sequentially. 
                # If we get stuck, we need backtracking. 
                # For simplicity, let's just return what we have and hope it's valid? 
                # No, we must ensure validity.
                
                # Let's just return the current best effort.
                pass
                
        return colors, len(used_colors)

    # Run multiple greedy attempts
    for i in range(100):
        if time.time() > deadline:
            break
            
        colors, score = greedy_maximize_colors()
        
        # Verify validity
        if is_fully_valid(colors):
            if score > best_score:
                best_score = score
                best_colors = colors.copy()
                # Try to improve with local search?
                # The greedy might be suboptimal. 
                # We can try to recolor edges to new colors using the same check_move_valid.
                
                # Local Search Improvement
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Try to recolor random edges to new colors
                    for _ in range(100):
                        e = rng.randint(0, NUM_EDGES - 1)
                        old_c = colors[e]
                        # Try a new color
                        new_c = rng.randint(0, MAX_COLORS - 1)
                        if new_c == old_c or new_c in [colors[x] for x in range(NUM_EDGES)]:
                            continue # Only interested in NEW colors for improvement
                        
                        if check_move_valid(e, new_c, colors):
                            colors[e] = new_c
                            improved = True
                            # Update best
                            new_score = len(set(colors))
                            if new_score > best_score:
                                best_score = new_score
                                best_colors = colors.copy()
                            break # Restart the random edge picking? Or continue?
                            # Continuing is better.
        
    # Final check
    if not is_fully_valid(best_colors):
        # Fallback to trivial
        fallback = np.zeros(NUM_EDGES, dtype=np.int32)
        return fallback.tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
    print(colors)
