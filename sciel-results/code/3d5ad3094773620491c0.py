import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
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
    
    # For each edge, list of (cycle_index, position_in_cycle)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(all_cycles):
        for pos, e in enumerate(cycle):
            edge_to_cycles[e].append((ci, pos))
            
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

    def greedy_maximize_colors(seed):
        rng = random.Random(seed)
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        used_colors = set([0])
        
        # Random order of edges
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            assigned = False
            # Try new colors first to maximize distinct count
            # We pick random new colors from the unused set
            unused = [c for c in range(MAX_COLORS) if c not in used_colors]
            rng.shuffle(unused)
            
            # Try a few random new colors
            for c in unused[:10]:
                if check_move_valid(e, c, colors):
                    colors[e] = c
                    used_colors.add(c)
                    assigned = True
                    break
            
            if not assigned:
                # Try existing colors
                existing = list(used_colors)
                rng.shuffle(existing)
                for c in existing:
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
                        
        return colors, len(used_colors)

    def local_search_improve(colors, rng, max_steps=1000):
        """
        Try to increase the number of distinct colors by recoloring edges to new colors.
        """
        improved = True
        steps = 0
        current_score = len(set(colors))
        
        while improved and steps < max_steps and time.time() < deadline:
            improved = False
            steps += 1
            
            # Try to recolor a random edge to a new color
            # We want to find an edge and a new color such that:
            # 1. The move is valid.
            # 2. The new color is not currently used.
            
            # Strategy: Pick a random edge. Check if it can take a new color.
            # If yes, do it.
            # To be more effective, we might prioritize edges that are "constrained" 
            # or just random.
            
            # Let's try all edges in random order? No, that's slow.
            # Random edge, random new color.
            
            e = rng.randint(0, NUM_EDGES - 1)
            old_c = colors[e]
            
            # Find a new color
            # We can't just pick random 0..119 and check if it's used, 
            # because if it IS used, it's not a "new" color for our objective.
            # We want to increase the count of distinct colors.
            
            used = set(colors)
            new_candidates = [c for c in range(MAX_COLORS) if c not in used]
            
            if not new_candidates:
                break # All colors used, max possible
            
            rng.shuffle(new_candidates)
            
            # Try a few new colors for this edge
            for new_c in new_candidates[:5]:
                if check_move_valid(e, new_c, colors):
                    colors[e] = new_c
                    improved = True
                    break
            
            # If we improved, the score increases by 1.
            # We don't need to recalculate score every time if we trust the logic,
            # but let's keep it simple.
        
        return colors, len(set(colors))

    best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_score = 0
    
    # Phase 1: Greedy Construction
    # Run multiple greedy attempts with different seeds
    num_greedy_attempts = 50
    for i in range(num_greedy_attempts):
        if time.time() > deadline:
            break
            
        colors, score = greedy_maximize_colors(i)
        
        if is_fully_valid(colors):
            if score > best_score:
                best_score = score
                best_colors = colors.copy()
                
                # Phase 2: Local Search Improvement
                # Use a specific RNG seed for reproducibility of the improvement phase
                # or just continue with the same logic.
                # Let's use a fresh RNG for the improvement to decorrelate.
                imp_rng = random.Random(1000 + i)
                improved_colors, imp_score = local_search_improve(colors.copy(), imp_rng, max_steps=200)
                
                if imp_score > best_score:
                    best_score = imp_score
                    best_colors = improved_colors.copy()
                    
                # If we hit the theoretical max or close to it, stop early?
                # Max possible is 120. But constraints might limit it.
                # If best_score > 40, we are doing well.
                
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
