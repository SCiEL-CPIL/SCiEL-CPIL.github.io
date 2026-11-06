import time
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    """Map (i,j) with i<j to linear index 0..119."""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _all_four_cycles():
    """Yield all 4-vertex subsets; for each, the three possible 4-cycles (as 4 edge indices)."""
    cycles = []
    verts = list(range(N))
    for quad in itertools.combinations(verts, 4):
        a, b, c, d = quad
        # Three cycle structures on {a,b,c,d}:
        # 1) a-b-c-d-a
        e1 = (_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a))
        # 2) a-b-d-c-a
        e2 = (_edge_index(a,b), _edge_index(b,d), _edge_index(d,c), _edge_index(c,a))
        # 3) a-c-b-d-a
        e3 = (_edge_index(a,c), _edge_index(c,b), _edge_index(b,d), _edge_index(d,a))
        cycles.append((e1, e2, e3))
    return cycles

# Precompute all cycles
_ALL_CYCLES = _all_four_cycles()
NUM_CYCLES = len(_ALL_CYCLES)  # 3 * C(16,4) = 3 * 1820 = 5460

def _check_validity(colors):
    """Check if no four-cycle has four distinct colors. Returns True if valid."""
    for cycles in _ALL_CYCLES:
        for cyc in cycles:
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if len({c0, c1, c2, c3}) == 4:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _violations(colors):
    """Count number of rainbow 4-cycles."""
    count = 0
    for cycles in _ALL_CYCLES:
        for cyc in cycles:
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if len({c0, c1, c2, c3}) == 4:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_score = 0  # number of distinct colors used
    
    # Try a structured initial approach: use few colors in a pattern
    # Strategy: assign colors based on a partition of vertices
    # If we partition vertices into groups and use group-pair colors,
    # then any 4-cycle uses at most 4 group-pair colors, but we need to ensure
    # no 4-cycle has 4 distinct ones.
    
    # Approach: Use a small number of colors initially, then try to increase.
    # Key insight: if we use at most 3 colors total, no 4-cycle can have 4 distinct.
    # So start with 3 colors, valid, score=3.
    
    # Better: try to find colorings with many colors.
    # The constraint is: for every 4-cycle, at least two edges share a color.
    
    # Randomized local search approach:
    # 1. Start with random assignment
    # 2. Iteratively fix violations by changing edge colors
    # 3. Try to increase color count by perturbing
    
    rng = random.Random(42)
    
    # Initialize with a valid coloring using few colors
    # Use 2 colors: even/odd edge index
    colors = [i % 2 for i in range(NUM_EDGES)]
    # This is valid (max 2 colors per cycle)
    best_colors = colors[:]
    best_score = _count_colors(colors)
    
    # Now try to increase colors via local search
    # Phase 1: Try to find a valid coloring with many colors using simulated annealing-like approach
    
    # Better strategy: start with a structure that allows many colors
    # Use a "block" structure: partition vertices into 4 groups of 4
    # Color edges within same block with one color, between blocks with another
    # But this gives only 2 colors.
    
    # Let's try: assign colors randomly, then repair
    def _repair(colors, max_iters=50000):
        """Try to fix all violations by changing individual edge colors."""
        colors = colors[:]
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            vcount = _violations(colors)
            if vcount == 0:
                return colors
            # Pick a random violation
            # Find a random violating cycle
            while True:
                cycles = rng.choice(_ALL_CYCLES)
                cyc = rng.choice(cycles)
                c0 = colors[cyc[0]]
                c1 = colors[cyc[1]]
                c2 = colors[cyc[2]]
                c3 = colors[cyc[3]]
                if len({c0, c1, c2, c3}) == 4:
                    break
            # Change one edge in this cycle to match another
            # Pick which edge to change
            idx = rng.choice(cyc)
            # Set it to the color of another edge in the cycle
            other_idx = rng.choice([x for x in cyc if x != idx])
            colors[idx] = colors[other_idx]
        return colors
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization with limited color palette
        # Use a palette of size k
        k = rng.randint(3, 15)
        palette = list(range(k))
        colors = [rng.choice(palette) for _ in range(NUM_EDGES)]
        
        # Repair
        colors = _repair(colors, max_iters=20000)
        
        if _violations(colors) == 0:
            score = _count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
        
        # Try to increase colors: pick a random edge and try a new color
        if _violations(colors) == 0:
            for _ in range(5000):
                if time.time() > deadline - 5.0:
                    break
                idx = rng.randint(0, NUM_EDGES - 1)
                old_color = colors[idx]
                new_color = rng.randint(0, 119)
                if new_color == old_color:
                    continue
                colors[idx] = new_color
                if _violations(colors) == 0:
                    score = _count_colors(colors)
                    if score > best_score:
                        best_score = score
                        best_colors = colors[:]
                else:
                    colors[idx] = old_color
    
    # Final: try greedy color increase on best
    if time.time() < deadline - 2.0:
        colors = best_colors[:]
        for idx in range(NUM_EDGES):
            if time.time() > deadline - 1.0:
                break
            old_color = colors[idx]
            # Try all colors from 0 to 119, prefer new colors
            for new_color in list(range(120, 0, -1)) + list(range(0, 120)):
                if new_color == old_color:
                    continue
                colors[idx] = new_color
                if _violations(colors) == 0:
                    score = _count_colors(colors)
                    if score > best_score:
                        best_score = score
                        best_colors = colors[:]
                    break
            colors[idx] = old_color  # revert
    
    # Ensure we return a valid coloring
    if _violations(best_colors) != 0:
        # Fallback: all zeros
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    return best_colors
