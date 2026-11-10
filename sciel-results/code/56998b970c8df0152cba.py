import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
NUM_COLORS = MAX_COLOR + 1  # 120

# Precompute edge indices for lexicographic ordering
# (i, j) with i < j
# Index = i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build all 4-cycles (rainbow checks)
# A 4-cycle is defined by 4 distinct vertices a, b, c, d.
# There are 3 distinct cycles on these 4 vertices:
# 1. a-b-c-d-a  (edges: ab, bc, cd, da)
# 2. a-b-d-c-a  (edges: ab, bd, dc, ca)
# 3. a-c-b-d-a  (edges: ac, cb, bd, da)

CYCLES = []
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]

def _build_cycles():
    global CYCLES, EDGE_TO_CYCLES
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d))
                    # Cycle 2: a-b-d-c-a
                    e2 = (_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c))
                    # Cycle 3: a-c-b-d-a
                    e3 = (_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d))
                    
                    for cyc in [e1, e2, e3]:
                        idx = len(CYCLES)
                        CYCLES.append(cyc)
                        for e in cyc:
                            EDGE_TO_CYCLES[e].append(idx)

_build_cycles()
NUM_CYCLES = len(CYCLES)

# For fast validity checking, we can check if any cycle has 4 distinct colors.
# colors[cyc[0]] != colors[cyc[1]] != colors[cyc[2]] != colors[cyc[3]] and all distinct.

def _is_rainbow(cyc, colors):
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def _validate(colors):
    """Full validation."""
    for cyc in CYCLES:
        if _is_rainbow(cyc, colors):
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _initial_block_coloring(rng):
    """
    Create a coloring based on partitioning vertices into blocks.
    Edges within the same block get color 0.
    Edges between block i and block j get a unique color based on (i, j).
    This structure tends to limit rainbow cycles because vertices in the same block
    share colors with other vertices in that block, creating collisions.
    """
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Try different numbers of blocks
    # If we use k blocks, we have k(k-1)/2 inter-block colors.
    # We want to maximize distinct colors, so we want many blocks, 
    # but too many blocks (like 16) means all edges are inter-block, 
    # and we might get rainbows.
    # Let's try k=4, 5, 6, 7, 8.
    
    best_colors = None
    best_distinct = -1
    
    for k in range(3, 10):
        # Random partition
        vertices = list(range(N))
        rng.shuffle(vertices)
        blocks = []
        for i in range(k):
            start = i * (N // k)
            end = (i + 1) * (N // k)
            if i == k - 1:
                end = N
            block = vertices[start:end]
            if block:
                blocks.append(set(block))
        
        # Map vertex to block index
        v_to_block = {}
        for idx, block in enumerate(blocks):
            for v in block:
                v_to_block[v] = idx
        
        # Assign colors
        # Color 0 for intra-block
        # Unique colors for inter-block pairs
        color_map = {}
        current_color = 1
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                color_map[(i, j)] = current_color
                current_color += 1
        
        for i in range(N):
            for j in range(i + 1, N):
                bi = v_to_block[i]
                bj = v_to_block[j]
                if bi == bj:
                    c = 0
                else:
                    key = (min(bi, bj), max(bi, bj))
                    c = color_map[key]
                colors[_edge_index(i, j)] = c
        
        if _validate(colors):
            d = _count_distinct(colors)
            if d > best_distinct:
                best_distinct = d
                best_colors = colors.copy()
    
    if best_colors is None:
        # Fallback: all zeros
        best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        
    return best_colors

def _local_search_swaps(colors, deadline, rng):
    """
    Local search using edge color swaps.
    A swap of colors on edges e1 and e2 is valid if the resulting coloring has no rainbow cycles.
    Since only cycles containing e1 or e2 can change validity, we can check only those.
    """
    current_colors = colors.copy()
    current_distinct = _count_distinct(current_colors)
    best_colors = current_colors.copy()
    best_distinct = current_distinct
    
    # Precompute list of edges for faster iteration
    edges = list(range(NUM_EDGES))
    
    while time.time() < deadline:
        # Pick a random pair of edges to swap
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
        
        c1 = current_colors[e1]
        c2 = current_colors[e2]
        if c1 == c2:
            continue
            
        # Tentatively swap
        current_colors[e1] = c2
        current_colors[e2] = c1
        
        # Check validity
        # Only cycles containing e1 or e2 need to be checked
        valid = True
        # We can use a set to avoid checking the same cycle twice if e1 and e2 share a cycle
        cycles_to_check = set(EDGE_TO_CYCLES[e1]) | set(EDGE_TO_CYCLES[e2])
        
        for ci in cycles_to_check:
            if _is_rainbow(CYCLES[ci], current_colors):
                valid = False
                break
        
        if valid:
            # Check if distinct count improves or stays same (we want to maximize distinct)
            # Note: A swap doesn't change the set of colors used, so distinct count is invariant!
            # Wait. If we swap colors, the set of colors in the graph is the same.
            # So _count_distinct remains the same.
            # This means swaps alone cannot increase the number of distinct colors.
            # We need moves that introduce new colors.
            # Let's switch to "recoloring" moves but with a smarter heuristic.
            # Or, we can use swaps to escape local minima where recoloring fails.
            # But to *increase* distinct colors, we must introduce a color not currently used,
            # or split a color into two.
            
            # Actually, the goal is to MAXIMIZE distinct colors.
            # A swap preserves the set of colors.
            # So if we start with a coloring with D distinct colors, we can never exceed D with swaps.
            # We need to use recoloring to introduce new colors.
            
            # Let's combine: 
            # 1. Recoloring to introduce new colors (greedy).
            # 2. Swaps to rearrange existing colors to allow more recoloring.
            
            # For now, accept the swap to move the state, even if distinct doesn't change.
            # This helps escape local minima for future recoloring steps.
            pass
        else:
            # Revert
            current_colors[e1] = c1
            current_colors[e2] = c2
            
        # Try to introduce a new color via recoloring
        # Pick a random edge
        e = rng.choice(edges)
        old_c = current_colors[e]
        
        # Try to assign a random new color
        new_c = rng.randint(0, MAX_COLOR)
        if new_c == old_c:
            continue
            
        # Check if new_c is already used? 
        # If we want to maximize distinct, we prefer unused colors.
        # But we must maintain validity.
        
        current_colors[e] = new_c
        valid = True
        for ci in EDGE_TO_CYCLES[e]:
            if _is_rainbow(CYCLES[ci], current_colors):
                valid = False
                break
        
        if valid:
            # Accept
            # Update distinct count
            # We need to track color counts to know if distinct changes
            # This is getting complex to track incrementally without a full count.
            # Given 120 edges, full count is fast.
            new_distinct = _count_distinct(current_colors)
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = current_colors.copy()
            # Accept the move even if distinct doesn't increase, to explore?
            # Yes, to allow future improvements.
        else:
            # Revert
            current_colors[e] = old_c
            
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s safety margin
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(42)
    
    # Multi-start
    while time.time() < deadline - 5.0:
        # 1. Generate initial block coloring
        init_colors = _initial_block_coloring(rng)
        init_distinct = _count_distinct(init_colors)
        
        if init_distinct > best_overall_distinct:
            best_overall_distinct = init_distinct
            best_overall_colors = init_colors.copy()
            
        # 2. Local search with swaps and recoloring
        local_deadline = min(time.time() + 10.0, deadline - 2.0)
        refined_colors, refined_distinct = _local_search_swaps(init_colors, local_deadline, rng)
        
        if refined_distinct > best_overall_distinct:
            best_overall_distinct = refined_distinct
            best_overall_colors = refined_colors.copy()
            
    # Final validation
    if best_overall_colors is None:
        best_overall_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        
    if not _validate(best_overall_colors):
        return [0] * NUM_EDGES
        
    return best_overall_colors.tolist()

# Run a quick self-test if executed directly
if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Distinct colors: {len(set(res))}")
    # Validate
    colors = np.array(res, dtype=np.int32)
    valid = True
    for cyc in CYCLES:
        if _is_rainbow(cyc, colors):
            valid = False
            break
    print(f"Valid: {valid}")
