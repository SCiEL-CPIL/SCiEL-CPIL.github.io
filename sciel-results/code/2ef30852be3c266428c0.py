import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute for each edge, which 4-sets it participates in
EDGE_4SETS = [[] for _ in range(NUM_EDGES)]
# Precompute the 3 cycle edge-triples for each 4-set
FOUR_SETS = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    e_ab = edge_idx(a, b)
    e_ac = edge_idx(a, c)
    e_ad = edge_idx(a, d)
    e_bc = edge_idx(b, c)
    e_bd = edge_idx(b, d)
    e_cd = edge_idx(c, d)
    cycles = [(e_ab, e_bc, e_cd, e_ad),
              (e_ab, e_bd, e_cd, e_ac),
              (e_ac, e_bc, e_bd, e_ad)]
    FOUR_SETS.append((quad, cycles))
    for e in [e_ab, e_ac, e_ad, e_bc, e_bd, e_cd]:
        EDGE_4SETS[e].append(len(FOUR_SETS) - 1)

def is_rainbow(c1, c2, c3, c4):
    return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

def check_edge_valid(colors, e_idx, new_color):
    """Check if setting edge e_idx to new_color violates any rainbow 4-cycle."""
    for fs_idx in EDGE_4SETS[e_idx]:
        quad, cycles = FOUR_SETS[fs_idx]
        for cyc in cycles:
            if e_idx in cyc:
                pos = cyc.index(e_idx)
                c = [new_color if p == pos else colors[cyc[p]] for p in range(4)]
                if is_rainbow(c[0], c[1], c[2], c[3]):
                    return False
    return True

def backtracking_search(edge_order, max_new_colors, deadline, start_color=15):
    """
    Backtracking search that assigns colors to edges in edge_order.
    Tries new colors first, then existing colors.
    Returns (colors, num_new_colors_used) or None if time exceeded.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    next_color = start_color
    new_colors_used = 0
    
    def _bt(pos):
        nonlocal next_color, new_colors_used
        if time.time() > deadline:
            return False
        if pos == len(edge_order):
            return True
        
        e = edge_order[pos]
        old_color = colors[e]
        
        # Try new color first (maximize diversity)
        if next_color < 120 and check_edge_valid(colors, e, next_color):
            colors[e] = next_color
            nc = next_color
            next_color += 1
            new_colors_used += 1
            if _bt(pos + 1):
                return True
            next_color = nc
            new_colors_used -= 1
            colors[e] = old_color
        
        # Try existing colors (to unblock future edges)
        existing = set()
        for other_e in edge_order[pos+1:]:
            existing.add(colors[other_e])
        existing.discard(old_color)
        
        # Also include colors already assigned to earlier edges
        for prev_e in edge_order[:pos]:
            existing.add(colors[prev_e])
        existing.discard(old_color)
        
        # Try a few existing colors randomly
        for c in random.sample(list(existing), min(3, len(existing))):
            if check_edge_valid(colors, e, c):
                colors[e] = c
                if _bt(pos + 1):
                    return True
                colors[e] = old_color
        
        return False
    
    # Only search over a subset of edges (those most constrained)
    # to keep the search tractable
    search_edges = edge_order[:min(40, len(edge_order))]
    
    if _bt(0):
        return colors, new_colors_used
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Start with a valid Gallai coloring: color(i,j) = min(i,j)
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    best_count = len(set(colors))
    best_result = colors[:]
    
    # Precompute constraint degree for each edge (number of 4-sets it's in)
    constraint_deg = [len(EDGE_4SETS[e]) for e in range(NUM_EDGES)]
    
    # Sort edges by constraint degree (most constrained first for MRV)
    edges_by_constraint = sorted(range(NUM_EDGES), key=lambda e: -constraint_deg[e])
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a randomized edge order biased toward high-constraint edges
        edge_order = edges_by_constraint[:]
        # Add some randomness: shuffle within constraint groups
        random.shuffle(edge_order)
        # But keep high-constraint edges somewhat prioritized
        # Use a weighted shuffle: high-constraint edges more likely to come early
        for i in range(len(edge_order)):
            if random.random() < 0.3:
                # Swap with a random position
                j = random.randint(i, len(edge_order) - 1)
                edge_order[i], edge_order[j] = edge_order[j], edge_order[i]
        
        # Run backtracking search
        result = backtracking_search(edge_order, 120, deadline)
        
        if result is not None:
            new_colors, nc = result
            cur_count = len(set(new_colors))
            if cur_count > best_count:
                best_count = cur_count
                best_result = new_colors[:]
        
        # Also try a greedy pass on top of the base coloring for quick wins
        greedy_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                greedy_colors[edge_idx(i, j)] = i
        next_c = 15
        for e in edge_order:
            if time.time() > deadline - 3:
                break
            if check_edge_valid(greedy_colors, e, next_c):
                greedy_colors[e] = next_c
                next_c += 1
        gc = len(set(greedy_colors))
        if gc > best_count:
            best_count = gc
            best_result = greedy_colors[:]
    
    return best_result

if __name__ == "__main__":
    c = construct()
    print(len(set(c)))
