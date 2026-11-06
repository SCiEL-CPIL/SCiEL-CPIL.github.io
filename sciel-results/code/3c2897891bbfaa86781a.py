# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import defaultdict

N = 16
E = 120

_edge_idx = {}
_eid = 0
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[(i, j)] = _eid
        _eid += 1

# Precompute all C4 cycles
all_cycles = []
cycles_of_edge = [[] for _ in range(E)]

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Three C4s on vertices {a,b,c,d}
                for (v1, v2, v3, v4) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                    e1 = _edge_idx[(min(v1,v2), max(v1,v2))]
                    e2 = _edge_idx[(min(v2,v3), max(v2,v3))]
                    e3 = _edge_idx[(min(v3,v4), max(v3,v4))]
                    e4 = _edge_idx[(min(v4,v1), max(v4,v1))]
                    cyc = (e1, e2, e3, e4)
                    all_cycles.append(cyc)
                    for e in (e1, e2, e3, e4):
                        others = tuple(x for x in (e1, e2, e3, e4) if x != e)
                        cycles_of_edge[e].append(others)

def count_violations(colors):
    v = 0
    for (e1, e2, e3, e4) in all_cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            v += 1
    return v

def is_valid_change(colors, edge, new_color):
    for (o1, o2, o3) in cycles_of_edge[edge]:
        if len({new_color, colors[o1], colors[o2], colors[o3]}) == 4:
            return False
    return True

def construct():
    deadline = time.time() + 55
    rng = random.Random(12345)
    
    # Strategy: Block decomposition
    # Partition 16 vertices into 4 groups of 4: G0={0,1,2,3}, G1={4,5,6,7}, G2={8,9,10,11}, G3={12,13,14,15}
    # Color edges based on (group_i, group_j) pair + local structure
    # Within-group edges: 4 groups * C(4,2) = 4*6 = 24 edges
    # Between-group edges: C(4,2) * 4*4 = 6*16 = 96 edges
    # Total: 120 edges
    
    # Key insight: If we color all edges between groups Gi and Gj with colors from a small palette Pij,
    # and ensure that for any C4, the colors don't all differ, we can use many colors.
    
    # Approach: Use a "ranked" coloring
    # Color edge (i,j) with a color determined by a hash of the structure
    # Start with a valid base coloring, then greedily add colors
    
    # Base: color by (i//4, j//4) block pair, with some variation
    # Blocks: 0,1,2,3
    # Block pair (a,b) for a<b: assign a base color
    # Within block: assign another set of colors
    
    colors = [0] * E
    
    # Assign colors based on block structure
    # Block pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) -> 6 pairs
    # Within blocks: 4 blocks * 6 edges = 24
    
    # Use distinct color ranges for different block pairs to maximize colors
    # Block pair (a,b): colors start at a*10 + b (roughly)
    # Within block k: colors start at 60 + k*6
    
    block_color_offset = {}
    pair_idx = 0
    for a in range(4):
        for b in range(a+1, 4):
            block_color_offset[(a,b)] = pair_idx * 10
            pair_idx += 1
    
    for i in range(N):
        for j in range(i+1, N):
            bi, bj = i // 4, j // 4
            if bi == bj:
                # Within block: use local edge index
                # Edges within block bi: (bi*4+x, bi*4+y) for x<y
                x, y = i - bi*4, j - bi*4
                local_idx = x * 4 + y  # 0..10 for 6 edges
                colors[_edge_idx[(i,j)]] = 60 + bi * 6 + local_idx
            else:
                a, b = min(bi, bj), max(bi, bj)
                # Between blocks: use a base color + variation
                # To keep C4-free, use limited colors per block pair
                # Use 2 colors per block pair based on local structure
                xi, xj = i - bi*4, j - bj*4
                # Use (xi + xj) % 2 to pick between 2 colors
                c = block_color_offset[(a,b)] + ((xi + xj) % 2)
                colors[_edge_idx[(i,j)]] = c
    
    # Verify and count
    distinct = len(set(colors))
    violations = count_violations(colors)
    
    best_colors = colors[:]
    best_count = distinct if violations == 0 else 0
    
    # Now greedily try to add more colors
    used_colors = set(colors)
    
    # Try to recolor edges to introduce new colors
    for target in range(120):
        if time.time() > deadline - 10:
            break
        if target in used_colors:
            continue
        # Try to assign this color to some edge
        found = False
        edges_to_try = list(range(E))
        rng.shuffle(edges_to_try)
        for e in edges_to_try:
            if colors[e] == target:
                continue
            if is_valid_change(colors, e, target):
                colors[e] = target
                used_colors.add(target)
                found = True
                break
        if found:
            distinct = len(used_colors)
            v = count_violations(colors)
            if v == 0 and distinct > best_count:
                best_count = distinct
                best_colors = colors[:]
    
    # Local search: try to swap colors to free up space for new ones
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Try random valid recoloring moves
        e = rng.randint(0, E-1)
        old_c = colors[e]
        new_c = rng.randint(0, 119)
        if new_c != old_c and is_valid_change(colors, e, new_c):
            colors[e] = new_c
            # Update used colors
            # Check if old_c is still used
            still_used = False
            for e2 in range(E):
                if e2 != e and colors[e2] == old_c:
                    still_used = True
                    break
            if not still_used:
                used_colors.discard(old_c)
            used_colors.add(new_c)
            
            v = count_violations(colors)
            if v == 0:
                d = len(used_colors)
                if d > best_count:
                    best_count = d
                    best_colors = colors[:]
    
    # Final verification
    v = count_violations(best_colors)
    if v != 0:
        # Fallback to a known valid coloring
        fallback = [0] * E
        for i in range(N):
            for j in range(i+1, N):
                fallback[_edge_idx[(i,j)]] = min(i, j)
        return fallback
    
    return best_colors
# EVOLVE-BLOCK-END
