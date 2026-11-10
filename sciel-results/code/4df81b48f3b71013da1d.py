"""
Strategy: Vectorized constraint precomputation + Local Search (Hill Climbing).
1. Precompute all 4-cycle edge index tuples using NumPy.
2. Start with a valid coloring (e.g., all 0s, or a simple valid pattern).
3. Iteratively pick a random edge, try changing its color to a random new value.
4. Check validity using the precomputed array (fast NumPy set operations or loop).
   To make it faster, we only check cycles involving the changed edge.
   Actually, checking all cycles is O(120 * 4) per move if optimized, but 
   precomputing "cycles containing edge e" is better.
   There are C(16,4) * 3 = 1820 * 3 = 5460 cycles.
   Each edge is in many cycles.
   
   Let's use a different fast check: 
   Maintain a set of colors. 
   When changing edge e from c_old to c_new:
   - If c_new is already used, distinct count doesn't change (unless c_old was unique and removed).
   - If c_new is new, distinct count increases.
   - Validity check: Only check cycles containing edge e.
   
   Precompute for each edge e, the list of cycle indices (or just the edge tuples) that contain e.
   There are 120 edges. Total cycle-edge incidences: 5460 * 4 = 21840.
   Average ~182 cycles per edge.
   Checking 182 cycles per move is very fast in Python if we use precomputed arrays.
"""

import numpy as np
import time
import random

N = 16
NEDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all edges
edges = []
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))

# Map edge index to (i, j)
edge_to_ij = {}
for idx, (i, j) in enumerate(edges):
    edge_to_ij[idx] = (i, j)

# Precompute all 4-cycles
# A 4-cycle is defined by 4 distinct vertices {a,b,c,d}.
# The 3 cycles are:
# 1. a-b-c-d-a : edges (a,b), (b,c), (c,d), (d,a)
# 2. a-b-d-c-a : edges (a,b), (b,d), (c,d), (a,c)
# 3. a-c-b-d-a : edges (a,c), (b,c), (b,d), (a,d)
# Note: d,a is same as a,d.

cycles = [] # List of tuples of 4 edge indices
cycles_for_edge = [[] for _ in range(NEDGES)]

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                e1 = (_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a))
                cycles.append(e1)
                for e in e1:
                    cycles_for_edge[e].append(len(cycles) - 1)
                
                # Cycle 2: a-b, b-d, c-d, a-c
                e2 = (_edge_idx(a, b), _edge_idx(b, d), _edge_idx(c, d), _edge_idx(a, c))
                cycles.append(e2)
                for e in e2:
                    cycles_for_edge[e].append(len(cycles) - 1)
                
                # Cycle 3: a-c, b-c, b-d, a-d
                e3 = (_edge_idx(a, c), _edge_idx(b, c), _edge_idx(b, d), _edge_idx(a, d))
                cycles.append(e3)
                for e in e3:
                    cycles_for_edge[e].append(len(cycles) - 1)

# Convert cycles to a numpy array for fast access if needed, but list of tuples is fine for small N
# cycles_np = np.array(cycles)

def _is_valid_fast(colors, changed_edge_idx):
    """
    Check if the coloring is valid, assuming only changed_edge_idx might have caused a violation.
    We only need to check cycles containing changed_edge_idx.
    """
    for cycle_idx in cycles_for_edge[changed_edge_idx]:
        c = cycles[cycle_idx]
        # Check if 4 distinct colors
        # Unpack for speed
        c0 = colors[c[0]]
        c1 = colors[c[1]]
        c2 = colors[c[2]]
        c3 = colors[c[3]]
        # Quick check
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Initial valid coloring: All 0s.
    # But we want to maximize distinct colors.
    # Start with a pattern that is likely valid and has some colors.
    # E.g., color(i,j) = i % 16. 
    # Let's check if color(i,j) = i is valid.
    # Cycle a-b-c-d-a: colors i, j, k, l? No, edge (a,b) color a, (b,c) color b, (c,d) color c, (d,a) color d.
    # If a,b,c,d distinct, colors are a,b,c,d which are distinct. So INVALID.
    
    # Valid pattern: color(i,j) = 0. Distinct = 1.
    # Let's try to build up.
    
    best_colors = [0] * NEDGES
    best_distinct = 1
    
    # Try a few simple valid seeds
    seeds = []
    
    # Seed 1: All 0
    seeds.append([0] * NEDGES)
    
    # Seed 2: color(i,j) = i % 2 (Bipartite-like, but not quite)
    # Let's check validity of color(i,j) = i % 2
    # Cycle a-b-c-d-a: colors a%2, b%2, c%2, d%2.
    # If a,b,c,d are 0,1,2,3 -> 0,1,0,1. Not distinct.
    # If a,b,c,d are 0,1,4,5 -> 0,1,0,1.
    # It seems color(i,j) = i % 2 is always valid?
    # Let's verify.
    # Edges from even vertices have color 0, odd vertices have color 1.
    # Cycle: e1, e2, e3, e4.
    # e1=(a,b), color a%2.
    # e2=(b,c), color b%2.
    # e3=(c,d), color c%2.
    # e4=(d,a), color d%2.
    # Can we have 4 distinct colors? No, only 2 colors available.
    # So this is valid. Distinct = 2.
    s2 = [0] * NEDGES
    for i in range(N):
        for j in range(i+1, N):
            s2[_edge_idx(i,j)] = i % 2
    seeds.append(s2)
    
    # Seed 3: color(i,j) = i % 3
    # Max distinct in cycle is 3. Valid.
    s3 = [0] * NEDGES
    for i in range(N):
        for j in range(i+1, N):
            s3[_edge_idx(i,j)] = i % 3
    seeds.append(s3)
    
    # Seed 4: color(i,j) = (i+j) % 3
    # Check validity.
    # Cycle a-b-c-d-a:
    # (a+b)%3, (b+c)%3, (c+d)%3, (d+a)%3.
    # Sum of colors = 2(a+b+c+d)%3.
    # Can they be 0,1,2,0? Sum=3=0. Possible?
    # If a=0,b=1,c=0,d=2:
    # 1, 1, 2, 2. Not distinct.
    # If a=0,b=1,c=2,d=0:
    # 1, 0, 2, 0. Distinct {0,1,2}. Size 3.
    # Since mod 3, max distinct is 3. Valid.
    s4 = [0] * NEDGES
    for i in range(N):
        for j in range(i+1, N):
            s4[_edge_idx(i,j)] = (i+j) % 3
    seeds.append(s4)
    
    # Seed 5: color(i,j) = (i*j) % 5
    # Max distinct 5. Need to check if valid.
    s5 = [0] * NEDGES
    for i in range(N):
        for j in range(i+1, N):
            s5[_edge_idx(i,j)] = (i*j) % 5
    # We should verify these seeds before using them, but for speed in construct, 
    # we'll just run local search from them. If invalid, first move will fix or we check.
    # Actually, let's just pick one good seed and run local search.
    # The local search will handle validity.
    
    # To be safe, let's pick the seed with highest distinct count that we *know* is valid.
    # All 0 is valid. i%2 is valid. i%3 is valid.
    # Let's start from i%3 (distinct 3) or i%2 (distinct 2).
    # Actually, starting from a higher distinct seed helps.
    # Let's try to generate a seed with more colors.
    # color(i,j) = i % 4?
    # Cycle a,b,c,d distinct. Colors a%4, b%4, c%4, d%4.
    # If a,b,c,d are 0,1,2,3, colors are 0,1,2,3. Distinct 4. INVALID.
    # So i%k is valid only if k < 4? No, if k=4, we can have 4 distinct.
    # So i%3 is the max for "color depends only on i".
    
    # What about color(i,j) = (i+j) % 4?
    # Cycle a,b,c,d.
    # (a+b)%4, (b+c)%4, (c+d)%4, (d+a)%4.
    # Can we get 4 distinct?
    # Try a=0,b=1,c=2,d=3.
    # 1, 3, 1, 3. No.
    # Try a=0,b=1,c=3,d=2.
    # 1, 0, 1, 2. Distinct {0,1,2}.
    # Try a=0,b=2,c=1,d=3.
    # 2, 3, 0, 3.
    # It might be valid for mod 4?
    # Let's just use local search to find out.
    
    # Strategy:
    # 1. Start with a valid seed (e.g., i%3).
    # 2. Local Search:
    #    - Pick random edge.
    #    - Pick random new color (0..119).
    #    - Check validity of cycles involving that edge.
    #    - If valid, accept.
    #    - Track distinct count.
    
    # To speed up, we can bias new colors towards "new" colors to increase distinct count.
    
    current_colors = seeds[2][:]  # i % 3
    current_distinct = _count_distinct(current_colors)
    
    if not _is_valid_fast(current_colors, -1): # Check all if -1? No, _is_valid_fast only checks one edge.
        # We need a full check for initialization
        for e in range(NEDGES):
            if not _is_valid_fast(current_colors, e):
                # If invalid, reset to all 0
                current_colors = [0] * NEDGES
                current_distinct = 1
                break
    
    best_colors = current_colors[:]
    best_distinct = current_distinct
    
    # Local Search
    # We have ~80 seconds.
    # Each move: check ~182 cycles. Very fast.
    # Let's run until deadline.
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, NEDGES - 1)
        
        # Pick new color
        # Strategy: Try to use a color not yet in the set, or a random one.
        # If we always pick random 0..119, we might not increase distinct count fast enough.
        # Let's pick from 0..119.
        new_c = random.randint(0, 119)
        
        old_c = current_colors[e]
        if new_c == old_c:
            continue
            
        current_colors[e] = new_c
        
        if _is_valid_fast(current_colors, e):
            # Calculate new distinct count
            # Optimization: 
            # If new_c was not in set, distinct increases.
            # If old_c was unique and removed, distinct decreases.
            # For simplicity, recalculate? 120 elements is fast.
            new_distinct = _count_distinct(current_colors)
            
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = current_colors[:]
            
            # Optional: Simulated Annealing? 
            # If new_distinct < current_distinct, we might want to keep it if it opens up future moves?
            # But greedy is safer for "maximize distinct".
            # Let's stick to greedy: only keep if valid. We don't revert if valid but distinct doesn't increase?
            # If we don't revert, we might get stuck in a local minimum of distinct count?
            # Actually, if we keep a valid move that doesn't increase distinct, we might block a future move?
            # No, having more colors or same colors is not necessarily better for future moves.
            # However, "random walk" on valid states is good.
            # Let's accept all valid moves.
            
        else:
            # Revert
            current_colors[e] = old_c
            
        # Periodically check time
        if time.time() > deadline - 2:
            break
            
    return best_colors

# EVOLVE-BLOCK-END
