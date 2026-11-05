"""Optimized coloring search for K16 with no rainbow C4."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge indices
def edge_idx(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + j - i - 1

_edge_idx = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[i][j] = edge_idx(i, j)
        _edge_idx[j][i] = _edge_idx[i][j]

# Build all 4-cycles
def build_four_cycles():
    cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        cycles.append((_edge_idx[a][b], _edge_idx[b][c], _edge_idx[c][d], _edge_idx[d][a]))
        cycles.append((_edge_idx[a][b], _edge_idx[b][d], _edge_idx[d][c], _edge_idx[c][a]))
        cycles.append((_edge_idx[a][c], _edge_idx[c][b], _edge_idx[b][d], _edge_idx[d][a]))
    return cycles

FOUR_CYCLES = build_four_cycles()
NUM_CYCLES = len(FOUR_CYCLES)

# For each edge, store the cycles it is in, and for each cycle, the other 3 edges
# edge_cycle_info[e] = list of (cycle_id, [other_edge_1, other_edge_2, other_edge_3])
edge_cycle_info = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(FOUR_CYCLES):
    e0, e1, e2, e3 = cycle
    edge_cycle_info[e0].append((ci, [e1, e2, e3]))
    edge_cycle_info[e1].append((ci, [e0, e2, e3]))
    edge_cycle_info[e2].append((ci, [e0, e1, e3]))
    edge_cycle_info[e3].append((ci, [e0, e1, e2]))

def is_valid_change(colors, edge, new_color):
    """Check if setting edge to new_color keeps all cycles valid."""
    for ci, other_edges in edge_cycle_info[edge]:
        c1 = colors[other_edges[0]]
        c2 = colors[other_edges[1]]
        c3 = colors[other_edges[2]]
        
        # If the other 3 are all distinct, new_color must match one of them
        if c1 != c2 and c1 != c3 and c2 != c3:
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    random.seed(12345)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Phase 1: Greedy build
    # We want to maximize distinct colors.
    # Start with all 0s.
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    # Try to assign new colors to edges in a random order
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e in edge_order:
        if time.time() > deadline:
            break
        
        # Try to find a new color (not in used_colors) that is valid
        # Or an existing color that is valid?
        # To maximize count, we prefer new colors.
        # But if we use a new color, we might block future edges.
        # Greedy: try new colors first.
        
        # Generate a list of candidate colors
        # Unused colors are 0..119 not in used_colors
        # We can just pick a random unused color and check.
        # If it fails, pick a random used color?
        # Actually, if we pick a used color, the count doesn't increase.
        # So we should only accept a used color if it's the only option?
        # No, we can always keep the current color (which is valid).
        # So we should only change if we can increase the count or if we are stuck?
        # Wait, the current color is valid. Changing to a used color is neutral for count.
        # Changing to a new color is +1 for count.
        # So we should ONLY change to a new color if it's valid.
        # If no new color is valid, keep the current one.
        
        found_new = False
        # Try a few random new colors
        for _ in range(20):
            # Pick a random color not in used_colors
            # To avoid O(N) check, we can just pick random 0..119 and check set
            c = random.randint(0, MAX_COLOR)
            if c in used_colors:
                continue
            if is_valid_change(colors, e, c):
                colors[e] = c
                used_colors.add(c)
                found_new = True
                break
        
        # If no new color worked, we might try to change to a different used color
        # to potentially help other edges?
        # This is a "swap" idea. But for now, let's stick to adding new colors.
        # If we are stuck, we might need to backtrack or swap.
        # For now, just keep current color.
        
    cur_count = count_colors(colors)
    if cur_count > best_count:
        best_count = cur_count
        best_colors = colors[:]
        used_colors = set(best_colors)

    # Phase 2: Local Search
    # Strategy: Try to change an edge to a NEW color.
    # If successful, count increases.
    # If we can't find any edge that can take a new color, we are stuck.
    # To escape, we might need to change an edge to an EXISTING color,
    # which might allow another edge to take a NEW color.
    # This is a 2-move improvement.
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            colors = best_colors[:]
            used_colors = set(colors)
        else:
            # Perturb best
            colors = best_colors[:]
            used_colors = set(colors)
            # Randomly change some edges to random colors (valid or not? must be valid)
            # If we change to invalid, we break. So we must ensure validity.
            # Let's just do random valid moves.
            for _ in range(random.randint(5, 20)):
                e = random.randint(0, NUM_EDGES - 1)
                # Pick a random color
                c = random.randint(0, MAX_COLOR)
                if is_valid_change(colors, e, c):
                    old_c = colors[e]
                    colors[e] = c
                    # Update used_colors
                    # We don't strictly need used_colors for validity, only for counting.
                    # But for "new color" selection, it helps.
                    # Let's just recompute used_colors if we accept a move that changes the set.
                    # Or just maintain it carefully.
                    if c not in used_colors:
                        used_colors.add(c)
                    # If old_c is no longer used, we should remove it?
                    # It's expensive to check if old_c is still used.
                    # Let's just recompute used_colors from scratch every restart or periodically.
                    # Or, since we are perturbing, let's just recompute after perturbation.
            used_colors = set(colors)
            
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
            used_colors = set(best_colors)
        
        # Local search loop
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            e = random.randint(0, NUM_EDGES - 1)
            
            # Try to introduce a new color
            # Sample from unused colors
            # To avoid generating a list of unused colors every time (O(N)),
            # we can just pick a random color and check if it's in used_colors.
            # If it is, retry or pick a used color?
            # We want NEW colors.
            
            # Pick a random color
            new_c = random.randint(0, MAX_COLOR)
            
            # If it's already used, it's not "new".
            # But maybe we should try to change to a used color to facilitate a swap?
            # For now, let's focus on adding new colors.
            if new_c in used_colors:
                # It's not a new color.
                # We could skip, or try to change to it if it helps?
                # Changing to a used color is neutral for count.
                # Let's skip for now to focus on increasing count.
                # But if we skip too often, we might not explore the space.
                # Let's try a few new colors.
                found = False
                for _ in range(5):
                    nc = random.randint(0, MAX_COLOR)
                    if nc in used_colors:
                        continue
                    if is_valid_change(colors, e, nc):
                        colors[e] = nc
                        used_colors.add(nc)
                        cur_count += 1
                        if cur_count > best_count:
                            best_count = cur_count
                            best_colors = colors[:]
                            used_colors = set(best_colors)
                        found = True
                        break
                if found:
                    continue
                # If no new color found for this edge, try next edge
                continue
            
            # If new_c is not in used_colors, it's a candidate for a new color
            if is_valid_change(colors, e, new_c):
                colors[e] = new_c
                used_colors.add(new_c)
                cur_count += 1
                if cur_count > best_count:
                    best_count = cur_count
                    best_colors = colors[:]
                    used_colors = set(best_colors)
            else:
                # Try to change to an existing color to "shuffle" the configuration
                # This might allow other edges to take new colors later.
                # Pick a random existing color
                if used_colors:
                    old_c = colors[e]
                    new_c2 = random.choice(list(used_colors))
                    if new_c2 != old_c and is_valid_change(colors, e, new_c2):
                        colors[e] = new_c2
                        # Count doesn't change, but structure does.
                        # We accept this move to explore the neighborhood.
                        # We don't update best_count.
                        # But we should update used_colors?
                        # If old_c is no longer used, remove it.
                        # This is tricky.
                        # Let's just recompute used_colors every 100 steps?
                        # Or just keep it as a superset?
                        # If we keep it as a superset, we might think a color is "used" when it's not.
                        # This prevents us from trying it as a "new" color.
                        # That's safe (we might miss some opportunities) but not incorrect.
                        # However, if we add a color to used_colors, and then it's no longer used,
                        # we never try it as "new" again.
                        # This is bad.
                        # So we MUST maintain used_colors accurately.
                        # Checking if old_c is still used is O(N).
                        # Let's do it.
                        if old_c not in colors:
                            used_colors.remove(old_c)
                        # new_c2 is already in used_colors
    
    # Phase 3: Final Polish
    # Try to add colors to specific edges
    if time.time() < deadline - 2.0:
        colors = best_colors[:]
        used_colors = set(colors)
        cur_count = best_count
        
        for e in range(NUM_EDGES):
            if time.time() > deadline:
                break
            # Try all unused colors
            # To avoid iterating 0..119, we can just try a few random unused ones
            # Or iterate all if we are close to the limit?
            # Let's iterate all unused colors for this edge.
            # There are at most 119 - cur_count unused colors.
            # If cur_count is high, this is fast.
            unused_list = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            random.shuffle(unused_list)
            for c in unused_list:
                if is_valid_change(colors, e, c):
                    colors[e] = c
                    used_colors.add(c)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_colors = colors[:]
                    break # Only one new color per edge in this pass
    
    return best_colors

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
