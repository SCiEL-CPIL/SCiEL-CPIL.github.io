import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
# OTHER_EDGES[e] = list of tuples (o1, o2, o3)
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))


def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (60s limit)
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    
    # Multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        rng = random.Random(1000 + restart)
        
        # Initialization: Use a simple pattern to get a valid start
        # c_ij = (i * 7 + j * 13) % 120 might have violations.
        # Let's start with all 0s and greedily add colors.
        colors = [0] * NUM_EDGES
        color_counts = [0] * (MAX_COLOR + 1)
        color_counts[0] = NUM_EDGES
        num_colors = 1
        unused_colors = list(range(1, MAX_COLOR + 1))
        
        # Greedy phase: Try to assign new colors to random edges
        edges = list(range(NUM_EDGES))
        rng.shuffle(edges)
        
        for e in edges:
            if time.time() > deadline - 5.0:
                break
            if not unused_colors:
                break
            
            # Try a few random unused colors
            candidates = rng.sample(unused_colors, min(5, len(unused_colors)))
            for c in candidates:
                if check_edge_change(colors, e, c):
                    # Apply change
                    old_c = colors[e]
                    colors[e] = c
                    color_counts[old_c] -= 1
                    if color_counts[old_c] == 0:
                        # old_c becomes unused
                        # We don't strictly need to add it to unused_colors immediately
                        # but it's good for consistency if we want to reuse it.
                        # However, we are trying to MAXIMIZE colors, so we prefer unused.
                        pass 
                    color_counts[c] += 1
                    if color_counts[c] == 1:
                        # c was unused, now used. Remove from unused list.
                        try:
                            unused_colors.remove(c)
                        except ValueError:
                            pass
                        num_colors += 1
                    break
        
        current_score = num_colors
        if current_score > best_score:
            best_score = current_score
            best_colors = list(colors)
        
        # Local Search Phase: Try to add more colors or swap to enable more colors
        # We focus on adding new colors.
        # If we are stuck, we might need to recolor an edge to a different *used* color
        # to free up constraints for a new color?
        # Or simply try random edges with random unused colors again.
        
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 0.5:
            # Fast local search loop
            # We will try to pick an edge and an unused color.
            # If valid, accept.
            # If not, try a different edge or color.
            
            # To speed up, we can pre-select a set of "active" edges?
            # No, just random.
            
            # We need to be careful with `unused_colors` list updates.
            # `remove` is O(N). N=120. This is okay.
            
            while time.time() < start_time + remaining_time:
                if not unused_colors:
                    break
                
                # Pick random edge
                e = rng.randint(0, NUM_EDGES - 1)
                old_c = colors[e]
                
                # Pick random unused color
                c = rng.choice(unused_colors)
                
                if check_edge_change(colors, e, c):
                    # Accept
                    colors[e] = c
                    color_counts[old_c] -= 1
                    if color_counts[old_c] == 0:
                        # old_c is now unused. Add to list?
                        # It might be useful to reuse it later?
                        # But we want to maximize DISTINCT colors.
                        # If old_c count becomes 0, it's unused.
                        # We should add it to unused_colors so we can try to use it again?
                        # No, if we use it again, num_colors doesn't increase.
                        # We only increase num_colors if we use a color that was previously unused.
                        # So if old_c becomes 0, it becomes unused.
                        # We should add it to unused_colors.
                        unused_colors.append(old_c)
                    
                    color_counts[c] += 1
                    if color_counts[c] == 1:
                        # c was unused, now used.
                        # Remove from unused_colors.
                        try:
                            unused_colors.remove(c)
                        except ValueError:
                            pass
                        num_colors += 1
                        
                        if num_colors > best_score:
                            best_score = num_colors
                            best_colors = list(colors)
                else:
                    # Reject. Do nothing.
                    pass
    
    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
        best_score = 1

    return list(best_colors)
