import time
import random
import numpy as np
from collections import defaultdict

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

def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations

def get_forbidden_colors(colors, edge_idx):
    """
    Returns a set of colors that CANNOT be assigned to edge_idx 
    without creating a rainbow cycle, given the current coloring of other edges.
    """
    forbidden = set()
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        # If the other 3 edges already have a repeated color, 
        # then ANY color assigned to edge_idx will result in a repeat 
        # (either with the new color or within the existing 3). 
        # Wait, the constraint is "No four-cycle may have four distinct edge colors".
        # If c1, c2, c3 are not all distinct, then even if new_color is distinct from them, 
        # the set {new_color, c1, c2, c3} has size <= 3, so it's not a rainbow cycle.
        # So if c1, c2, c3 have a duplicate, no color is forbidden by THIS cycle.
        if c1 == c2 or c1 == c3 or c2 == c3:
            continue
        
        # If c1, c2, c3 are all distinct, then assigning new_color = c1, c2, or c3 
        # would create a rainbow cycle (4 distinct colors).
        # Actually, if new_color is any of c1, c2, c3, the set of colors is {c1, c2, c3} (size 3), 
        # which is NOT 4 distinct. 
        # Wait. 
        # Rainbow cycle = 4 distinct colors.
        # Current colors on other 3 edges: {c1, c2, c3}.
        # If I assign new_color to edge_idx:
        # The set of colors on the cycle is {new_color, c1, c2, c3}.
        # This set has size 4 iff new_color is distinct from c1, c2, and c3, 
        # AND c1, c2, c3 are distinct from each other.
        
        # So, if c1, c2, c3 are distinct:
        # Forbidden colors are those NOT in {c1, c2, c3}? 
        # No. 
        # If new_color == c1, set is {c1, c2, c3} (size 3). Valid.
        # If new_color == 99 (not in {c1,c2,c3}), set is {99, c1, c2, c3} (size 4). INVALID.
        
        # Therefore, if c1, c2, c3 are distinct, 
        # the forbidden colors are ALL colors EXCEPT c1, c2, c3.
        # This is a huge set (117 colors).
        
        # This implies that for any edge participating in a cycle with 3 distinct other colors,
        # it can ONLY take one of those 3 colors to remain valid.
        
        # Let's re-read the constraint carefully.
        # "No four-cycle may have four distinct edge colors."
        
        # So, for a specific cycle (e1, e2, e3, e4):
        # If I change e1 to color X:
        # Let other colors be c2, c3, c4.
        # Violation occurs iff {X, c2, c3, c4} has size 4.
        # This happens iff X != c2, X != c3, X != c4, AND c2, c3, c4 are pairwise distinct.
        
        # So, if c2, c3, c4 are pairwise distinct:
        # X is forbidden if X is NOT in {c2, c3, c4}.
        # X is allowed if X is in {c2, c3, c4}.
        
        # If c2, c3, c4 are NOT pairwise distinct (i.e., at least two are equal):
        # Then {X, c2, c3, c4} can never have size 4.
        # So X is always allowed.
        
        # Conclusion:
        # For edge e, look at all cycles containing e.
        # For each cycle, if the other 3 edges have distinct colors, 
        # then e MUST be colored with one of those 3 colors.
        # If ANY cycle has distinct other colors, e is restricted to the intersection of those sets.
        # If multiple cycles restrict e, e must be in the intersection of all allowed sets.
        
        # So, allowed_colors = Intersection over all cycles C containing e of:
        #   If other_colors(C) are distinct: Set(other_colors(C))
        #   Else: All colors (0..119)
        
        # If the intersection is empty, the current state is invalid for this edge? 
        # No, the current state is valid, so the current color of e MUST be in the intersection.
        
        # This function should return the set of ALLOWED colors, or rather, 
        # it's easier to return the set of FORBIDDEN colors if we want to pick a new one.
        # But the logic above says:
        # If other colors are distinct, allowed = {c2, c3, c4}. Forbidden = All \ {c2, c3, c4}.
        # If other colors are not distinct, allowed = All. Forbidden = Empty.
        
        # So, for each cycle:
        if c1 != c2 and c1 != c3 and c2 != c3:
            # The edge MUST take one of c1, c2, c3 to avoid rainbow.
            # So any color NOT in {c1, c2, c3} is forbidden.
            # We add all other colors to forbidden? That's expensive.
            # Instead, we can track the "Allowed Set".
            pass
    
    # Let's rewrite to compute Allowed Set.
    allowed = set(range(MAX_COLOR + 1))
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        if c1 != c2 and c1 != c3 and c2 != c3:
            # The new color MUST be one of c1, c2, c3.
            # So the allowed colors for this constraint are {c1, c2, c3}.
            # The overall allowed set is the intersection.
            allowed.intersection_update({c1, c2, c3})
            
            # Optimization: If allowed becomes empty, we can break? 
            # But current state is valid, so current color must be in allowed.
            # If allowed is empty, something is wrong.
            if not allowed:
                break
        
    return allowed

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Initialize with a structured pattern
    # Using a linear congruential generator style pattern often creates blocks of same colors.
    # color(i,j) = (i * 3 + j) % 10
    # This limits initial colors to 10.
    
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        colors[idx] = (i * 3 + j) % 10
        
    # Verify initial validity
    if count_violations(colors) != 0:
        # Fallback to all 0s if pattern fails (unlikely for this simple one)
        colors = [0] * NUM_EDGES

    current_colors_used = len(set(colors))
    
    # 2. Greedy Color Injection
    # We want to introduce new colors.
    # Strategy: Iterate through edges. For each edge, check if we can assign a NEW color 
    # that is not currently used, while maintaining validity.
    
    # To make this more effective, we can prioritize edges that have "loose" constraints.
    # An edge is "loose" if its allowed set (from get_allowed_colors) is large 
    # or if it doesn't restrict other edges too much.
    
    # Actually, get_allowed_colors tells us what we CAN do.
    # If we want to use a NEW color K, K must be in allowed_set.
    # And K must not be in current_used_colors.
    
    # Let's maintain a list of unused colors.
    unused_colors = [c for c in range(MAX_COLOR + 1) if c not in set(colors)]
    
    # Shuffle edges to avoid bias
    edge_order = list(range(NUM_EDGES))
    random.seed(42)
    random.shuffle(edge_order)
    
    # We will try to inject colors in multiple passes.
    # In each pass, we try to assign a new unused color to an edge.
    
    max_passes = 100
    for _pass in range(max_passes):
        if time.time() > deadline - 5.0:
            break
            
        if not unused_colors:
            break
            
        progress = False
        current_used = set(colors)
        
        # Shuffle edges again for variety in each pass? 
        # Or stick to the order? Shuffling helps avoid getting stuck in a local pattern.
        random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline - 5.0:
                break
                
            # Get allowed colors for this edge
            allowed = get_allowed_colors(colors, e)
            
            # We want to pick a color from unused_colors that is in allowed.
            # To be efficient, we can check if any unused color is in allowed.
            # Since allowed might be small (size 1-3) or large (size 120),
            # and unused_colors is large, it's better to iterate through allowed 
            # and check if it's in unused.
            
            candidate_color = -1
            for c in allowed:
                if c not in current_used:
                    candidate_color = c
                    break
            
            if candidate_color != -1:
                # Check if assigning candidate_color is actually valid globally?
                # get_allowed_colors ensures no NEW rainbow cycles are formed 
                # involving this edge, assuming other edges stay same.
                # Since we are changing only this edge, this is sufficient.
                
                old_color = colors[e]
                colors[e] = candidate_color
                
                # Update used sets
                # If old_color is no longer used, add it to unused?
                # We don't strictly need to track unused dynamically if we just 
                # check "not in current_used" on the fly, but maintaining unused_colors 
                # list helps if we want to pick from it.
                # However, removing from a list is O(N). 
                # Let's just recalculate current_used if needed, or keep a set.
                
                # For simplicity and speed, let's just update the set.
                # We don't update the unused_colors list immediately, 
                # but we check membership in current_used.
                
                current_used.add(candidate_color)
                
                # If old_color is now unused, we could add it back to unused_colors, 
                # but since we are trying to MAXIMIZE distinct colors, 
                # replacing a used color with a new one is a net gain of 0 
                # IF the old color is still used elsewhere.
                # If the old color is NOT used elsewhere, we lose one color and gain one. Net 0.
                # Wait. 
                # If old_color was unique to this edge, and we change it to new_color:
                # We lose old_color, gain new_color. Total distinct count stays same.
                # This is not helpful for maximizing count.
                
                # We should only accept the move if it INCREASES the distinct count.
                # This happens if:
                # 1. candidate_color is new (not in current_used).
                # 2. old_color is STILL used elsewhere (so we don't lose it).
                
                # Let's check if old_color is used elsewhere.
                # We need to count occurrences.
                
                # To avoid O(N) count every time, we can maintain counts.
                # But for now, let's do a quick check.
                # If this is too slow, we'll optimize.
                
                # Actually, let's just accept the move if it increases the score.
                # We can calculate the new score.
                
                # Optimization: 
                # If old_color appears only once, and we change it, score stays same.
                # If old_color appears >1 times, score increases by 1 (if new color is new).
                
                # Let's maintain a count array.
                # This requires updating counts on every change.
                
                # Let's restart this part with counts.
                
                # Rollback for now to implement counts properly in the loop below.
                colors[e] = old_color
                current_used.discard(candidate_color)
                
                # Break out to implement count-based logic?
                # No, let's just use a simpler heuristic:
                # Only try to assign a new color if the edge's current color is "redundant" 
                # (appears > 1 time).
                
                # How to know if it's redundant? 
                # We can maintain a list of color counts.
                
                # Let's implement a proper count-based loop.
                
                # Break inner loop to restart with counts? 
                # No, let's just do the count check inline.
                
                # Count occurrences of old_color in colors (excluding e)
                # This is O(120). 120 * 120 * 100 passes = 1.4M ops. Fast enough.
                
                count_old = 0
                for i in range(NUM_EDGES):
                    if i != e and colors[i] == old_color:
                        count_old += 1
                        if count_old > 0:
                            break # We just need to know if > 0
                
                if count_old > 0:
                    # Safe to change, will increase distinct count by 1
                    colors[e] = candidate_color
                    current_used.add(candidate_color)
                    progress = True
                    # Remove candidate_color from unused_colors? 
                    # We can just let the "not in current_used" check handle it.
                    
                    # Optional: Remove from unused_colors list to speed up future checks?
                    # List removal is O(N). N=120. 
                    # try:
                    #     unused_colors.remove(candidate_color)
                    # except ValueError:
                    #     pass
                    pass
                # Else:
                # Changing would not increase distinct count. Skip.
        
        if not progress:
            # No progress in this pass. 
            # We might be stuck. 
            # Try a different strategy: 
            # Maybe we need to "unstick" by allowing a temporary decrease? 
            # Or just stop if we can't find any more single-edge improvements.
            
            # Let's try a second phase: 
            # If we are stuck, maybe we can swap colors to create redundancy?
            # E.g., if two edges have unique colors, swap them so one color becomes redundant?
            # No, that doesn't help.
            
            # If we are stuck, it means for every edge, 
            # either:
            # 1. Its current color is unique (so changing it doesn't help).
            # 2. No new unused color is allowed for it.
            
            # To break this, we might need to change an edge to a color that IS used, 
            # to make its old color unique? No, that makes old color more redundant.
            
            # Actually, if an edge has a unique color, we CAN'T use it to add a new color 
            # without losing the unique one.
            # So we need to "consolidate" colors first?
            # No, we want to MAXIMIZE distinct colors.
            
            # If we are stuck, it means the set of "flexible" edges (those with redundant colors) 
            # cannot accept any new unused colors.
            
            # We could try to relax constraints by changing other edges?
            # This is getting complex. 
            # Let's just break if no progress.
            break

    # Final safety check
    if count_violations(colors) != 0:
        # Fallback
        colors = [0] * NUM_EDGES

    return list(colors)

# To make the above work better, I need to actually implement the count logic correctly 
# inside the loop. The "rollback and re-check" is inefficient.
# Let's rewrite the construct function with a cleaner implementation.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Initialization
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        colors[idx] = (i * 3 + j) % 10
    
    if count_violations(colors) != 0:
        colors = [0] * NUM_EDGES

    # Maintain color counts
    color_counts = [0] * (MAX_COLOR + 1)
    for c in colors:
        color_counts[c] += 1
    
    current_distinct = sum(1 for c in color_counts if c > 0)
    
    # 2. Greedy Injection
    # We want to increase current_distinct.
    # An edge e with color c_e can be changed to new_color c_new (where c_new is unused)
    # if:
    # 1. c_new is in allowed_colors(e).
    # 2. c_e is redundant (color_counts[c_e] > 1).
    
    # If we do this, c_new becomes used (distinct +1), c_e count decreases by 1.
    # If c_e count was 1, distinct -1. Net 0.
    # If c_e count was >1, distinct +1. Net +1.
    
    # So we only want to change edges with redundant colors.
    
    # Precompute allowed colors? 
    # Allowed colors depend on the current state.
    # So we must calculate on the fly.
    
    # To speed up, we can prioritize edges with high redundancy?
    # Or just iterate.
    
    max_passes = 200
    for _pass in range(max_passes):
        if time.time() > deadline - 5.0:
            break
            
        progress = False
        # Iterate edges in random order
        edges_to_try = list(range(NUM_EDGES))
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if time.time() > deadline - 5.0:
                break
                
            old_color = colors[e]
            
            # Check if old color is redundant
            if color_counts[old_color] <= 1:
                continue
            
            # Get allowed colors
            allowed = get_allowed_colors(colors, e)
            
            # Find an unused color in allowed
            # Iterate through allowed (small set) and check if unused
            new_color = -1
            for c in allowed:
                if color_counts[c] == 0:
                    new_color = c
                    break
            
            if new_color != -1:
                # Perform change
                colors[e] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                current_distinct += 1
                progress = True
                
                # We can break early if we hit max? No, keep going.
                # But we should update the "unused" status implicitly via color_counts.
        
        if not progress:
            break

    # Final safety check
    if count_violations(colors) != 0:
        colors = [0] * NUM_EDGES
        # Recalculate counts for fallback? Not needed if returning.

    return list(colors)
