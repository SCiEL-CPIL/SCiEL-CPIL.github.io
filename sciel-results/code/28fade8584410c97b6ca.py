"""
Strategy: Max-SAT formulation with WalkSAT.
Variables:
1. Edge colors (0..119)
2. Active[color] boolean (1 if any edge uses 'color', else 0)

Objective: Maximize sum(Active[color]) - 1000 * Violations
Violations: Number of 4-cycles with 4 distinct colors.

We represent the state as a list of 120 edge colors.
We precompute all 4-cycles (sets of 4 edges) and their "forbidden" color patterns.
A 4-cycle is invalid if the 4 edges have 4 distinct colors.

To speed up evaluation, we maintain a count of violations and update it incrementally.
However, WalkSAT usually requires re-evaluating clauses. Given N=16, there are C(16,4) * 3 = 1820 * 3 = 5460 4-cycles.
Checking 5460 cycles per step is fast enough for ~10k-50k steps in 60s.

We will use a hybrid approach:
1. Start with a valid low-color coloring (e.g., color(i,j)=i).
2. Randomly pick an edge and a new color.
3. Evaluate the change in violations and active colors.
4. Accept if it improves the score, or with some probability if it maintains/improves violations but increases active colors (or vice versa).
Actually, simple hill climbing on (ActiveColors, -Violations) might get stuck.
WalkSAT picks a variable (edge) and a value (color) to flip.
It selects the flip that minimizes the number of newly violated clauses (or maximizes satisfied ones).
Since our constraint is "not (c1!=c2 and c1!=c3 and c1!=c4 and c2!=c3 and c2!=c4 and c3!=c4)", it's easier to just check validity of the whole graph or incremental updates.

Let's use incremental updates for violations.
For each 4-cycle, we store the 4 edge indices.
We maintain a `violations` count.
When changing edge `e` from `c_old` to `c_new`:
- For every 4-cycle containing `e`:
    - Check if it was violated before (based on old colors).
    - Check if it is violated after (based on new colors).
    - Update count.
- Also update `active_colors` set.

This allows O(1) amortized update per edge flip if we precompute the list of cycles per edge.
Each edge is in C(14,2) * 2 + ... roughly 100-200 cycles?
Total cycles = 5460.
Edges = 120.
Avg cycles per edge = 5460 / 120 = 45.5.
So 45 checks per move. Very fast.

We can do many moves.
"""

import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Map edge (i,j) to index
    # Lexicographic order: (0,1), (0,2)...(0,15), (1,2)...
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
            
    # 2. Precompute all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and one of the 3 pairings of edges.
    # The 3 cycles on {a,b,c,d} are:
    # (a,b), (b,c), (c,d), (d,a) -> Edges: ab, bc, cd, da
    # (a,c), (c,b), (b,d), (d,a) -> Edges: ac, cb, bd, da (Same set of edges as above? No.)
    # Let's list the edges for a 4-vertex set {v0, v1, v2, v3}:
    # The 6 edges are: v0v1, v0v2, v0v3, v1v2, v1v3, v2v3.
    # The 3 cycles are:
    # 1. v0-v1-v2-v3-v0: Edges (v0,v1), (v1,v2), (v2,v3), (v3,v0)
    # 2. v0-v1-v3-v2-v0: Edges (v0,v1), (v1,v3), (v3,v2), (v2,v0)
    # 3. v0-v2-v1-v3-v0: Edges (v0,v2), (v2,v1), (v1,v3), (v3,v0)
    
    cycles = [] # List of lists of 4 edge indices
    for v0 in range(N):
        for v1 in range(v0 + 1, N):
            for v2 in range(v1 + 1, N):
                for v3 in range(v2 + 1, N):
                    # Cycle 1
                    e1 = edge_to_idx[(v0, v1)]
                    e2 = edge_to_idx[(v1, v2)]
                    e3 = edge_to_idx[(v2, v3)]
                    e4 = edge_to_idx[(v3, v0)]
                    cycles.append([e1, e2, e3, e4])
                    
                    # Cycle 2
                    e1 = edge_to_idx[(v0, v1)]
                    e2 = edge_to_idx[(v1, v3)]
                    e3 = edge_to_idx[(v2, v3)]
                    e4 = edge_to_idx[(v0, v2)]
                    cycles.append([e1, e2, e3, e4])
                    
                    # Cycle 3
                    e1 = edge_to_idx[(v0, v2)]
                    e2 = edge_to_idx[(v1, v2)]
                    e3 = edge_to_idx[(v1, v3)]
                    e4 = edge_to_idx[(v0, v3)]
                    cycles.append([e1, e2, e3, e4])
    
    # 3. Precompute cycles per edge for fast updates
    cycles_per_edge = [[] for _ in range(NUM_EDGES)]
    for c_idx, cycle in enumerate(cycles):
        for e in cycle:
            cycles_per_edge[e].append(c_idx)
            
    # Helper to check if a cycle has 4 distinct colors
    def is_violated(cycle, colors):
        c1 = colors[cycle[0]]
        c2 = colors[cycle[1]]
        c3 = colors[cycle[2]]
        c4 = colors[cycle[3]]
        return len({c1, c2, c3, c4}) == 4

    # Initial state: color(i,j) = i
    # This is valid (15 colors).
    current_colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            current_colors[edge_to_idx[(i, j)]] = i
            
    # Calculate initial violations (should be 0)
    current_violations = sum(1 for c in cycles if is_violated(c, current_colors))
    current_active = set(current_colors)
    
    best_colors = current_colors[:]
    best_active_count = len(current_active)
    
    # WalkSAT / Local Search
    # We will try to increase active colors.
    # Score = 1000 * active_count - violations
    # We accept moves that improve score.
    # To escape local optima, we allow some random moves or "neutral" moves that don't decrease active count but might increase violations temporarily?
    # No, we must stay valid (violations=0) to be a valid solution?
    # The task says "No four-cycle may have four distinct edge colors".
    # So the final result MUST have 0 violations.
    # Intermediate states can have violations if we are doing a soft-constraint search, but we must return a valid one.
    # If we do hard constraint search (only valid moves), we might get stuck.
    # If we do soft constraint search, we must ensure we end up at 0 violations.
    
    # Let's stick to valid moves only (violations must remain 0).
    # If we can't find a valid move that increases active colors, we do a random valid move (neutral) or backtrack?
    # Actually, "random valid move" is hard to find if the space is tight.
    # Let's try:
    # 1. Pick random edge.
    # 2. Pick random new color.
    # 3. If new state is valid (0 violations) AND active count increases -> Accept.
    # 4. If new state is valid (0 violations) AND active count same -> Accept with prob 0.1 (to shuffle).
    # 5. If new state is valid (0 violations) AND active count decreases -> Reject.
    # 6. If new state is invalid -> Reject.
    
    # This might be too restrictive.
    # Alternative: Allow temporary violations?
    # If we allow violations, we need a strategy to fix them.
    # Given the small size, let's try the "Valid Moves Only" approach first, but with a broader search of colors.
    
    # Optimization: 
    # Instead of checking all 45 cycles for every candidate color (120 colors * 45 cycles * 120 edges = too slow for 60s?),
    # 120 edges * 120 colors = 14400 candidates per iteration.
    # 45 cycles * 4 checks = 180 ops per candidate.
    # 14400 * 180 = 2.5M ops per iteration.
    # 60s / 2.5M = 24 iterations. Too slow.
    
    # We need to be smarter.
    # Precompute for each edge and each color, the set of cycles that would be violated if that edge took that color, *given the current state of other edges*?
    # That's dynamic.
    
    # Better: 
    # For a fixed edge `e` and candidate color `c_new`:
    # We only need to check cycles containing `e`.
    # For each such cycle, we know the other 3 edge colors.
    # The cycle is violated if `c_new` is distinct from those 3, AND those 3 are distinct from each other.
    # Let `others` be the set of 3 other colors in the cycle.
    # If `len(others) == 3` (they are all distinct), then `c_new` MUST be in `others` to avoid violation.
    # If `len(others) < 3`, the cycle is NOT violated regardless of `c_new` (because we can't have 4 distinct if 3 are not distinct).
    
    # So, for edge `e`, for each cycle `c` containing `e`:
    # Let `others` be the colors of the other 3 edges in `c`.
    # If `len(set(others)) == 3`:
    #     Then `c_new` MUST be in `set(others)`.
    #     This imposes a constraint on `c_new`.
    
    # We can aggregate these constraints for edge `e`.
    # `AllowedColors[e]` = Intersection of `set(others)` for all cycles `c` containing `e` where `len(set(others))==3`.
    # If `AllowedColors[e]` is empty, no color is valid for `e` given the current state.
    # If `AllowedColors[e]` is non-empty, any color in it keeps the graph valid.
    
    # This allows O(1) validity check per (edge, color) pair after O(CyclesPerEdge) precomputation per step.
    
    # Algorithm:
    # 1. Compute `AllowedColors` for all edges.
    # 2. If any `AllowedColors[e]` is empty, state is "dead" (should not happen if we only move to valid states).
    # 3. To increase active colors:
    #    Find an edge `e` and a color `c` in `AllowedColors[e]` such that `c` is NOT in `current_active`.
    #    If found, assign `current_colors[e] = c`. Update `current_active`.
    # 4. If no such "new color" move exists, try a "neutral" move:
    #    Find an edge `e` and a color `c` in `AllowedColors[e]` such that `c` is in `current_active` but `c != current_colors[e]`.
    #    This shuffles the coloring without changing the set of active colors.
    # 5. If no moves exist, we are stuck.
    
    # This is a "Greedy Valid Move" search.
    
    def compute_allowed(colors):
        allowed = [set(range(120)) for _ in range(NUM_EDGES)]
        for e in range(NUM_EDGES):
            for c_idx in cycles_per_edge[e]:
                cycle = cycles[c_idx]
                others = []
                for ne in cycle:
                    if ne != e:
                        others.append(colors[ne])
                other_set = set(others)
                if len(other_set) == 3:
                    # c_new must be in other_set
                    allowed[e] = allowed[e].intersection(other_set)
        return allowed

    # Initial allowed
    allowed = compute_allowed(current_colors)
    
    steps = 0
    while time.time() < deadline:
        # 1. Try to add a new color
        found_new = False
        # Randomly shuffle edges to avoid bias
        edges_to_try = list(range(NUM_EDGES))
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if not allowed[e]:
                continue
            # Find a color in allowed[e] that is not in current_active
            # To be efficient, we can just iterate allowed[e] if it's small, or pick random
            # allowed[e] can be large.
            # Let's pick a random color from allowed[e] and check if it's new.
            # But we want to MAXIMIZE new colors.
            # Better: Iterate through allowed[e] and pick the first one not in current_active.
            # If allowed[e] is large, this is fast.
            
            # Optimization: Keep a list of inactive colors?
            # Or just check membership.
            
            for c in allowed[e]:
                if c not in current_active:
                    # Make the move
                    old_c = current_colors[e]
                    current_colors[e] = c
                    current_active.add(c)
                    
                    # Update best
                    if len(current_active) > best_active_count:
                        best_active_count = len(current_active)
                        best_colors = current_colors[:]
                    
                    # Recompute allowed for all edges?
                    # Only edges in cycles with e are affected?
                    # Actually, changing e affects the "others" set for neighbors.
                    # We must recompute allowed for all edges that share a cycle with e.
                    # Or just recompute all (120 edges * 45 cycles = 5400 ops). Fast.
                    allowed = compute_allowed(current_colors)
                    found_new = True
                    break
            if found_new:
                break
        
        if found_new:
            continue
            
        # 2. No new color found. Try a neutral move to escape local optimum.
        found_neutral = False
        edges_to_try = list(range(NUM_EDGES))
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if not allowed[e]:
                continue
            old_c = current_colors[e]
            # Find c in allowed[e] such that c != old_c
            # If allowed[e] has > 1 element, there is a choice.
            if len(allowed[e]) > 1:
                # Pick a random different color
                choices = [c for c in allowed[e] if c != old_c]
                if choices:
                    new_c = random.choice(choices)
                    current_colors[e] = new_c
                    # Active set doesn't change (new_c is in active, old_c might leave active if no other edge uses it)
                    # Wait, if old_c is no longer used, active set shrinks!
                    # We must ensure old_c is still used by some other edge.
                    # Check if old_c is in current_colors (excluding e)
                    # This is O(N).
                    if old_c in current_colors[:e] + current_colors[e+1:]:
                        # Safe to move
                        allowed = compute_allowed(current_colors)
                        found_neutral = True
                        break
                    else:
                        # Moving would lose a color. Reject this neutral move.
                        # Try next choice?
                        pass
        if found_neutral:
            # Small chance to accept a neutral move even if it doesn't immediately help?
            # Yes, to explore.
            pass
        else:
            # Stuck. Try a random restart? Or a "kick" that allows 1 violation temporarily?
            # For simplicity, let's just break and return best if we've been stuck for a while.
            # Or, do a random valid move that might lose a color? No, we want to maximize.
            # Let's try a different strategy: Randomly change an edge to a valid color, even if it loses an active color, 
            # hoping to find a path to more colors?
            # Risky. Let's just terminate search if no moves.
            break
            
    return best_colors

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
