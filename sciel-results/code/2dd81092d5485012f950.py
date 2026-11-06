import time
import random
import sys

N = 16
M = N * (N - 1) // 2  # 120 edges

def _edge_index(i, j):
    """Lexicographic index of edge (i,j), i<j, in the list of all C(N,2) edges."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _get_edges():
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Precompute all 4-cycles.
    Each cycle is a tuple of 4 edge indices.
    There are 3 distinct 4-cycles on every 4-vertex subset {a,b,c,d} (with a<b<c<d):
    1. (a,b), (b,c), (c,d), (d,a)
    2. (a,b), (b,d), (d,c), (c,a)
    3. (a,c), (c,b), (b,d), (d,a)
    """
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = _edge_index(a, b)
                    e2 = _edge_index(b, c)
                    e3 = _edge_index(c, d)
                    e4 = _edge_index(a, d)
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = _edge_index(a, b)
                    e2 = _edge_index(b, d)
                    e3 = _edge_index(c, d)
                    e4 = _edge_index(a, c)
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = _edge_index(a, c)
                    e2 = _edge_index(b, c)
                    e3 = _edge_index(b, d)
                    e4 = _edge_index(a, d)
                    cycles.append((e1, e2, e3, e4))
    return cycles

CYCLES = _precompute_cycles()
# Map edge index to list of cycles containing it
EDGE_CYCLES = [[] for _ in range(M)]
for idx, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(idx)

def _verify(colors):
    """Return True iff no 4-cycle has four distinct edge colors."""
    c = colors
    for cycle in CYCLES:
        if len({c[cycle[0]], c[cycle[1]], c[cycle[2]], c[cycle[3]]}) == 4:
            return False
    return True

def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s kill
    
    # Initial constructive solution: min(i,j) coloring
    # This is known to be valid and uses 15 colors.
    best_colors = [0] * M
    for i in range(N):
        base = _edge_index(i, i + 1)
        for j in range(i + 1, N):
            best_colors[base + (j - i - 1)] = i
            
    best_score = len(set(best_colors))
    
    # MIS-like Search Strategy
    # We view the problem as selecting a color for each edge.
    # However, a full MIS on 120 edges * 120 colors is too large.
    
    # Alternative Strategy: Randomized Restart Local Search with Constraint Propagation
    # The prompt suggests "Model the problem as a maximum independent set search...".
    # A direct MIS is hard. But we can interpret "conflict graph" differently:
    # Nodes = (edge, color) pairs.
    # Edges in conflict graph connect (e1, c1) and (e2, c2) if assigning c1 to e1 and c2 to e2 creates a rainbow C4 with some existing assignments?
    # This is dynamic.
    
    # Let's try a different structural approach:
    # Start with the 15-color solution.
    # Try to introduce new colors by recoloring edges, but use a "conflict check" that is very fast.
    # We will use a simulated annealing / random walk approach but with a stronger constraint:
    # We only allow recoloring if it doesn't create a rainbow C4.
    # To escape local optima, we allow "invalid" intermediate states with a penalty? No, task requires valid output.
    # So we must stay in the valid space.
    
    # The space of valid colorings is disconnected? Probably not, but local search might get stuck.
    # The 15-color solution is a "star" centered at 0.
    # Can we get 16 colors?
    # If we use 16 colors, by Pigeonhole, one color is used on at most 120/16 = 7.5 edges.
    # Actually, the max number of colors is the question.
    # The min(i,j) coloring uses 15.
    # Is it possible to use 16?
    # If we use 16 colors, every color must appear at least once.
    # Consider vertex 0. It has 15 edges. If all 15 edges have distinct colors, that's 15 colors.
    # The 16th color must be on an edge not incident to 0, say (1,2).
    # If (1,2) has color 15, does it create a rainbow C4?
    # In min(i,j) coloring:
    # (1,2) is color 1.
    # If we change (1,2) to 15:
    # Cycle 0-1-2-3-0:
    # (0,1)=0, (1,2)=15, (2,3)=2, (0,3)=0. Colors: 0, 15, 2, 0. Not rainbow.
    # Cycle 0-1-2-4-0:
    # (0,1)=0, (1,2)=15, (2,4)=2, (0,4)=0. Colors: 0, 15, 2, 0. Not rainbow.
    # ...
    # Cycle 1-2-3-4-1:
    # (1,2)=15, (2,3)=2, (3,4)=3, (1,4)=1. Colors: 15, 2, 3, 1. All distinct! Rainbow!
    # So simply changing (1,2) to 15 fails.
    
    # We need to change other edges too.
    # This suggests a coordinated change.
    
    # Let's use a Randomized Local Search that tries to swap colors or recolor sets of edges.
    
    random.seed(42)
    
    current_colors = best_colors[:]
    current_score = best_score
    
    # Precompute neighbors for fast conflict checking?
    # For an edge e, the cycles it belongs to are EDGE_CYCLES[e].
    # When changing color of e, we only need to check those cycles.
    
    def is_valid_change(edge_idx, new_color, colors):
        if colors[edge_idx] == new_color:
            return True
        for cycle_idx in EDGE_CYCLES[edge_idx]:
            c = CYCLES[cycle_idx]
            # Get colors of the 4 edges in this cycle
            # One of them is edge_idx, which will have new_color
            # The others have their current colors
            col_set = set()
            for e in c:
                if e == edge_idx:
                    col_set.add(new_color)
                else:
                    col_set.add(colors[e])
            if len(col_set) == 4:
                return False
        return True

    # Search Loop
    # We try to increase the number of distinct colors.
    # Strategy: Pick an edge, try to assign it a new color (not currently in use, or a color that increases diversity).
    # Actually, just try to assign any color that doesn't break validity, and see if it increases diversity.
    # To encourage new colors, we bias towards colors not currently in the set.
    
    used_colors = set(current_colors)
    
    steps = 0
    max_steps = 200000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Pick a random edge
        e = random.randint(0, M - 1)
        
        # Determine candidate colors
        # 1. A new color (randomly chosen from 0..119 not in used_colors)
        # 2. An existing color
        # 3. A random color
        
        if random.random() < 0.3:
            # Try a new color
            available_new = [c for c in range(120) if c not in used_colors]
            if available_new:
                new_c = random.choice(available_new)
            else:
                new_c = random.randint(0, 119)
        else:
            new_c = random.randint(0, 119)
            
        if is_valid_change(e, new_c, current_colors):
            old_c = current_colors[e]
            current_colors[e] = new_c
            
            # Update used_colors
            if old_c not in current_colors:
                used_colors.discard(old_c)
            if new_c not in used_colors:
                used_colors.add(new_c)
                
            new_score = len(used_colors)
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors[:]
                
            # Acceptance criterion:
            # Always accept if valid.
            # This is a random walk on the valid space.
            # To escape local optima, we don't need simulated annealing if we just wander.
            # But wandering might reduce score.
            # Let's add a slight bias to accept moves that maintain or improve score, 
            # and occasionally accept worse ones to escape.
            
            # For now, let's just do a pure random walk to explore the space.
            # If we find a better score, we update best.
            # We don't revert the move, so we continue from the new state.
            
        # Periodic check to ensure we don't drift too far from a good solution?
        # No, the valid space is likely connected.
        
        # Optimization: If we hit the deadline, break.
        
    # Final verification
    if not _verify(best_colors):
        # Fallback to constructive
        for i in range(N):
            base = _edge_index(i, i + 1)
            for j in range(i + 1, N):
                best_colors[base + (j - i - 1)] = i
        
    return best_colors

# Run construct to test? No, just define it.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Colors used: {len(set(res))}")
    print(f"Valid: {_verify(res)}")
