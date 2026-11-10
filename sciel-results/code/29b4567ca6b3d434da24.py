import time
import numpy as np
from itertools import combinations
import sys

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _get_edge_indices():
    """Return edge index mapping and cycle structures."""
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)]))
        # Cycle 2: a-b-d-c-a
        cycles.append((idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)]))
        # Cycle 3: a-c-b-d-a
        cycles.append((idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)]))
    
    cycles_np = np.array(cycles, dtype=np.int32)
    
    # For each edge, find which cycles contain it
    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    return idx_map, cycles_np, edge_to_cycles

IDX_MAP, CYCLES, EDGE_TO_CYCLES = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

# Precompute for faster access
# cycles is a list of tuples of 4 edge indices
# We need to check if assigning color c to edge e violates any cycle containing e.
# A cycle is violated if the other 3 edges have colors {A, B, C} (all distinct) and c is not in {A, B, C}.
# This implies that if we assign colors to edges one by one, we must ensure that for any cycle 
# where 3 edges are already colored with distinct colors, the 4th edge MUST be one of those 3 colors.

def _backtracking_search(target_distinct, deadline):
    """
    Attempt to find a valid coloring using exactly 'target_distinct' distinct colors.
    We force the first 'target_distinct' edges to have distinct colors 0, 1, ..., target_distinct-1.
    Then we fill the rest using backtracking.
    Returns a valid coloring array if found, else None.
    """
    colors = [-1] * NEDGES
    
    # Force the first 'target_distinct' edges to be distinct
    # To maximize diversity, we assign 0..K-1 to the first K edges in a specific order.
    # The order of edges matters for constraint propagation. 
    # A good order is to pick edges that are in many cycles early? 
    # Or just lexicographic? Let's try lexicographic first, it's simple.
    # Actually, edges connecting vertices 0 and 1 are in many cycles.
    # Let's just use the standard index order 0..NEDGES-1.
    
    for i in range(target_distinct):
        colors[i] = i
        
    # We need to check if this partial assignment is already invalid.
    # But since we are assigning distinct colors to the first K edges, 
    # we must ensure no cycle among these first K edges has 4 distinct colors.
    # Wait, if we assign 0, 1, 2, 3 to edges e1, e2, e3, e4 and they form a cycle, it's invalid.
    # So we must check validity of the initial forced assignment.
    
    # Optimization: Instead of blind backtracking, we can use a recursive function 
    # that picks the next unassigned edge and tries valid colors.
    
    # To make backtracking efficient, we need to order edges such that 
    # constraints are propagated quickly.
    # However, for N=16, the number of edges is 120.
    # A naive backtracking might be too slow if the search space is large.
    
    # Strategy:
    # 1. Use a heuristic to order edges. Edges with higher "constraint density" first?
    #    Or simply: pick the unassigned edge that appears in the most "almost-satisfied" cycles?
    #    Let's stick to a simpler heuristic: Pick the unassigned edge that is part of 
    #    the most cycles that already have 3 assigned edges.
    
    # Precompute: For each edge, the list of cycles it belongs to.
    # We will maintain a list of unassigned edges.
    
    unassigned = list(range(target_distinct, NEDGES))
    
    # We need a fast validity check for the current state.
    # Since we are doing backtracking, we should check validity incrementally.
    
    # Let's define the recursive step.
    # To avoid recursion depth issues (120 is fine for Python if we set limit or use iterative),
    # we can use a stack or recursion. 120 depth is safe.
    
    sys.setrecursionlimit(10000)
    
    def is_valid_partial():
        """Check if current partial assignment has no violations."""
        # We only need to check cycles where all 4 edges are assigned.
        # But we can be smarter: check cycles where the unassigned edge is the only one left,
        # or just check all completed cycles.
        # Given the small number of cycles (C(16,4)*3 = 1820*3 = 5460), 
        # checking all completed cycles every step is 5460 * 120 steps = ~650k ops. 
        # This is fast enough in Python if optimized? Maybe.
        # Better: Only check cycles involving the newly assigned edge.
        pass # We'll do incremental check in the recursion.

    # Let's implement a recursive backtracker.
    # State: colors array, set of used colors? No, we want to maximize distinct.
    # We forced the first K to be distinct. The rest can be anything.
    # To maximize distinct, we should prefer colors that are already used? 
    # No, the goal is to find *any* valid coloring with at least K distinct colors.
    # The distinct count is determined by the forced K + any new colors introduced.
    # But we are testing if a coloring with *at least* K distinct colors exists.
    # Actually, if we force K distinct colors, we have *at least* K.
    # If we find a valid completion, we have a valid coloring with >= K distinct colors.
    # We can then try K+1.
    
    # To speed up search, when assigning a new edge, we should prefer colors that 
    # don't create new distinct colors? No, that minimizes distinct.
    # We just need *one* valid coloring.
    # Heuristic: Try colors that are already in use first? 
    # If we use existing colors, we might satisfy constraints more easily 
    # (since constraints require repetition).
    # Yes! The constraint is "no 4 distinct". Reusing colors helps satisfy this.
    # So, prioritize colors that are already present in the cycle's other edges.
    
    def backtrack(edge_idx, unassigned_edges):
        if not unassigned_edges:
            return True
        
        # Choose an edge to assign
        # Heuristic: Pick the edge in unassigned_edges that has the most constraints
        # (i.e., appears in the most cycles where other 3 are already assigned).
        
        best_edge = -1
        max_constrained = -1
        
        # To avoid O(N) scan every time, we could maintain this, but N=120 is small.
        for e in unassigned_edges:
            count = 0
            for ci in EDGE_TO_CYCLES[e]:
                c_edges = CYCLES[ci]
                # Check how many of the other 3 are assigned
                assigned_count = 0
                for pos in range(4):
                    if c_edges[pos] == e:
                        continue
                    if colors[c_edges[pos]] != -1:
                        assigned_count += 1
                if assigned_count == 3:
                    count += 1
            if count > max_constrained:
                max_constrained = count
                best_edge = e
        
        if best_edge == -1:
            # No edge has 3 assigned neighbors? Just pick the first one.
            best_edge = unassigned_edges[0]
        
        # Remove best_edge from unassigned
        new_unassigned = [e for e in unassigned_edges if e != best_edge]
        
        # Determine candidate colors for best_edge
        # 1. Colors that appear in the other edges of any cycle where other 3 are assigned.
        #    If a cycle has colors {A, B, C} on other 3, best_edge MUST be A, B, or C.
        # 2. If multiple such cycles exist, the candidate set is the intersection of their allowed sets.
        # 3. If no cycle has 3 assigned neighbors, any color 0..MAX_COLOR-1 is allowed.
        
        allowed_colors = set(range(MAX_COLOR))
        
        for ci in EDGE_TO_CYCLES[best_edge]:
            c_edges = CYCLES[ci]
            other_colors = []
            all_assigned = True
            for pos in range(4):
                if c_edges[pos] == best_edge:
                    continue
                col = colors[c_edges[pos]]
                if col == -1:
                    all_assigned = False
                    break
                other_colors.append(col)
            
            if all_assigned:
                # The other 3 are assigned.
                # If they are all distinct {A, B, C}, best_edge must be in {A, B, C}.
                # If they have duplicates, e.g. {A, A, B}, best_edge can be anything?
                # If best_edge is D (new), cycle is {A, A, B, D} -> 3 distinct. Valid.
                # If best_edge is C (new), cycle is {A, A, B, C} -> 3 distinct. Valid.
                # So if other 3 are not all distinct, any color is allowed.
                if len(set(other_colors)) == 3:
                    allowed_colors.intersection_update(set(other_colors))
        
        if not allowed_colors:
            return False
        
        # Sort candidates: Prefer colors that are already used globally?
        # Or prefer colors that are in the allowed set and minimize new distinct colors?
        # Since we just want *a* solution, let's try colors in a deterministic order 
        # but biased towards existing colors to help future constraints.
        
        # Count frequency of each color in current assignment
        # To be fast, we can just sort by "is it used?".
        used_colors = set(colors)
        
        # Split allowed_colors into used and unused
        used_allowed = [c for c in allowed_colors if c in used_colors]
        unused_allowed = [c for c in allowed_colors if c not in used_colors]
        
        # Try used colors first (they are more likely to satisfy "repetition" constraints)
        # Sort used colors by frequency descending? Or just arbitrary?
        # Let's try arbitrary for now, or maybe sorted.
        # Actually, using a color that is already frequent might be safer.
        
        candidates = used_allowed + unused_allowed
        
        for c in candidates:
            colors[best_edge] = c
            if backtrack(best_edge, new_unassigned):
                return True
            colors[best_edge] = -1
        
        return False

    # Initial check: Is the forced assignment valid?
    # We forced colors[0..K-1] = 0..K-1.
    # We must ensure no cycle among these edges is violated.
    # Since they are all distinct, a cycle is violated if it contains 4 of these edges.
    # We can just run the backtracker starting from this state.
    # But wait, if the initial state is invalid, backtrack will fail immediately?
    # Not necessarily, because backtrack only checks constraints when assigning a NEW edge.
    # It does NOT re-check constraints for edges that were already assigned.
    # So we MUST verify the initial forced assignment is valid.
    
    # Check validity of forced edges
    valid_init = True
    for ci in range(NUM_CYCLES):
        c_edges = CYCLES[ci]
        cols = [colors[e] for e in c_edges]
        if all(c != -1 for c in cols):
            if len(set(cols)) == 4:
                valid_init = False
                break
        # If some are -1, it's not a violation yet.
    
    if not valid_init:
        return None
    
    # Run backtracking
    if backtrack(-1, unassigned):
        return colors
    else:
        return None

def _count_distinct(colors):
    return len(set(c for c in colors if c != -1))

def _is_valid_full(colors):
    """Final verification."""
    for ci in range(NUM_CYCLES):
        c_edges = CYCLES[ci]
        cols = [colors[e] for e in c_edges]
        if len(set(cols)) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_colors = None
    best_distinct = 0
    
    # We will try to find a coloring with K distinct colors, starting from a reasonable estimate.
    # The current best is 17. Let's try to go higher.
    # We can binary search K or just linearly increase.
    # Given the structure, K=20+ might be hard.
    
    # First, establish a baseline valid coloring.
    # A 2-coloring is always valid? No.
    # But we can construct a valid one quickly.
    # Let's use the algebraic coloring (i+j)%2 which is valid?
    # (i+j)%2: 
    # Cycle a-b-c-d. Colors: (a+b)%2, (b+c)%2, (c+d)%2, (d+a)%2.
    # Sum of colors = 2(a+b+c+d)%2 = 0.
    # If 4 distinct, sum is not 0? No, colors are 0 or 1.
    # Max distinct is 2. So it's valid.
    
    # Let's start with K=18, 19, 20...
    
    # To save time, we can try a few different edge orderings for the forced distinct colors.
    # The current code forces edges 0..K-1.
    # Edge 0 is (0,1). Edge 1 is (0,2). Edge 2 is (0,3)...
    # These edges share vertex 0. This creates a "star" of distinct colors.
    # This might be a good structure to start with.
    
    for K in range(18, 30):
        if time.time() > deadline - 5.0:
            break
            
        print(f"Trying K={K}", file=sys.stderr)
        
        result = _backtracking_search(K, deadline)
        
        if result is not None:
            # Verify
            if _is_valid_full(result):
                distinct = _count_distinct(result)
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = result.copy()
                    print(f"Found valid coloring with {distinct} distinct colors", file=sys.stderr)
                    # If we found one, we can try to go higher immediately?
                    # The loop increments K, so it will try K+1 next.
            else:
                print("Verification failed!", file=sys.stderr)
        else:
            print(f"Failed to find coloring for K={K}", file=sys.stderr)
            # If it fails for K, it might fail for K+1? 
            # Not necessarily, but often yes.
            # We can break if we fail for several consecutive Ks?
            # Let's just continue, but maybe skip to a higher K if we have time?
            # No, backtracking is exponential. If K=18 is hard, K=19 is harder.
            # If we time out on K=18, we should stop.
            # But _backtracking_search doesn't have an internal time check yet.
            # We need to add a time check inside the backtracker to abort early.
            
            # Let's modify _backtracking_search to accept a deadline and abort.
            # For now, if it returns None quickly, we continue.
            # If it hangs, we are in trouble.
            # We should add a step counter or time check in backtrack.
            
            # Let's assume the previous call returned quickly or found a solution.
            # If it returned None, it means it searched the whole space (or pruned everything) and failed.
            # This is dangerous if the space is huge.
            # We MUST add a time limit to the backtracker.

    # Fallback: If backtracking didn't find anything better, use a simple valid coloring.
    if best_colors is None:
        # Generate a valid 2-coloring
        best_colors = [0] * NEDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = IDX_MAP[(i, j)]
                best_colors[idx] = (i + j) % 2
        best_distinct = 2
    
    return best_colors

# We need to update _backtracking_search to handle time limits.
# Let's rewrite it with a time check.

def _backtracking_search_timed(target_distinct, deadline):
    """
    Attempt to find a valid coloring using exactly 'target_distinct' distinct colors.
    """
    colors = [-1] * NEDGES
    
    for i in range(target_distinct):
        colors[i] = i
        
    # Check initial validity
    for ci in range(NUM_CYCLES):
        c_edges = CYCLES[ci]
        cols = [colors[e] for e in c_edges]
        if all(c != -1 for c in cols):
            if len(set(cols)) == 4:
                return None

    unassigned = list(range(target_distinct, NEDGES))
    steps = 0
    max_steps = 50000 # Limit steps to prevent infinite loops
    
    def backtrack(edge_idx, unassigned_edges):
        nonlocal steps
        steps += 1
        if steps > max_steps:
            return False
        if time.time() > deadline:
            return False
            
        if not unassigned_edges:
            return True
        
        best_edge = -1
        max_constrained = -1
        
        for e in unassigned_edges:
            count = 0
            for ci in EDGE_TO_CYCLES[e]:
                c_edges = CYCLES[ci]
                assigned_count = 0
                for pos in range(4):
                    if c_edges[pos] == e:
                        continue
                    if colors[c_edges[pos]] != -1:
                        assigned_count += 1
                if assigned_count == 3:
                    count += 1
            if count > max_constrained:
                max_constrained = count
                best_edge = e
        
        if best_edge == -1:
            best_edge = unassigned_edges[0]
        
        new_unassigned = [e for e in unassigned_edges if e != best_edge]
        
        allowed_colors = set(range(MAX_COLOR))
        
        for ci in EDGE_TO_CYCLES[best_edge]:
            c_edges = CYCLES[ci]
            other_colors = []
            all_assigned = True
            for pos in range(4):
                if c_edges[pos] == best_edge:
                    continue
                col = colors[c_edges[pos]]
                if col == -1:
                    all_assigned = False
                    break
                other_colors.append(col)
            
            if all_assigned:
                if len(set(other_colors)) == 3:
                    allowed_colors.intersection_update(set(other_colors))
        
        if not allowed_colors:
            return False
        
        used_colors = set(colors)
        used_allowed = [c for c in allowed_colors if c in used_colors]
        unused_allowed = [c for c in allowed_colors if c not in used_colors]
        
        candidates = used_allowed + unused_allowed
        
        for c in candidates:
            colors[best_edge] = c
            if backtrack(best_edge, new_unassigned):
                return True
            colors[best_edge] = -1
        
        return False

    if backtrack(-1, unassigned):
        return colors
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_colors = None
    best_distinct = 0
    
    # Try increasing K
    for K in range(18, 25):
        if time.time() > deadline - 5.0:
            break
            
        result = _backtracking_search_timed(K, deadline)
        
        if result is not None:
            if _is_valid_full(result):
                distinct = _count_distinct(result)
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = result.copy()
            else:
                pass
        else:
            # If it failed, it might be due to time or no solution.
            # If time.time() is close to deadline, break.
            if time.time() > deadline - 2.0:
                break

    if best_colors is None:
        best_colors = [0] * NEDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = IDX_MAP[(i, j)]
                best_colors[idx] = (i + j) % 2
        best_distinct = 2
    
    return best_colors

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
    print(len(set(res)))
