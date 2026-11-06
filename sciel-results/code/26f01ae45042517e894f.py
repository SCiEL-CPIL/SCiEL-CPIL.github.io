import random
import time
import math
import numpy as np
from collections import defaultdict

N = 64
NUM_POINTS = N * N

# Precompute all grid points as a list of (x, y) tuples
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

# Precompute conflict graph: two points conflict if they can form an isosceles triangle
# with some third grid point. We use bitsets for fast neighborhood queries.
# conflict_mask[i] is a bitset where bit j is 1 if point i conflicts with point j

def precompute_conflicts():
    """Precompute the conflict graph using bitsets."""
    # For each pair of points, check if they form an isosceles triangle with any third point
    # Two points A and B conflict if there exists C such that:
    #   d(A,C) == d(B,C)  [A and B are base, C is apex]
    #   d(A,C) == d(A,B)  [A is apex, B and C are base]
    #   d(B,C) == d(A,B)  [B is apex, A and C are base]
    
    points_arr = np.array(ALL_POINTS, dtype=np.int32)
    
    # Compute all pairwise squared distances
    # d2[i][j] = squared distance between point i and point j
    diffs = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
    d2 = np.sum(diffs ** 2, axis=2)  # shape (NUM_POINTS, NUM_POINTS)
    
    # For efficiency, we'll use a different approach:
    # For each point C, find all pairs (A, B) such that d(A,C) == d(B,C)
    # These pairs conflict (C is the apex)
    # Also for each point A and distance d, find all B with d(A,B) == d,
    # then all C with d(A,C) == d conflict with B (A is apex)
    
    conflict_masks = [0] * NUM_POINTS
    
    # Method: For each apex C, group points by their distance to C
    # Points in the same group conflict with each other
    for C in range(NUM_POINTS):
        # Group points by distance to C
        dist_to_C = d2[:, C]
        groups = defaultdict(list)
        for A in range(NUM_POINTS):
            if A == C:
                continue
            groups[dist_to_C[A]].append(A)
        
        # For each group with multiple points, all pairs conflict
        for dist, group in groups.items():
            if len(group) >= 2:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        A = group[i]
                        B = group[j]
                        conflict_masks[A] |= (1 << B)
                        conflict_masks[B] |= (1 << A)
    
    # Additionally, for each point A as apex:
    # If d(A, B) == d(A, C), then B and C conflict (already handled above)
    # If d(A, B) == d(B, C), then A and C conflict
    # If d(A, B) == d(A, C), then B and C conflict (same as above)
    
    # The above handles the case where C is apex.
    # Now handle: A is apex, so d(A,B) == d(A,C) means B and C conflict.
    # This is already handled by the loop above when we consider A as the "C" apex.
    
    # Wait, I need to be more careful. Let me reconsider.
    # An isosceles triangle has two equal sides. The three cases are:
    # 1. d(A,C) == d(B,C): C is apex, A and B are base -> A and B conflict
    # 2. d(A,C) == d(A,B): A is apex, B and C are base -> B and C conflict  
    # 3. d(B,C) == d(A,B): B is apex, A and C are base -> A and C conflict
    
    # My loop above handles case 1: for each C, points equidistant from C conflict.
    # But cases 2 and 3 are different.
    
    # Let me redo this more carefully.
    # Actually, the loop above only handles case 1.
    # For case 2: A is apex. d(A,B) == d(A,C). This means B and C are at the same distance from A.
    # This is the same as saying: for apex A, B and C are in the same distance group.
    # So B and C conflict. This is exactly what my loop does when C=A in the loop variable.
    # Wait no, in my loop I have C as the apex, and I'm grouping points by distance to C.
    # So if A is the apex, then in the loop when C=A, I group all points by distance to A.
    # Points B and C in the same group (same distance to A) conflict.
    # So yes, my loop handles all three cases! Because:
    # - Case 1: C is apex, A and B same distance to C -> handled when loop var = C
    # - Case 2: A is apex, B and C same distance to A -> handled when loop var = A
    # - Case 3: B is apex, A and C same distance to B -> handled when loop var = B
    
    # So the loop is correct.
    
    return conflict_masks

# Precompute at module level
CONFLICT_MASKS = precompute_conflicts()

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    best_set = []
    best_size = 0
    
    # Fast validity check using bitsets
    def is_valid_addition_fast(P_idx, selected_mask):
        """Check if point P_idx can be added given the set of selected points (as bitset)."""
        # P_idx conflicts with any selected point if (conflict_masks[P_idx] & selected_mask) != 0
        return (CONFLICT_MASKS[P_idx] & selected_mask) == 0
    
    # Greedy construction with a given order
    def greedy_construction(order):
        """Build a set by greedily adding points in the given order."""
        S = []
        selected_mask = 0
        for P in order:
            P_idx = P[0] * N + P[1]
            if is_valid_addition_fast(P_idx, selected_mask):
                S.append(P)
                selected_mask |= (1 << P_idx)
        return S, selected_mask
    
    # Local search: try to improve by adding points
    def local_search(S, selected_mask, time_limit):
        """Try to improve set S by adding more points."""
        S = list(S)
        selected_mask = selected_mask
        
        # Repeatedly try to add a point
        added_any = True
        while added_any and time.time() < time_limit:
            added_any = False
            # Find candidates: points not in selected_mask
            candidates = []
            for idx in range(NUM_POINTS):
                if not (selected_mask & (1 << idx)):
                    x = idx // N
                    y = idx % N
                    candidates.append((x, y))
            
            random.shuffle(candidates)
            for P in candidates:
                if time.time() >= time_limit:
                    break
                P_idx = P[0] * N + P[1]
                if is_valid_addition_fast(P_idx, selected_mask):
                    S.append(P)
                    selected_mask |= (1 << P_idx)
                    added_any = True
                    break  # restart after adding
        
        return S, selected_mask
    
    # Structured randomization: create different ordering strategies
    def create_orderings(num_orders):
        """Create various random/structured orderings."""
        orderings = []
        
        # Pure random
        n_random = max(1, num_orders // 3)
        for _ in range(n_random):
            order = ALL_POINTS[:]
            random.shuffle(order)
            orderings.append(order)
        
        # Row-major with random row permutation
        n_row = max(1, num_orders // 3)
        for _ in range(n_row):
            rows = list(range(N))
            random.shuffle(rows)
            order = []
            for r in rows:
                cols = list(range(N))
                random.shuffle(cols)
                for c in cols:
                    order.append((c, r))
            orderings.append(order)
        
        # Diagonal-based ordering
        n_diag = max(1, num_orders // 3)
        for _ in range(n_diag):
            # Order by (x+y) mod N, then random within each diagonal
            diag_order = {}
            for x in range(N):
                for y in range(N):
                    d = (x + y) % N
                    if d not in diag_order:
                        diag_order[d] = []
                    diag_order[d].append((x, y))
            
            diags = list(range(N))
            random.shuffle(diags)
            order = []
            for d in diags:
                pts = diag_order.get(d, [])
                random.shuffle(pts)
                order.extend(pts)
            orderings.append(order)
        
        return orderings
    
    # Main search loop
    num_trials = 0
    while time.time() < deadline:
        # Create batch of orderings
        remaining = deadline - time.time()
        if remaining < 5:
            break
        batch_size = min(5, max(1, int(remaining / 30)))
        orderings = create_orderings(batch_size)
        
        for order in orderings:
            if time.time() >= deadline - 10:
                break
            
            S, selected_mask = greedy_construction(order)
            
            # Apply local search if we have time
            if time.time() < deadline - 15:
                S, selected_mask = local_search(S, selected_mask, deadline - 10)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            
            num_trials += 1
    
    return best_set
