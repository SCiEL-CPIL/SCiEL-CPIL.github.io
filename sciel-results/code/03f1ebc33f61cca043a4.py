"""Minimal maximal no-isosceles set via priority-based greedy with blocking coverage."""
import itertools
import numpy as np

N = 16


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    # Precompute for each point p, which pairs (i,j) make (p, point_i, point_j) isosceles
    # But that's O(n^3) = 256^3 ~ 16M, feasible but let's be smart.
    
    # For a candidate set S, a point p outside S is blocked if ∃a,b in S with _is_isosceles(p,a,b)
    
    # Greedy: maintain S, and track which points are blocked.
    # At each step, pick a point that (1) doesn't create isosceles with any pair in S,
    # and (2) blocks the most currently unblocked points.
    
    chosen = []
    chosen_set = set()
    
    # Track for each point not in chosen, whether it's blocked
    # blocked[p] = True if some pair in chosen makes isosceles with p
    
    # We'll maintain a list of "free" points (not in chosen, not yet considered blocked)
    # Actually, let's just recompute blocking efficiently.
    
    # For efficiency: when we add a new point c to chosen, it can block points p where
    # for some a in chosen_before, _is_isosceles(p, c, a) is True.
    
    # Let's maintain: for each point p, the count of "blocking pairs" in chosen that block p.
    # But pairs are what matter. Let's track for each p, the number of pairs (a,b) in chosen
    # such that _is_isosceles(p, a, b).
    
    # Simpler: just check on the fly. With N=16, 256 points, and greedy picking maybe 20-40 points,
    # the inner checks are manageable.
    
    def can_add(p, current_chosen):
        """Check if p can be added without creating an isosceles triangle."""
        for a, b in itertools.combinations(current_chosen, 2):
            if _is_isosceles(p, a, b):
                return False
        return True
    
    def count_blocked_by(p, current_chosen):
        """Count how many points q (not in current_chosen, q != p) are blocked by adding p to current_chosen."""
        # q is blocked if ∃a,b in (current_chosen ∪ {p}) such that _is_isosceles(q, a, b)
        # But we only care about NEW blocks from adding p.
        # A point q is newly blocked if:
        #   - q was not blocked before (by current_chosen alone)
        #   - q is blocked after (by current_chosen ∪ {p})
        # This is complex. Let's just compute total blocked after.
        
        # For simplicity in greedy, let's just count how many points q would be blocked
        # that aren't already in chosen.
        blocked_count = 0
        for q in all_points:
            if q in current_chosen or q == p:
                continue
            # Check if q is blocked by current_chosen ∪ {p}
            is_blocked = False
            for a, b in itertools.combinations(current_chosen + [p], 2):
                if _is_isosceles(q, a, b):
                    is_blocked = True
                    break
            if is_blocked:
                blocked_count += 1
        return blocked_count
    
    # This is too slow. Let me use a different strategy.
    
    # Alternative: use a smarter greedy.
    # Start with empty set.
    # At each step, among all points that can be added (no isosceles with existing pairs),
    # pick the one that blocks the most other points.
    
    # To make this efficient, precompute isosceles relationships.
    
    # Precompute: for each triple (i,j,k), whether it's isosceles. Store as a set of frozensets.
    # Actually, let's precompute for each pair (i,j), the set of points k that form isosceles with i,j.
    
    # iso_with_pair[(i,j)] = set of k such that (k, point_i, point_j) is isosceles
    # This is O(n^3) = 16M operations, which is fine.
    
    point_index = {p: i for i, p in enumerate(all_points)}
    
    # For each pair (i, j), compute which points k form isosceles with points[i], points[j]
    # iso_pair[i][j] = bitmask or set of k
    # Using bitmasks for speed
    
    n = n_points
    # iso_pair[i][j] as a Python int bitmask of length n
    iso_pair = [[0] * n for _ in range(n)]
    
    for i in range(n):
        pi = all_points[i]
        for j in range(i + 1, n):
            pj = all_points[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = all_points[k]
                if _is_isosceles(pi, pj, pk):
                    mask |= (1 << k)
            iso_pair[i][j] = mask
            iso_pair[j][i] = mask
    
    # Now, for a set S (as a set of indices), a point q is blocked if:
    # ∃i,j in S such that bit q is set in iso_pair[i][j]
    
    # For greedy:
    # - Maintain chosen as a list of indices
    # - Maintain a bitmask `blocked_mask` of all points that are blocked
    # - When we add a new point c to chosen, update blocked_mask:
    #   For each existing a in chosen, new_blocked = iso_pair[a][c]
    #   blocked_mask |= new_blocked
    #   Also, c itself should not be in blocked_mask (it's chosen)
    
    chosen = []
    chosen_mask = 0  # bitmask of chosen points
    blocked_mask = 0  # bitmask of blocked points
    
    # Free points: not chosen and not blocked
    # Initially all free
    
    while True:
        # Check if all non-chosen points are blocked
        all_non_chosen_mask = ((1 << n) - 1) & ~chosen_mask
        if (all_non_chosen_mask & ~blocked_mask) == 0:
            # All non-chosen points are blocked, so S is maximal
            break
        
        # Find candidate points: not chosen, and can be added (no isosceles with any pair in chosen)
        # A point c can be added if: for all pairs (i,j) in chosen, bit c is NOT in iso_pair[i][j]
        # i.e., c is not blocked by any pair in chosen... wait, that's the same as c not being in blocked_mask?
        
        # Actually: c can be added iff c is NOT blocked by chosen.
        # c is blocked by chosen iff ∃i,j in chosen such that bit c in iso_pair[i][j]
        # Which is exactly: bit c is in blocked_mask.
        
        # So candidates are: not chosen, not blocked.
        candidate_mask = all_non_chosen_mask & ~blocked_mask
        
        if candidate_mask == 0:
            # No candidates but not all blocked? That shouldn't happen if our logic is right.
            # Actually, if a point is not blocked, it should be a candidate.
            # Let's break to avoid infinite loop.
            break
        
        # Among candidates, pick the one that blocks the most new points.
        # For each candidate c, new_blocked = union of iso_pair[a][c] for a in chosen
        # new_points_blocked = new_blocked & ~blocked_mask & ~chosen_mask
        # Score = popcount(new_points_blocked)
        
        best_c = -1
        best_score = -1
        
        # Iterate through candidates
        c = 0
        while c < n:
            if (candidate_mask >> c) & 1:
                # Compute new blocked points if we add c
                new_blocked = 0
                for a in chosen:
                    new_blocked |= iso_pair[a][c]
                # New points that become blocked (not already blocked, not chosen)
                new_points = new_blocked & ~blocked_mask & ~chosen_mask
                score = bin(new_points).count('1')
                if score > best_score:
                    best_score = score
                    best_c = c
            c += 1
        
        if best_c < 0:
            break
        
        # Add best_c to chosen
        chosen.append(best_c)
        chosen_mask |= (1 << best_c)
        
        # Update blocked_mask
        for a in chosen[:-1]:  # all previously chosen
            blocked_mask |= iso_pair[a][best_c]
        
        # Make sure chosen points are not in blocked_mask (they're chosen, not blocked)
        blocked_mask &= ~chosen_mask
    
    # Convert back to (x,y) points
    result = [all_points[i] for i in chosen]
    return result
