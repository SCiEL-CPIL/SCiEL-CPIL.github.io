import time
import itertools
import random

# Precompute: for each x in 1..192, and each triple (i,j,k) of distinct elements,
# store whether x divides their sum. We'll use this for fast checking.

def _precompute_divisibility():
    """Precompute divisibility tables for fast constraint checking."""
    # div_sum3[x] = set of (i,j,k) tuples where x divides sum of elements[i]+elements[j]+elements[k]
    # But we need this dynamically as selected changes. Instead, precompute:
    # For each x, for each pair (u,v), compute (u+v) % x
    # Then checking if x divides u+v+w becomes: (u+v+w) % x == 0
    
    # Actually, let's precompute all triple sums modulo each x
    # div_table[x][a][b][c] = 1 if x divides a+b+c else 0
    # This is 192 * 192^3 which is too large.
    
    # Better: for each x, precompute pair_sums_mod[x][a][b] = (a+b) % x
    # Then x divides a+b+c iff (a+b)%x + c ≡ 0 (mod x) iff (a+b)%x == (-c) % x
    
    # Even better: precompute for each x, for each c, the set of pairs (a,b) such that (a+b)%x == (-c)%x
    # This is complex. Let's use a different approach.
    
    # For fast checking, we can precompute:
    # conflict[x][a] = set of values b such that x divides a+b+c for some c... no.
    
    # Let me think differently. The constraint is:
    # 1. x does not divide sum of any 3 distinct elements in selected
    # 2. For each a in selected, a does not divide x + u + v for any u,v in selected\{a}
    
    # Precompute: for each x in 1..192, for each a in 1..192:
    #   pair_conflicts[x][a] = set of (u,v) pairs such that a divides x+u+v
    #   But u,v are from selected, so we need to check dynamically.
    
    # Alternative: precompute for each a, for each x, the residues of pairs.
    # For fixed a and x, a divides x+u+v iff (x+u+v) % a == 0 iff (u+v) % a == (-x) % a
    # So for fixed a, precompute pair_residues[a][r] = set of pairs (u,v) with (u+v)%a == r
    # Then checking is: is there a pair (u,v) in selected\{a} such that (u+v)%a == (-x)%a?
    
    # This is still O(n^2) per check. Let's just optimize the inner loops.
    
    # Actually, the key insight: we can precompute for each a in 1..192:
    #   For each residue r in 0..a-1, store which pairs (u,v) have (u+v)%a == r
    # But storing all pairs is 192^2 per residue, times 192 residues per a, times 192 a's = too much.
    
    # Let's just use numpy for vectorized checks.
    pass

def _can_add_fast(x, selected, n):
    """Check if x can be added to selected without violating the constraint.
    Uses vectorized operations for speed."""
    if len(selected) < 3:
        # Only need to check constraint 2 (no a in selected divides x+u+v since need 2 others)
        if len(selected) >= 2:
            for a_idx, a in enumerate(selected):
                others = [selected[i] for i in range(n) if i != a_idx]
                # Check if a divides x + u + v for any pair u,v in others
                # (x+u+v) % a == 0
                target = (-x) % a
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        if (others[i] + others[j]) % a == target:
                            return False
        return True
    
    # Constraint 1: x must not divide sum of any 3 distinct elements in selected
    # Precompute all triple sums and check modulo x
    sel = selected
    n_sel = len(sel)
    
    # Check constraint 1: for all triples i<j<k, (sel[i]+sel[j]+sel[k]) % x != 0
    # We can do this more efficiently:
    # For each pair (i,j), compute (sel[i]+sel[j]) % x, then for each k not in {i,j},
    # check if (sel[i]+sel[j]+sel[k]) % x == 0
    # i.e., sel[k] % x == -(sel[i]+sel[j]) % x
    
    # Build a count of residues
    residue_count = {}
    for val in sel:
        r = val % x
        residue_count[r] = residue_count.get(r, 0) + 1
    
    # For each pair (i,j), check if there exists k != i,j with sel[k] % x == -(sel[i]+sel[j])%x
    # But we need to be careful: k must be distinct from i and j.
    
    # Simpler approach: just iterate over triples but break early
    for i in range(n_sel):
        for j in range(i+1, n_sel):
            partial = (sel[i] + sel[j]) % x
            needed = (-partial) % x
            for k in range(j+1, n_sel):
                if sel[k] % x == needed:
                    return False
    
    # Constraint 2: for each a in selected, a must not divide x + u + v for any u,v in selected\{a}
    for a_idx, a in enumerate(sel):
        target = (-x) % a
        # Check all pairs (u,v) in sel\{a}
        for i in range(n_sel):
            if i == a_idx:
                continue
            for j in range(i+1, n_sel):
                if j == a_idx:
                    continue
                if (sel[i] + sel[j]) % a == target:
                    return False
    
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(x, selected, len(selected)):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve the set by adding or swapping elements."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any element not in selected
        in_set = set(selected)
        for x in range(1, 193):
            if time.time() > deadline:
                return selected
            if x not in in_set and _can_add_fast(x, selected, len(selected)):
                selected.append(x)
                in_set.add(x)
                improved = True
                break
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected
                selected_without = selected[:i] + selected[i+1:]
                for x in range(1, 193):
                    if x in in_set:
                        continue
                    if _can_add_fast(x, selected_without, len(selected_without)):
                        selected = selected_without + [x]
                        in_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 90s
    
    best = []
    
    # Strategy 1: Greedy descending
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 5:
        result = _greedy(range(1, 193), deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Random shuffles with more iterations (faster checks enable more)
    random.seed(42)
    num_shuffles = 30
    for s in range(num_shuffles):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Local search on best
    if time.time() < deadline - 5:
        best = _local_search(best, deadline)
    
    return best
