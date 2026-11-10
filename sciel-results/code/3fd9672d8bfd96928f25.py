"""Constraint-based backtracking solver for no-divide-sum-of-three-others task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAX_K = 48

def _precompute_conflicts():
    """
    For each element a in 1..192, compute a bitset of elements x such that
    there exist distinct y, z (all distinct from a and each other) with a | (x+y+z).
    
    Actually, we need: for a given set S containing a, x is "conflicting" with a
    if there exist y, z in S \\ {a, x} such that a | (x + y + z).
    
    This depends on S, so we can't fully precompute. But we can precompute
    for each a, and for each pair (x, y), what residue z needs.
    
    Better approach: precompute for each a, a mapping from residue r to the set
    of elements with that residue mod a. Then during search, for a candidate x
    to add to set S (which already contains a), we check if there exist y, z in S
    (excluding a and x) such that (x + y + z) % a == 0.
    
    For efficiency, we'll precompute:
    - For each a, for each pair (i, j) where i, j are indices in 1..192,
      the needed residue: need = (-(vals[i] + vals[j])) % a
    - Then during search, we maintain for each a in S, a count of how many
      elements in S have each residue mod a.
    """
    # Precompute: for each a, a list of (residue, [values]) 
    # Actually, let's precompute a 2D structure:
    # conflict[a][x] = set of pairs (y, z) such that a | (x + y + z) and y != z, y != x, z != x
    # This is too large. 
    # 
    # Alternative: For each a, precompute residue_map: residue -> sorted list of values
    # Then during search, for each a in selected, and candidate x, we need to check
    # if there exist y, z in selected (excluding a, x) with (x+y+z) % a == 0.
    # 
    # Let's precompute for each a:
    #   a_residues[a][r] = set of values v in 1..192 where v % a == r
    # During search, for a in selected, we maintain count_a[r] = number of elements in selected
    # (excluding a itself) with residue r mod a.
    # For candidate x, needed = (-(x + y)) % a for some y... no, we need to check if
    # there exist y, z in selected\{a,x} with (x+y+z) % a == 0.
    # This is equivalent to: is there a pair (y,z) in selected\{a,x} such that (y+z) % a == (-x) % a?
    # 
    # Precompute for each a: pair_sums[a][s] = set of sums mod a achievable by pairs from 1..192
    # But this is still complex.
    #
    # Let's use a different approach: precompute for each a, and for each value x,
    # the set of "witness pairs" (y, z) such that a | (x + y + z).
    # Then during search, when we have selected set S, and we want to add x,
    # for each a in S, we check if there exist y, z in S\{a, x} such that (y, z) is a witness pair for (a, x).
    #
    # Precompute: witnesses[a][x] = set of (y, z) pairs
    # This is 192 * 192 * (number of pairs) which could be large but manageable.
    
    witnesses = {}  # witnesses[a][x] = set of (y, z)
    for a in range(1, N + 1):
        witnesses[a] = {}
        for x in range(1, N + 1):
            if x == a:
                witnesses[a][x] = set()
                continue
            wp = set()
            # Find all pairs (y, z) with y < z, y != x, z != x, y != a, z != a, such that (x + y + z) % a == 0
            needed_mod = (-x) % a
            for y in range(1, N + 1):
                if y == x or y == a:
                    continue
                for z in range(y + 1, N + 1):
                    if z == x or z == a:
                        continue
                    if (x + y + z) % a == 0:
                        wp.add((y, z))
            witnesses[a][x] = wp
    return witnesses

def _precompute_pair_lookup():
    """
    Precompute for each a, a mapping from (y, z) to whether they form a valid pair
    for each x. Actually, let's precompute:
    pair_exists[a][y][z] = set of x values such that a | (x + y + z)
    """
    # pair_exists[a] is a 2D array: pair_exists[a][y][z] = set of x
    pair_exists = {}
    for a in range(1, N + 1):
        pe = {}
        for y in range(1, N + 1):
            pe[y] = {}
            for z in range(y + 1, N + 1):
                if y == a or z == a:
                    pe[y][z] = set()
                    continue
                needed = (-y - z) % a
                # x must satisfy x % a == needed, x != a, x != y, x != z
                xs = set()
                for x in range(1, N + 1):
                    if x == a or x == y or x == z:
                        continue
                    if x % a == needed:
                        xs.add(x)
                pe[y][z] = xs
        pair_exists[a] = pe
    return pair_exists

def _is_valid_full(selected):
    """Check if the full set is valid."""
    s = set(selected)
    k = len(s)
    if k < 4:
        return True
    lst = list(s)
    for a in lst:
        others = [x for x in lst if x != a]
        n = len(others)
        others_set = set(others)
        residue_map = {}
        for x in others:
            r = x % a
            if r not in residue_map:
                residue_map[r] = set()
            residue_map[r].add(x)
        
        found = False
        for i in range(n):
            x = others[i]
            for j in range(i + 1, n):
                y = others[j]
                needed = (-(x + y)) % a
                if needed in residue_map:
                    candidates = residue_map[needed]
                    for z in candidates:
                        if z != x and z != y:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if found:
            return False
    return True

def _greedy_fast(order, limit=48):
    """
    Greedy construction with fast incremental validity check.
    For each candidate x to add to selected, check against each a in selected:
    does there exist y, z in selected\{a, x} such that a | (x + y + z)?
    """
    selected = []
    selected_set = set()
    for x in order:
        if len(selected) >= limit:
            break
        # Check if x can be added
        valid = True
        for a in selected:
            # Check if there exist y, z in selected\{a, x} such that a | (x + y + z)
            # i.e., (y + z) % a == (-x) % a
            needed = (-x) % a
            found = False
            others = [v for v in selected if v != a and v != x]
            # Use a set for fast lookup
            others_set = set(others)
            # For each y in others, check if there exists z in others\{y} with (y + z) % a == needed
            # i.e., z % a == (needed - y % a) % a
            residue_count = {}
            for v in others:
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            for y in others:
                r_y = y % a
                r_z_needed = (needed - r_y) % a
                # Need z in others, z != y, with z % a == r_z_needed
                count = residue_count.get(r_z_needed, 0)
                if count > 0:
                    # Check if there's a z != y with this residue
                    if r_z_needed == r_y:
                        # Need at least 2 elements with this residue, one is y
                        if count >= 2:
                            found = True
                            break
                    else:
                        found = True
                        break
            if found:
                valid = False
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, rng, deadline):
    """Local search: try to add, swap, or remove elements to improve."""
    selected = list(selected)
    best = selected
    best_len = len(selected)
    
    while time.time() < deadline:
        improved = False
        # Try adding
        available = [v for v in range(1, N + 1) if v not in set(selected)]
        rng.shuffle(available)
        for x in available:
            if time.time() >= deadline:
                break
            if len(selected) >= MAX_K:
                break
            candidate = selected + [x]
            if _is_valid_full(candidate):
                selected = candidate
                best = candidate
                best_len = len(selected)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add one
        if len(selected) >= 2:
            for i in range(len(selected)):
                if time.time() >= deadline:
                    break
                without = selected[:i] + selected[i + 1:]
                without_set = set(without)
                for x in available:
                    if time.time() >= deadline:
                        break
                    if x in without_set:
                        continue
                    candidate = without + [x]
                    if _is_valid_full(candidate) and len(candidate) > best_len:
                        selected = candidate
                        best = candidate
                        best_len = len(selected)
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Try random perturbation
            if len(selected) >= 2:
                i = rng.randrange(len(selected))
                selected = selected[:i] + selected[i + 1:]
            # Try to add back
            available = [v for v in range(1, N + 1) if v not in set(selected)]
            rng.shuffle(available)
            for x in available[:10]:
                if time.time() >= deadline:
                    break
                candidate = selected + [x]
                if _is_valid_full(candidate):
                    selected = candidate
                    if len(selected) > best_len:
                        best = candidate
                        best_len = len(selected)
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    rng = random.Random(42)
    best = []
    best_len = 0
    
    # Phase 1: Multiple greedy constructions
    orderings = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(N // 2, 0, -1)) + list(range(N, N // 2, -1)),
        list(range(N, N // 2, -1)) + list(range(N // 2, 0, -1)),
    ]
    
    # Add random orderings
    for _ in range(30):
        shuffled = list(range(1, N + 1))
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() >= deadline - 5:
            break
        result = _greedy_fast(order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 2: Local search on best
    if time.time() < deadline - 2:
        improved = _local_search(best, rng, deadline - 1)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Phase 3: More random restarts
    remaining = deadline - time.time()
    if remaining > 5:
        num_restarts = min(50, int(remaining / 2))
        for _ in range(num_restarts):
            if time.time() >= deadline - 2:
                break
            shuffled = list(range(1, N + 1))
            rng.shuffle(shuffled)
            result = _greedy_fast(shuffled)
            if len(result) > best_len:
                best = result
                best_len = len(result)
            if len(result) >= best_len and time.time() < deadline - 3:
                improved = _local_search(result, rng, deadline - 1)
                if len(improved) > best_len:
                    best = improved
                    best_len = len(best)
    
    # Final local search
    if time.time() < deadline - 0.5:
        improved = _local_search(best, rng, deadline)
        if len(improved) > best_len:
            best = improved
    
    return best
# EVOLVE-BLOCK-END
