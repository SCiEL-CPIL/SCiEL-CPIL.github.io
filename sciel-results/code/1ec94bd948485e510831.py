"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from itertools import combinations

def construct():
    N = 128
    deadline = time.time() + 50  # safety margin
    
    # Precompute all bad triples: (a,b,c) with a<b<c, gcd(a,b)==gcd(b,c)==gcd(c,a)
    # For each element, precompute which pairs it forms bad triples with
    # bad_pairs[x] = set of frozensets or sorted tuples (y,z) such that {x,y,z} is bad
    # Better: for each x, bad_with[x] = set of (y,z) pairs where y,z != x and {x,y,z} bad
    
    # Actually, let's precompute a conflict matrix:
    # conflict[x][y] = set of z such that {x,y,z} is a bad triple
    # This is O(N^3) = 128^3 ~ 2 million, fine.
    
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd(a, b)
            for c in range(b + 1, N + 1):
                gbc = gcd(b, c)
                if gbc != gab:
                    continue
                gca = gcd(c, a)
                if gca == gab:
                    conflict[a].add(c)
                    conflict[b].add(c)
                    conflict[c].add(a)
                    conflict[c].add(b)
    
    def is_valid(selected):
        """Check if selected set has no bad triple."""
        sel = list(selected)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                # Check if any c in sel (c != a, b) forms bad triple with a,b
                for c in sel:
                    if c == a or c == b:
                        continue
                    if c in conflict[a] and c in conflict[b]:
                        # Need to verify it's actually a bad triple
                        # conflict[a] contains c means there exists some bad triple with a and c
                        # But we need the specific triple {a,b,c} to be bad
                        pass
                # This approach is flawed. Let me use direct check.
        # Direct check for small sets
        for a, b, c in combinations(sel, 3):
            if gcd(a, b) == gcd(b, c) == gcd(c, a):
                return False
        return True
    
    def is_valid_fast(selected_set):
        """Check validity using precomputed gcd table."""
        sel = list(selected_set)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                gab = gcd(a, b)
                for k in range(j + 1, len(sel)):
                    c = sel[k]
                    if gcd(b, c) == gab and gcd(c, a) == gab:
                        return False
        return True
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a bad triple."""
        sel = list(selected_set)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                gab = gcd(a, b)
                if gcd(b, x) == gab and gcd(x, a) == gab:
                    return False
        return True
    
    def greedy_build(order):
        """Greedy build following given order."""
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    def local_search(selected_set, time_limit=5.0):
        """Local search: try to add elements, swap elements."""
        selected = set(selected_set)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element not in selected
            for x in range(1, N + 1):
                if x not in selected:
                    if can_add(selected, x):
                        selected.add(x)
                        improved = True
                        break
            # If we can't add, try swap: remove one, add another
            if not improved:
                for x in list(selected):
                    remaining = selected - {x}
                    for y in range(1, N + 1):
                        if y not in selected:
                            if can_add(remaining, y):
                                if len(remaining) + 1 > len(selected):
                                    selected = remaining | {y}
                                    improved = True
                                    break
                    if improved:
                        break
        return selected
    
    best = set()
    
    # Multiple restarts with different orderings
    base_order = list(range(1, N + 1))
    
    # Try natural order first
    result = greedy_build(base_order)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)
    
    # Random restarts
    restart_count = 0
    while time.time() < deadline:
        random.shuffle(base_order)
        result = greedy_build(base_order)
        result = local_search(result, time_limit=2.0)
        if len(result) > len(best):
            best = set(result)
        restart_count += 1
        # If we've done enough restarts and no improvement in a while, break
        if restart_count > 100 and len(best) >= 80:
            break
    
    return list(best) if best else [1]
# EVOLVE-BLOCK-END
