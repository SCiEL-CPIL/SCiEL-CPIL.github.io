"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Precompute conflict sets: conflict[x] = set of pairs (a,b) such that {a,b,x} is a bad triple
    # More useful: for each x, the set of y such that there exists z with (x,y,z) bad
    # Actually, we need: for a given x and current selected set S, x is incompatible iff
    # there exist a,b in S such that lcm(x,a)==lcm(x,b)==lcm(a,b)
    # Precompute: for each x, and for each lcm value L, the set of y with lcm(x,y)==L
    # Then x conflicts with pair (a,b) in S iff lcm(a,b) == lcm(x,a) == lcm(x,b)
    
    # Alternative precomputation: for each x, compute conflict_pairs[x] = set of (a,b) where a<b and (a,b,x) is bad
    # This is O(n^3) but n=128 so ~2M operations, fine.
    
    # Better: for each x, compute the set of "conflicting partners" in a smarter way.
    # For x to be incompatible with S, we need two elements a,b in S such that:
    #   lcm(x,a) == lcm(x,b) == lcm(a,b)
    # Let L_a = lcm(x,a), L_b = lcm(x,b). We need L_a == L_b == lcm(a,b).
    # Note: lcm(a,b) must equal L_a = lcm(x,a). This means lcm(a,b) | lcm(x,a) and vice versa.
    # Since lcm(x,a) is a multiple of a, and lcm(a,b) is a multiple of a, this is a constraint.
    
    # Precompute for each x: groups[x][L] = list of y such that lcm(x,y) == L
    # Then x is incompatible with S iff there exist L and a,b in S with lcm(a,b)==L and a,b in groups[x][L]
    
    groups = [{} for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            if x == y:
                continue
            L = lcm_table[x][y]
            if L not in groups[x]:
                groups[x][L] = []
            groups[x][L].append(y)
    
    def is_compatible_fast(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for L, y_list in groups[x].items():
            # Count how many elements of y_list are in selected_set
            count = 0
            for y in y_list:
                if y in selected_set:
                    count += 1
                    if count >= 2:
                        # Check if lcm(y1, y2) == L for some pair
                        # Actually we need lcm(a,b) == L where a,b in selected_set and lcm(x,a)==lcm(x,b)==L
                        # We already know lcm(x,a)==L and lcm(x,b)==L, so we need lcm(a,b)==L
                        # Let's just check directly
                        pass
            if count >= 2:
                # Need to verify that for some pair a,b in selected_set ∩ y_list, lcm(a,b) == L
                y_in_selected = [y for y in y_list if y in selected_set]
                for i in range(len(y_in_selected)):
                    for j in range(i+1, len(y_in_selected)):
                        if lcm_table[y_in_selected[i]][y_in_selected[j]] == L:
                            return False
        return True
    
    # Even faster: precompute for each x and each L, the "witness pairs"
    # witness[x][L] = set of (a,b) such that lcm(x,a)==L, lcm(x,b)==L, lcm(a,b)==L
    # Then x is incompatible with S iff there exists (a,b) in witness[x][L] for some L with a,b in S
    
    witness = [{} for _ in range(n + 1)]
    for x in range(1, n + 1):
        for L, y_list in groups[x].items():
            if len(y_list) >= 2:
                pairs = set()
                for i in range(len(y_list)):
                    for j in range(i+1, len(y_list)):
                        a, b = y_list[i], y_list[j]
                        if lcm_table[a][b] == L:
                            pairs.add((min(a,b), max(a,b)))
                if pairs:
                    witness[x][L] = pairs
    
    def is_compatible_witness(x, selected_set):
        """Fast compatibility check using precomputed witness pairs."""
        for L, pairs in witness[x].items():
            for a, b in pairs:
                if a in selected_set and b in selected_set:
                    return False
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if compatible."""
        selected = []
        selected_set = set()
        for x in order:
            if is_compatible_witness(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    def local_improve(selected):
        """Try to improve by adding or swapping elements."""
        selected = list(selected)
        selected_set = set(selected)
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding any missing element
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if x not in selected_set and is_compatible_witness(x, selected_set):
                    selected.append(x)
                    selected_set.add(x)
                    improved = True
            # Try 1-for-2 swaps: remove one element, try to add two
            if time.time() > deadline - 5:
                break
            for a in list(selected):
                if time.time() > deadline - 5:
                    break
                temp_set = selected_set - {a}
                # Try to add elements
                added = []
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if x not in temp_set and is_compatible_witness(x, temp_set):
                        temp_set.add(x)
                        added.append(x)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    selected = list(temp_set)
                    selected_set = temp_set
                    improved = True
                    break
        return selected
    
    # Try multiple orderings
    best = []
    
    # Precompute helper functions for ordering
    def num_divisors(x):
        count = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count
    
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
                d += 1
            else:
                d += 1
        if x > 1:
            count += 1
        return count
    
    def conflict_count(x):
        """Number of witness pairs involving x (lower is better for greedy)."""
        total = 0
        for L, pairs in witness[x].items():
            total += len(pairs)
        return total
    
    # Order 1: natural order
    if time.time() < deadline - 10:
        result = greedy_solve(numbers)
        if len(result) > len(best):
            best = result
    
    # Order 2: reverse
    if time.time() < deadline - 10:
        result = greedy_solve(list(reversed(numbers)))
        if len(result) > len(best):
            best = result
    
    # Order 3: by number of divisors (fewer first)
    if time.time() < deadline - 10:
        order = sorted(numbers, key=num_divisors)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 4: by number of distinct prime factors
    if time.time() < deadline - 10:
        order = sorted(numbers, key=num_distinct_primes)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 5: by conflict count (fewer conflicts first)
    if time.time() < deadline - 10:
        order = sorted(numbers, key=conflict_count)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 6: by conflict count descending (most constrained first)
    if time.time() < deadline - 10:
        order = sorted(numbers, key=conflict_count, reverse=True)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Local improvement on best
    if time.time() < deadline - 10:
        best = local_improve(best)
    
    # Random restarts
    if time.time() < deadline - 15:
        import random
        random.seed(42)
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            order = list(numbers)
            random.shuffle(order)
            result = greedy_solve(order)
            if len(result) > len(best):
                best = result
    
    # Final local improvement
    if time.time() < deadline - 5:
        best = local_improve(best)
    
    return best
# EVOLVE-BLOCK-END
