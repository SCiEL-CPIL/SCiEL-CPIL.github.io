import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build conflict structure: for each element x, store set of (a,b) pairs
    # such that {x,a,b} is a bad triple (all three pairwise LCMs equal)
    # conflict[x] = set of frozenset({a,b}) where {a,b,x} is bad
    conflict = [set() for _ in range(n + 1)]
    
    # For efficiency, group by LCM value
    # For each LCM value L, find all pairs (a,b) with lcm(a,b)=L
    # Then for each such pair, any c with lcm(a,c)=L and lcm(b,c)=L forms a bad triple
    from collections import defaultdict
    lcm_to_pairs = defaultdict(list)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm_table[a][b]
            lcm_to_pairs[L].append((a, b))
    
    # For each LCM value, find all elements that have LCM L with at least 2 elements in the pair set
    for L, pairs in lcm_to_pairs.items():
        if len(pairs) < 2:
            continue
        # For each element x, count how many pairs (a,b) in this L group have lcm(x,a)=L and lcm(x,b)=L
        # Actually, simpler: for each pair (a,b) in this L group, find all c such that
        # lcm(a,c)=L and lcm(b,c)=L, then {a,b,c} is a bad triple
        # But this is still O(pairs * n) per L group
        
        # Alternative: for each c, check if c has lcm L with at least 2 elements from the pair set
        # Precompute: for each c, which elements in [1,n] have lcm(c, elem) = L
        # This is expensive to do naively
        
        # Simpler approach: just enumerate triples directly but with early termination
        pass
    
    # Direct triple enumeration with optimization
    for a in range(1, n + 1):
        if time.time() > deadline - 20:
            break
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict[a].add(frozenset((b, c)))
                    conflict[b].add(frozenset((a, c)))
                    conflict[c].add(frozenset((a, b)))
    
    def count_conflicts(x, selected_set):
        """Check if adding x to selected_set creates a bad triple."""
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                return 1
        return 0
    
    def count_conflicts_excluding(x, selected_set, excluded_y):
        """Check if adding x to (selected_set - {excluded_y}) creates a bad triple."""
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                if a != excluded_y and b != excluded_y:
                    return 1
        return 0
    
    best_result = set()
    
    # Define various orderings
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
        sorted(range(1, n + 1), key=lambda x: len(conflict[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict[x])),
        # New: order by "freedom" - prefer elements with fewer conflicts
        sorted(range(1, n + 1), key=lambda x: (len(conflict[x]), x)),
        # Powers of 2 first (they have unique prime factorizations)
        [2**k for k in range(0, 8) if 2**k <= n] + [x for x in range(1, n+1) if x not in {2**k for k in range(0, 8) if 2**k <= n}],
    ]
    
    # Run greedy for each deterministic order
    for order in orders:
        if time.time() > deadline - 20:
            break
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Randomized greedy restarts with more iterations
    import random
    rng = random.Random(42)
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Local search improvement
    current = best_result.copy()
    
    # Phase 1: Try to add any element that fits (multiple passes)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
    
    # Phase 2: 1-for-1 Swaps with better ordering
    swap_attempts = 0
    max_swap_attempts = 20000
    while time.time() < deadline - 10 and swap_attempts < max_swap_attempts:
        swap_found = False
        # Try removing elements with high conflict count first
        for y in sorted(list(current), key=lambda y: -len(conflict[y])):
            if swap_attempts >= max_swap_attempts or time.time() > deadline - 10:
                break
            for x in range(1, n + 1):
                if x in current:
                    continue
                swap_attempts += 1
                if swap_attempts > max_swap_attempts or time.time() > deadline - 10:
                    break
                if count_conflicts_excluding(x, current, y) == 0:
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found or swap_attempts >= max_swap_attempts:
                break
        if not swap_found:
            break
    
    # Phase 3: Try 1-for-2 swaps (remove 1, add 2)
    if time.time() < deadline - 10:
        for x1 in range(1, n + 1):
            if x1 in current or time.time() > deadline - 10:
                continue
            for x2 in range(x1 + 1, n + 1):
                if x2 in current or time.time() > deadline - 10:
                    continue
                for y in list(current):
                    if count_conflicts_excluding(x1, current, y) == 0 and count_conflicts_excluding(x2, current - {y}, y) == 0:
                        current.remove(y)
                        current.add(x1)
                        current.add(x2)
                        break
    
    # Phase 4: Try 2-for-2 swaps (remove 2, add 2)
    if time.time() < deadline - 8:
        for y1 in range(1, n + 1):
            if y1 not in current or time.time() > deadline - 8:
                continue
            for y2 in range(y1 + 1, n + 1):
                if y2 not in current or time.time() > deadline - 8:
                    continue
                for x1 in range(1, n + 1):
                    if x1 in current or time.time() > deadline - 8:
                        continue
                    for x2 in range(x1 + 1, n + 1):
                        if x2 in current or time.time() > deadline - 8:
                            continue
                        temp = current - {y1, y2}
                        if count_conflicts_excluding(x1, current, y1) == 0 and count_conflicts_excluding(x1, current, y2) == 0:
                            if count_conflicts_excluding(x2, current, y1) == 0 and count_conflicts_excluding(x2, current, y2) == 0:
                                # Check both together
                                if count_conflicts(x1, temp) == 0 and count_conflicts(x2, temp | {x1}) == 0:
                                    current = temp | {x1, x2}
                                    break
                    if time.time() > deadline - 8:
                        break
                if time.time() > deadline - 8:
                    break
            if time.time() > deadline - 8:
                break
    
    # Phase 5: Final add pass
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
    
    # Phase 6: More aggressive 1-for-1 swaps after additions
    if time.time() < deadline - 3:
        swap_found = True
        while swap_found and time.time() < deadline - 3:
            swap_found = False
            for y in list(current):
                if time.time() > deadline - 3:
                    break
                for x in range(1, n + 1):
                    if x in current:
                        continue
                    if count_conflicts_excluding(x, current, y) == 0:
                        current.remove(y)
                        current.add(x)
                        swap_found = True
                        break
                if swap_found:
                    break
    
    return list(current) if current else list(best_result)
