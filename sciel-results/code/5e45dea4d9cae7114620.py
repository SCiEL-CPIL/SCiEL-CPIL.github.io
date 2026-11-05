import time
import random
from math import gcd
from itertools import combinations

def construct():
    # Strict deadline to avoid timeout
    deadline = time.time() + 55
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def is_compatible(x, selected_list):
        """Check if x can be added to selected_list without creating a bad triple."""
        n = len(selected_list)
        for i in range(n):
            for j in range(i + 1, n):
                if is_bad_triple(selected_list[i], selected_list[j], x):
                    return False
        return True

    def is_compatible_with_exclusion(x, selected_list, excluded_idx):
        """Check if x is compatible with selected_list minus element at excluded_idx."""
        n = len(selected_list)
        for i in range(n):
            if i == excluded_idx:
                continue
            for j in range(i + 1, n):
                if j == excluded_idx:
                    continue
                if is_bad_triple(selected_list[i], selected_list[j], x):
                    return False
        return True

    # Precompute conflict info: for each pair (a,b), which c values conflict
    # conflict[a][b] = set of c such that (a,b,c) is a bad triple
    conflict = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if is_bad_triple(a, b, c):
                    conflict[a][b].add(c)
                    conflict[b][a].add(c)

    def is_compatible_fast(x, selected_set):
        """Fast compatibility check using precomputed conflicts."""
        # For x to be compatible, for every pair (a,b) in selected, x must not be in conflict[a][b]
        # Equivalently: for every a in selected, x must not conflict with any other b in selected via a
        # Better: for every pair (a,b) in selected, check if x in conflict[a][b]
        sel_list = list(selected_set)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                if x in conflict[a][b]:
                    return False
        return True

    def is_compatible_fast_excl(x, selected_set, excl_elem):
        """Fast compatibility check excluding one element."""
        sel_list = [s for s in selected_set if s != excl_elem]
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                if x in conflict[a][b]:
                    return False
        return True

    # Greedy construction with multiple orderings
    def greedy_build(order):
        sel = set()
        for x in order:
            if is_compatible_fast(x, sel):
                sel.add(x)
        return sel

    # Calculate degree: number of bad triples each number is part of
    degrees = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad_triple(a, b, c):
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1

    # Try multiple greedy orderings
    best_selected = set()
    best_size = 0

    # Ordering 1: by degree ascending
    sorted_by_degree = sorted(numbers, key=lambda x: degrees[x])
    sel1 = greedy_build(sorted_by_degree)
    if len(sel1) > best_size:
        best_size = len(sel1)
        best_selected = sel1

    # Ordering 2: random shuffle
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        sel = greedy_build(shuffled)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel

    selected = best_selected

    # Local search
    last_improvement = time.time()
    improved = True

    while time.time() < deadline - 3 and improved:
        improved = False
        current_size = len(selected)

        # Phase A: Try to add any compatible number
        for x in numbers:
            if x not in selected:
                if is_compatible_fast(x, selected):
                    selected.add(x)
                    improved = True
                    last_improvement = time.time()
                    break

        if improved:
            continue

        # Phase B: Try 1-for-1 swaps
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if time.time() > deadline - 3:
                break
            y = sel_list[i]
            temp_sel = selected - {y}
            for x in numbers:
                if x == y or x in selected:
                    continue
                if is_compatible_fast_excl(x, selected, y):
                    selected = temp_sel | {x}
                    improved = True
                    last_improvement = time.time()
                    break
            if improved:
                break

        if improved:
            continue

        # Phase C: Try 2-for-2 swaps (remove two, add two)
        # This helps escape local optima
        sel_list = list(selected)
        n_sel = len(sel_list)
        found_2swap = False
        # Limit pairs to try
        pair_count = 0
        max_pairs = min(100, n_sel * (n_sel - 1) // 2)
        for i in range(n_sel):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, n_sel):
                if pair_count >= max_pairs:
                    break
                pair_count += 1
                y1 = sel_list[i]
                y2 = sel_list[j]
                temp_sel = selected - {y1, y2}
                
                # Find two elements that can be added
                candidates = [x for x in numbers if x not in selected and x != y1 and x != y2]
                # Sort candidates by some heuristic (e.g., degree)
                candidates.sort(key=lambda x: degrees[x])
                
                found_two = False
                for x1 in candidates[:50]:  # Limit candidates
                    if not is_compatible_fast_excl(x1, selected, y1) and not is_compatible_fast_excl(x1, selected, y2):
                        # x1 is compatible with selected minus y1 and selected minus y2
                        # Now find x2
                        for x2 in candidates:
                            if x2 <= x1:
                                continue
                            if not is_compatible_fast_excl(x2, selected, y1) and not is_compatible_fast_excl(x2, selected, y2):
                                # Check x1 and x2 together with temp_sel
                                if is_compatible_fast(x1, temp_sel) and is_compatible_fast(x2, temp_sel | {x1}):
                                    selected = temp_sel | {x1, x2}
                                    improved = True
                                    last_improvement = time.time()
                                    found_two = True
                                    found_2swap = True
                                    break
                        if found_two:
                            break
                if found_2swap:
                    break
            if found_2swap:
                break

        if improved:
            continue

        # Stagnation detection
        if not improved and time.time() - last_improvement > 5:
            # Try a random restart from a slightly perturbed version
            if len(selected) > 5:
                # Remove 2 random elements and try to rebuild
                perturb = set(selected)
                to_remove = random.sample(list(perturb), min(2, len(perturb)))
                for r in to_remove:
                    perturb.discard(r)
                
                # Try to add back
                for x in numbers:
                    if x not in perturb and is_compatible_fast(x, perturb):
                        perturb.add(x)
                
                if len(perturb) > len(selected):
                    selected = perturb
                    improved = True
                    last_improvement = time.time()
                elif time.time() - last_improvement > 15:
                    break

    return list(selected)
