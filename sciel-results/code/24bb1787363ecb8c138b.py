import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflict pairs
    conflict_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                if g_ab == gcd_table[a][c] == gcd_table[b][c]:
                    conflict_pairs[a].add(frozenset((b, c)))
                    conflict_pairs[b].add(frozenset((a, c)))
                    conflict_pairs[c].add(frozenset((a, b)))
    
    def can_add(x, selected_set):
        for pair in conflict_pairs[x]:
            y, z = tuple(pair)
            if y in selected_set and z in selected_set:
                return False
        return True
    
    best_set = set()
    orderings = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: len(conflict_pairs[x])),
        sorted(range(1, 129), key=lambda x: -len(conflict_pairs[x])),
        sorted(range(1, 129), key=lambda x: (len(conflict_pairs[x]), x)),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        selected = set()
        for x in ordering:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
        if len(selected) > len(best_set):
            best_set = selected.copy()
    
    if time.time() < deadline:
        current_set = set(best_set)
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            # Try adding
            for x in range(1, 129):
                if time.time() > deadline:
                    break
                if x not in current_set and can_add(x, current_set):
                    current_set.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-2 swap
            for rem in list(current_set):
                if time.time() > deadline:
                    break
                test_set = current_set - {rem}
                added = 0
                for x in range(1, 129):
                    if time.time() > deadline:
                        break
                    if x not in test_set and can_add(x, test_set):
                        test_set.add(x)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    current_set = test_set
                    improved = True
                    break
            if not improved:
                break
    
    return sorted(current_set)
