import random
import time
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    deadline = time.time() + 55.0

    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute for each pair (a,b) the gcd
    # A triple (a,b,c) is bad iff gcd(a,b)==gcd(a,c)==gcd(b,c)
    # For efficiency, precompute conflict sets:
    # conflicts[y] = set of pairs (a,b) with a<b, a!=y, b!=y, such that adding y to {a,b} makes bad triple
    # i.e., gcd(a,b)==gcd(a,y)==gcd(b,y)
    conflicts = [set() for _ in range(N + 1)]
    for y in range(1, N + 1):
        for a in range(1, y):
            for b in range(a + 1, N + 1):
                if b == y:
                    continue
                if gcd_table[a][b] == gcd_table[a][y] == gcd_table[b][y]:
                    conflicts[y].add((a, b))
        # Also need a > y cases, but we store with a<b convention
        # For a > y, the pair is (y, a) if y < a, but we need all pairs
        # Let me redo: for each y, for each pair (a,b) with a<b, a!=y, b!=y
        conflicts[y] = set()
        for a in range(1, N + 1):
            if a == y:
                continue
            for b in range(a + 1, N + 1):
                if b == y:
                    continue
                if gcd_table[a][b] == gcd_table[a][y] == gcd_table[b][y]:
                    conflicts[y].add((a, b))

    def is_valid_add(S_set, S_list, y):
        """Check if adding y to S creates a bad triple."""
        for a in S_list:
            for b in S_list:
                if a == b:
                    continue
                if a < b:
                    if (a, b) in conflicts[y]:
                        return False
                else:
                    if (b, a) in conflicts[y]:
                        return False
        return True

    # Better: use a bitset-like approach with numpy for speed
    # conflicts[y] as a set of frozenset pairs, but we need O(|S|^2) check
    # Alternative: for each y, precompute a boolean matrix conflict_matrix[y][a][b]
    # That's 128 * 129 * 129 booleans = ~2M, fine.
    import numpy as np
    conflict_matrix = np.zeros((N + 1, N + 1, N + 1), dtype=bool)
    for y in range(1, N + 1):
        for a in range(1, N + 1):
            if a == y:
                continue
            for b in range(1, N + 1):
                if b == y or b <= a:
                    continue
                if gcd_table[a][b] == gcd_table[a][y] == gcd_table[b][y]:
                    conflict_matrix[y, a, b] = True
                    conflict_matrix[y, b, a] = True

    def is_valid_add_fast(S_bool, y):
        """S_bool is a numpy bool array of shape (N+1)."""
        # For each a in S, b in S, a != b, check conflict_matrix[y, a, b]
        # This is O(N^2) but with numpy it's fast
        # Actually, we can do: for each a where S_bool[a], check if any b where S_bool[b] and conflict_matrix[y,a,b]
        # Sum over a: sum of conflict_matrix[y, a, :] & S_bool, then check if any
        # But we need to exclude a==b, but conflict_matrix[y,a,a] is False anyway
        for a in range(1, N + 1):
            if S_bool[a]:
                if np.any(conflict_matrix[y, a, :] & S_bool):
                    return False
        return True

    # Greedy initialization with multiple orderings
    best_S = []
    best_size = 0

    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, N + 1), key=lambda x: (x % 2, x)),
        list(range(1, N + 1, 2)),  # odd first
    ]

    for ordering in orderings:
        S = []
        S_bool = np.zeros(N + 1, dtype=bool)
        for x in ordering:
            if is_valid_add_fast(S_bool, x):
                S.append(x)
                S_bool[x] = True
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    # Local search: 1-add, 1-swap, 2-swap
    random.seed(42)
    S = list(best_S)
    S_bool = np.zeros(N + 1, dtype=bool)
    for x in S:
        S_bool[x] = True

    while time.time() < deadline:
        improved = False

        # Try to add
        candidates = [x for x in range(1, N + 1) if not S_bool[x]]
        random.shuffle(candidates)
        for y in candidates:
            if is_valid_add_fast(S_bool, y):
                S.append(y)
                S_bool[y] = True
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                break

        if not improved:
            # Try swap: remove x, add y
            swapped = False
            remove_candidates = list(S)
            random.shuffle(remove_candidates)
            for x in remove_candidates:
                S_bool[x] = False
                S.remove(x)
                add_candidates = [y for y in range(1, N + 1) if not S_bool[y]]
                random.shuffle(add_candidates)
                for y in add_candidates:
                    if is_valid_add_fast(S_bool, y):
                        S.append(y)
                        S_bool[y] = True
                        swapped = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = list(S)
                        break
                if not swapped:
                    # Revert
                    S.append(x)
                    S_bool[x] = True
                else:
                    break

        if not improved and not swapped:
            # Kick: remove 2 random, refill greedily
            if len(S) >= 2:
                to_remove = random.sample(S, 2)
                for x in to_remove:
                    S.remove(x)
                    S_bool[x] = False
                # Greedy refill
                for y in range(1, N + 1):
                    if not S_bool[y] and is_valid_add_fast(S_bool, y):
                        S.append(y)
                        S_bool[y] = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)

    return best_S
