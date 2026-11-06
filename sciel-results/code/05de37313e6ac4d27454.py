"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _has_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # Precompute LCM table for all pairs in [1, 128]
    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # A triple (a,b,c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Precompute for each pair (a,b), the set of c that would form a bad triple
    # i.e., c such that lcm(a,c) == lcm(a,b) and lcm(b,c) == lcm(a,b)
    # Let L = lcm(a,b). Then c must divide L, and lcm(a,c) = L and lcm(b,c) = L.
    
    # For a given pair (a,b) with L = lcm(a,b), find all c != a,b such that
    # lcm(a,c) = L and lcm(b,c) = L
    # This means c | L, and c must have "enough" of each prime power in L
    # that's not fully in a (for lcm(a,c)=L) and not fully in b (for lcm(b,c)=L)
    
    # Simpler: just check directly. For efficiency, precompute bad_triples.
    # For each triple (a,b,c) with a<b<c, check if bad.
    
    # Build adjacency: for each element x, which other elements y have a "conflict potential"
    # Actually, let's use a different approach: local search on the set.
    
    # Start with greedy construction
    selected = []
    for x in range(1, n + 1):
        ok = True
        for a, b in combinations(selected, 2):
            if _lcm(a, b) == _lcm(a, x) == _lcm(b, x):
                ok = False
                break
        if ok:
            selected.append(x)
    
    best_set = list(selected)
    best_size = len(selected)
    
    # Local search: try to add elements, swap elements
    in_set = set(best_set)
    
    def is_valid_addition(x, s):
        for a, b in combinations(s, 2):
            if _lcm(a, b) == _lcm(a, x) == _lcm(b, x):
                return False
        return True
    
    def is_valid_set(s):
        for a, b, c in combinations(s, 3):
            if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                return False
        return True
    
    # Try to add any element not in set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n + 1):
            if x not in in_set:
                if is_valid_addition(x, best_set):
                    best_set.append(x)
                    in_set.add(x)
                    best_size += 1
                    improved = True
    
    # Try swaps: remove one, add another
    if time.time() < deadline:
        for _ in range(50):  # limited swaps
            if time.time() >= deadline:
                break
            improved = False
            for i, x in enumerate(list(best_set)):
                if time.time() >= deadline:
                    break
                # Try removing x and adding some y
                temp_set = best_set[:i] + best_set[i+1:]
                for y in range(1, n + 1):
                    if y not in in_set or y == x:
                        if y in in_set and y != x:
                            continue
                        if is_valid_addition(y, temp_set):
                            if len(temp_set) + 1 > best_size:
                                best_set = temp_set + [y]
                                in_set = set(best_set)
                                best_size += 1
                                improved = True
                                break
            if not improved:
                break
    
    # More aggressive local search: try all single removals and re-additions
    if time.time() < deadline:
        for _ in range(100):
            if time.time() >= deadline:
                break
            improved = False
            # Try removing each element and see if we can add two back
            for i in range(len(best_set)):
                if time.time() >= deadline:
                    break
                x = best_set[i]
                temp_set = best_set[:i] + best_set[i+1:]
                temp_in = set(temp_set)
                # Try to add back x and possibly another
                for y in range(1, n + 1):
                    if y in temp_in:
                        continue
                    if is_valid_addition(y, temp_set):
                        new_set = temp_set + [y]
                        if len(new_set) > best_size:
                            best_set = new_set
                            in_set = set(best_set)
                            best_size = len(best_set)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    # Final verification
    assert len(best_set) > 0
    for a, b, c in combinations(best_set, 3):
        assert not (_lcm(a, b) == _lcm(a, c) == _lcm(b, c)), f"Bad triple: {a},{b},{c}"
    
    return best_set
# EVOLVE-BLOCK-END
