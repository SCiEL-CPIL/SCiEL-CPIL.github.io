import time
from itertools import combinations
from math import gcd, lcm
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # safety margin before 90s kill

    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            lcm_table[a][b] = lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    bad_triples.append((a, b, c))

    # Build conflict structure: for each pair, count bad triples containing them
    # And for each element, list of bad triples it's in
    elem_triples = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    all_elements = set(range(1, n + 1))

    def is_valid_set(S):
        """Check if set S contains no bad triple using element-triple index."""
        S_set = S if isinstance(S, set) else set(S)
        for idx, (a, b, c) in enumerate(bad_triples):
            if a in S_set and b in S_set and c in S_set:
                return False
        return True

    def can_add(S, elem):
        """Check if adding elem to S creates any bad triple."""
        for idx in elem_triples[elem]:
            a, b, c = bad_triples[idx]
            others = [x for x in (a, b, c) if x != elem]
            if all(x in S for x in others):
                return False
        return True

    def greedy_fill(start_set, order):
        """Greedily add elements in given order."""
        S = set(start_set)
        for elem in order:
            if elem not in S and can_add(S, elem):
                S.add(elem)
        return S

    def local_search_add(S):
        """Try to add back any missing element."""
        improved = True
        while improved:
            improved = False
            for elem in range(1, n + 1):
                if elem not in S and can_add(S, elem):
                    S.add(elem)
                    improved = True
        return S

    def local_search_swap(S, max_iters=500):
        """Try 1-for-1 swaps to enable further additions."""
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            S_list = sorted(S)
            R_list = sorted(set(range(1, n + 1)) - S)
            found = False
            for a in S_list:
                for b in R_list:
                    if a == b:
                        continue
                    trial = (S - {a}) | {b}
                    # Check if this swap allows adding more elements
                    trial2 = local_search_add(trial)
                    if len(trial2) > len(S):
                        S = trial2
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return S

    # Strategy: Start from odd numbers (structurally safe), then greedily add evens
    odds = set(range(1, n + 1, 2))
    
    best_S = set(odds)
    best_size = len(best_S)

    # Phase 1: Greedy fill from odd base with different orderings
    evens = list(range(2, n + 1, 2))
    
    # Try multiple orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > DEADLINE:
            break
        order = evens[:]
        random.shuffle(order)
        S = greedy_fill(set(odds), order)
        S = local_search_add(S)
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)

    # Phase 2: Local search swaps on best found
    S = set(best_S)
    S = local_search_swap(S, max_iters=2000)
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Phase 3: More random restarts with mixed strategies
    for restart in range(300):
        if time.time() > DEADLINE:
            break
        
        # Random subset of odds as base
        base_size = random.randint(30, 64)
        base = random.sample(list(odds), min(base_size, len(odds)))
        base_set = set(base)
        
        # Greedy fill
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_fill(base_set, order)
        S = local_search_add(S)
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Quick swap improvement
        if time.time() < DEADLINE - 5:
            S2 = local_search_swap(S, max_iters=200)
            if len(S2) > best_size:
                best_S = set(S2)
                best_size = len(S2)

    # Phase 4: Final polish - try adding any remaining elements
    S = set(best_S)
    S = local_search_add(S)
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Final swap pass
    if time.time() < DEADLINE - 2:
        S = local_search_swap(best_S, max_iters=500)
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)

    return sorted(best_S) if best_S else [1]
