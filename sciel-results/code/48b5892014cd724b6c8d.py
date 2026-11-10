"""Combined program: precomputed triples + fast greedy + local search + swaps."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = b*c / (b+c), so (b+c) | (b*c).
    # Since 1/a = 1/b + 1/c > 1/c, we have a < c. Similarly a < b.
    # So a is always the smallest of the three distinct elements.
    
    triples = []  # list of (a, b, c) with a < b < c
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))
    
    # For fast checking, build: for each a, the set of (b, c) pairs that form a triple with a.
    triple_map = {}
    for a, b, c in triples:
        if a not in triple_map:
            triple_map[a] = set()
        triple_map[a].add((b, c))
    
    # Also build reverse maps: for each b, set of (a, c) such that (a,b,c) is a triple
    # and for each c, set of (a, b) such that (a,b,c) is a triple
    # This allows O(1)-ish checks when adding an element.
    # Actually, for the has_triple check on a set S:
    # For each a in S, check if any (b,c) in triple_map[a] has both b,c in S.
    # This is fast enough if the number of triples per a is small.
    
    def has_triple(S):
        """Check if set S contains any forbidden triple."""
        for a in S:
            if a in triple_map:
                for (b, c) in triple_map[a]:
                    if b in S and c in S:
                        return True
        return False
    
    # Greedy construction with a given order
    def greedy_order(order):
        S = set()
        for x in order:
            trial = S | {x}
            if not has_triple(trial):
                S = trial
        return S
    
    # Phase 1: Multiple greedy constructions with different orderings
    best = set(range(91, N + 1))  # baseline: 90 elements
    best_size = len(best)
    
    # Descending order
    cand = greedy_order(list(range(N, 0, -1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Ascending order
    cand = greedy_order(list(range(1, N + 1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Random orderings
    for _ in range(100):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
    
    # Phase 2: Local search - try to add elements
    for _ in range(500):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            trial = best | {x}
            if not has_triple(trial):
                best = trial
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Phase 3: Swap search - remove one element, add another
    for _ in range(20000):
        if time.time() >= deadline:
            break
        if not best:
            break
        x = rng.randint(1, N)
        if x in best:
            continue
        y = rng.choice(list(best))
        trial = (best - {y}) | {x}
        if not has_triple(trial):
            best = trial
    
    # Phase 4: More local search after swaps
    for _ in range(500):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            trial = best | {x}
            if not has_triple(trial):
                best = trial
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Phase 5: Random restarts with diverse seeds
    for _ in range(5000):
        if time.time() >= deadline:
            break
        size = rng.randint(60, 120)
        s = set(rng.sample(range(1, N + 1), size))
        if has_triple(s):
            continue
        remaining = [x for x in range(1, N + 1) if x not in s]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() >= deadline:
                break
            trial = s | {x}
            if not has_triple(trial):
                s = trial
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Final verification
    if has_triple(best):
        best = set(range(91, N + 1))
    
    return list(best)
# EVOLVE-BLOCK-END
