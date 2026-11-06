"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 180
    
    # Precompute all bad triples: distinct a,b,c in [1,180] with 1/a = 1/b + 1/c
    # This means a < b and a < c (since 1/a > 1/b and 1/a > 1/c)
    # c = ab/(b-a) where b > a, and we need c > a, c != b, c integer, c in [1,180]
    # Also by symmetry, we can have b > c instead of c > b, but the formula gives c = ab/(b-a)
    # when we solve for c given a and b with b > a.
    # Let's be systematic: for each a, and each b > a, compute c = ab/(b-a).
    # If c is a positive integer, c != a, c != b, c <= 180, then {a,b,c} is a bad triple.
    # Note: since the equation is symmetric in b and c, if we fix a and vary b > a,
    # we get all triples where a is the smallest. But what if a is not the smallest?
    # Actually, 1/a = 1/b + 1/c implies a < b and a < c. So a is always the smallest.
    # So we only need to consider a as the smallest element.
    
    bad_triples = set()
    for a in range(1, 181):
        for b in range(a+1, 181):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c <= 180 and c != b:
                    # Ensure distinct: a < b, c > a, c != b
                    bad_triples.add(frozenset((a, b, c)))
    
    # For efficient lookup: for each element x, store the list of bad triples containing x
    triples_by_elem = {i: [] for i in range(1, 181)}
    for t in bad_triples:
        for elem in t:
            triples_by_elem[elem].append(t)
    
    def is_valid(S):
        """Check if set S has no bad triple."""
        S_set = set(S)
        for t in bad_triples:
            if t.issubset(S_set):
                return False
        return True
    
    def count_conflicts(S, x):
        """Count how many bad triples would be formed if we add x to S."""
        S_set = set(S)
        count = 0
        for t in triples_by_elem[x]:
            others = t - {x}
            if others.issubset(S_set):
                count += 1
        return count
    
    # Strategy: Start with a good initial set, then use local search to improve.
    # Initial set: [91, 180] U {1, 2} as a starting point, then greedily add more.
    
    # First, try a greedy construction from scratch
    def greedy_construct(order):
        S = set()
        for x in order:
            if count_conflicts(S, x) == 0:
                S.add(x)
        return S
    
    # Try multiple random orderings and keep the best
    best_S = set(range(91, 181))  # fallback
    best_size = len(best_S)
    
    # Try adding small numbers greedily
    S = set(range(91, 181))
    for x in range(1, 91):
        if count_conflicts(S, x) == 0:
            S.add(x)
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
    
    # Try random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 181))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Local search: try to add/remove elements to improve
    # Phase 1: Try to add elements not in best_S
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = [x for x in range(1, 181) if x not in best_S]
        random.shuffle(candidates)
        for x in candidates:
            if count_conflicts(best_S, x) == 0:
                best_S.add(x)
                improved = True
                best_size = len(best_S)
    
    # Phase 2: Try swapping: remove one element, add another
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        S = set(best_S)
        # Try random swaps
        elems = list(S)
        random.shuffle(elems)
        for rem in elems[:20]:  # try removing up to 20 random elements
            S_test = set(S)
            S_test.discard(rem)
            # Try to add elements not in S_test
            added = False
            for add in range(1, 181):
                if add not in S_test and count_conflicts(S_test, add) == 0:
                    S_test.add(add)
                    added = True
            if len(S_test) > len(S):
                best_S = S_test
                best_size = len(S_test)
                break
    
    # Phase 3: More aggressive local search with multiple additions
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        S = set(best_S)
        # Try removing a small number of elements and re-adding greedily
        elems = list(S)
        random.shuffle(elems)
        for num_remove in [1, 2, 3]:
            if time.time() > deadline - 3:
                break
            S_test = set(S)
            for rem in elems[:num_remove]:
                S_test.discard(rem)
            # Re-add greedily
            candidates = [x for x in range(1, 181) if x not in S_test]
            random.shuffle(candidates)
            for x in candidates:
                if count_conflicts(S_test, x) == 0:
                    S_test.add(x)
            if len(S_test) > len(S):
                best_S = S_test
                best_size = len(S_test)
                break
    
    # Final verification
    if not is_valid(best_S):
        # Fallback to safe set
        best_S = set(range(91, 181))
    
    return list(best_S)

# EVOLVE-BLOCK-END
