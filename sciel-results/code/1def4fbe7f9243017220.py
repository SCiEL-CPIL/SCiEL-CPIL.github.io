import time
import math
import random

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    From 1/a = 1/b + 1/c, we get a < b and a < c (since 1/a > 1/b and 1/a > 1/c).
    WLOG assume b <= c. The equation gives c = ab/(b-a).
    For each pair (a, b) with b > a, if (b-a) divides ab, then c = ab/(b-a) is an integer.
    We need c in [1, N] and c distinct from a and b.
    """
    # pair_to_k: dict mapping (a, b) -> list of c values that form a triple
    pair_to_c = {}
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
                    # Store in pair_to_c for quick lookup
                    if (a, b) not in pair_to_c:
                        pair_to_c[(a, b)] = []
                    pair_to_c[(a, b)].append(c)
                    if (a, c) not in pair_to_c:
                        pair_to_c[(a, c)] = []
                    pair_to_c[(a, c)].append(b)
                    if (b, c) not in pair_to_c:
                        pair_to_c[(b, c)] = []
                    pair_to_c[(b, c)].append(a)
    return triples, pair_to_c

def _check_valid(S, triples):
    """Check if set S has no three distinct elements forming a reciprocal triple."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def _greedy_construct(N, pair_to_c, order='descending', random_seed=None):
    """
    Greedy construction: try to add elements in a given order, skipping any that
    would create a forbidden triple with already-added elements.
    """
    S = set()
    conflict_count = {}  # element -> number of conflicts
    
    if order == 'descending':
        candidates = list(range(N, 0, -1))
    elif order == 'ascending':
        candidates = list(range(1, N + 1))
    elif order == 'random':
        if random_seed is not None:
            rng = random.Random(random_seed)
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
    else:
        candidates = list(range(N, 0, -1))
    
    for x in candidates:
        # Check if adding x creates any forbidden triple
        can_add = True
        for y in S:
            if x < y:
                pair = (x, y)
            else:
                pair = (y, x)
            if pair in pair_to_c:
                for z in pair_to_c[pair]:
                    if z != x and z != y and z in S:
                        can_add = False
                        break
            if not can_add:
                break
        if can_add:
            S.add(x)
    
    return list(S)

def _local_search(S, pair_to_c, triples, max_iterations=500, deadline=None):
    """
    Local search: try to add/remove/swap elements to improve the set size.
    """
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    
    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try adding a random element not in S
        not_in_S = [x for x in range(1, 181) if x not in S]
        if not_in_S:
            random.shuffle(not_in_S)
            improved = False
            for x in not_in_S[:20]:  # Try a few random additions
                new_S = S | {x}
                if _check_valid(new_S, triples):
                    S = new_S
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    improved = True
                    break
            if improved:
                continue
        
        # Try swapping: remove one element, add another
        if len(S) > 0:
            S_list = list(S)
            random.shuffle(S_list)
            for rem in S_list[:10]:
                S_without = S - {rem}
                for add in not_in_S[:10]:
                    if add == rem:
                        continue
                    new_S = S_without | {add}
                    if _check_valid(new_S, triples):
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        break
    
    return list(best_S)

def construct():
    """Return a nonempty subset of [1,180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a=1/b+1/c. Maximize cardinality."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    N = 180
    triples, pair_to_c = _precompute_triples(N)
    
    best_set = set(range(91, 181))  # Initial seed
    best_size = len(best_set)
    
    # Try multiple greedy constructions with different orders
    for i in range(50):
        if time.time() > deadline:
            break
        
        if i < 5:
            # Deterministic orders
            if i == 0:
                S = _greedy_construct(N, pair_to_c, order='descending')
            elif i == 1:
                S = _greedy_construct(N, pair_to_c, order='ascending')
            else:
                continue
        else:
            S = _greedy_construct(N, pair_to_c, order='random', random_seed=i)
        
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Local search from best set found
    if time.time() < deadline:
        improved = _local_search(list(best_set), pair_to_c, triples, 
                                  max_iterations=200, deadline=deadline)
        if len(improved) > best_size:
            best_set = set(improved)
            best_size = len(best_set)
    
    # If we still have time, do more iterations
    while time.time() < deadline - 5:
        S = _greedy_construct(N, pair_to_c, order='random', random_seed=random.randint(0, 1000000))
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
        # Quick local search
        if time.time() < deadline - 5:
            improved = _local_search(list(best_set), pair_to_c, triples, 
                                      max_iterations=50, deadline=deadline)
            if len(improved) > best_size:
                best_set = set(improved)
                best_size = len(improved)
    
    return list(best_set)

# EVOLVE-BLOCK-END
