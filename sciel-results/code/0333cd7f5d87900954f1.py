"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _is_valid_addition(triple, triple_set):
    """Check if adding triple to triple_set creates a K(2,2,2)."""
    a, b, c = triple
    # For each pair of other points (a', b'), check if all 8 transversals would be present
    # The 8 transversals for pairs {a,a'}, {b,b'}, {c,c'} are:
    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    # We already have (a,b,c). Need to check if other 7 exist.
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            for c_prime in range(16):
                if c_prime in (a, b, c, a_prime, b_prime):
                    continue
                # Check all 7 other transversals
                t1 = tuple(sorted((a, b, c_prime)))
                t2 = tuple(sorted((a, b_prime, c)))
                t3 = tuple(sorted((a, b_prime, c_prime)))
                t4 = tuple(sorted((a_prime, b, c)))
                t5 = tuple(sorted((a_prime, b, c_prime)))
                t6 = tuple(sorted((a_prime, b_prime, c)))
                t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                if (t1 in triple_set and t2 in triple_set and t3 in triple_set and
                    t4 in triple_set and t5 in triple_set and t6 in triple_set and t7 in triple_set):
                    return False
    return True

def _greedy_construct(deadline):
    """Greedy construction with randomization."""
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    random.shuffle(all_triples)
    
    triple_set = set()
    for triple in all_triples:
        if time.time() > deadline:
            break
        if _is_valid_addition(triple, triple_set):
            triple_set.add(triple)
    return list(triple_set)

def _local_search(triples, deadline, max_iters=50):
    """Local search: try to add more triples by removing conflicting ones."""
    triple_set = set(tuple(t) for t in triples)
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    
    best_size = len(triple_set)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        # Find a triple not in set that could be added
        added = False
        for t in all_triples:
            if t in triple_set:
                continue
            if _is_valid_addition(t, triple_set):
                triple_set.add(t)
                added = True
                break
        if not added:
            break
    
    return list(triple_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple random greedy constructions
    best_result = []
    num_attempts = 10
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        random.seed(i * 42 + 123)
        result = _greedy_construct(deadline)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search on best result
    if time.time() < deadline:
        best_result = _local_search(best_result, deadline, max_iters=30)
    
    return best_result
# EVOLVE-BLOCK-END
