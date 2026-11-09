"""Max independent set in reciprocal-Schur hypergraph via structure-exploiting greedy + local search."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = 85.0
N = 180


def _precompute_triples():
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:
                continue
            b = a + d
            c = a + e
            if b > N or c > N:
                continue
            if a == b or a == c or b == c:
                continue
            if b > c:
                b, c = c, b
            triples.append((a, b, c))
    return triples


def _build_partners(triples):
    partners = defaultdict(list)
    for (a, b, c) in triples:
        partners[a].append((b, c))
        partners[b].append((a, c))
        partners[c].append((a, b))
    return partners


def _can_add(x, set_bits, partners):
    for (y, z) in partners[x]:
        if set_bits[y] and set_bits[z]:
            return False
    return True


def _greedy_remove(triples, partners, random_seed=None):
    """Greedy: start with all, remove to eliminate violations."""
    if random_seed is not None:
        random.seed(random_seed)
    
    set_bits = [True] * (N + 1)
    
    degree = [0] * (N + 1)
    for (a, b, c) in triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    changed = True
    while changed:
        changed = False
        violated = []
        for (a, b, c) in triples:
            if set_bits[a] and set_bits[b] and set_bits[c]:
                violated.append((a, b, c))
        if not violated:
            break
        
        vc = [0] * (N + 1)
        for (a, b, c) in violated:
            vc[a] += 1
            vc[b] += 1
            vc[c] += 1
        
        # Find candidates with violations
        candidates = [x for x in range(1, N + 1) if vc[x] > 0 and set_bits[x]]
        if not candidates:
            break
        
        # Score: higher vc is worse, lower degree is safer to remove
        # Add small random perturbation for diversity
        def score(x):
            perturbation = random.random() * 0.1 if random_seed is not None else 0
            return (vc[x], -degree[x], perturbation)
        
        worst = max(candidates, key=score)
        set_bits[worst] = False
        changed = True
    
    return {x for x in range(1, N + 1) if set_bits[x]}


def _local_search(initial_set, triples, partners, time_limit):
    """Local search with add-back and swap moves."""
    set_bits = [False] * (N + 1)
    current = set(initial_set)
    for x in current:
        set_bits[x] = True
    
    start_time = time.time()
    best = set(current)
    best_size = len(current)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Add back all possible
        for x in range(1, N + 1):
            if time.time() - start_time > time_limit:
                break
            if not set_bits[x] and _can_add(x, set_bits, partners):
                set_bits[x] = True
                current.add(x)
                improved = True
        
        # Try 1-swap moves
        if time.time() - start_time > time_limit:
            break
        for x in list(current):
            if time.time() - start_time > time_limit:
                break
            set_bits[x] = False
            current.discard(x)
            
            added = 0
            for y in range(1, N + 1):
                if time.time() - start_time > time_limit:
                    break
                if not set_bits[y] and _can_add(y, set_bits, partners):
                    set_bits[y] = True
                    current.add(y)
                    added += 1
            
            if added > 0:
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
            else:
                set_bits[x] = True
                current.add(x)
    
    return best


def construct():
    start_time = time.time()
    
    triples = _precompute_triples()
    partners = _build_partners(triples)
    
    # Multiple greedy restarts with different random seeds
    best_set = set()
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() - start_time > _DEADLINE - 15:
            break
        gset = _greedy_remove(triples, partners, random_seed=i)
        if len(gset) > len(best_set):
            best_set = set(gset)
    
    # Local search on best greedy result
    remaining = _DEADLINE - (time.time() - start_time) - 5
    if remaining > 1:
        ls_set = _local_search(best_set, triples, partners, remaining)
        if len(ls_set) > len(best_set):
            best_set = ls_set
    
    # Final validation
    set_bits = [False] * (N + 1)
    for x in best_set:
        set_bits[x] = True
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            set_bits[c] = False
    best_set = {x for x in range(1, N + 1) if set_bits[x]}
    
    return sorted(best_set)
