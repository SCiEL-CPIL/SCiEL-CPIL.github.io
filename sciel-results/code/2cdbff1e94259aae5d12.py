import time
import random

_N = 256
_DEADLINE = 85.0

def _construct_neighbors():
    """
    Precompute neighbors for each node 1..256.
    i and j are neighbors if (i*j) % (i+j) == 0.
    """
    neighbors = [[] for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors

def _can_add(x, S_set, neighbors):
    """Check if x can be added to S_set."""
    for y in neighbors[x]:
        if y in S_set:
            return False
    return True

def _greedy_init(neighbors, rng):
    """
    Greedy initialization.
    Sort nodes by degree (ascending) to pick 'safe' ones first?
    Or random order?
    Let's try sorting by degree ascending.
    """
    degrees = [len(neighbors[i]) for i in range(_N + 1)]
    order = list(range(1, _N + 1))
    # Sort by degree, then value
    order.sort(key=lambda x: (degrees[x], x))
    
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _greedy_refill(S_set, complement, neighbors, order=None):
    """
    Greedily add nodes from complement to S_set.
    If order is given, use that order; otherwise use sorted(complement).
    Returns (new_S_set, new_complement).
    """
    if order is None:
        order = sorted(complement)
    new_S_set = set(S_set)
    new_complement = set(complement)
    for x in order:
        if x not in new_complement:
            continue
        if _can_add(x, new_S_set, neighbors):
            new_S_set.add(x)
            new_complement.discard(x)
    return new_S_set, new_complement

def _local_search(S, neighbors, rng, time_limit):
    """
    Local search to improve S.
    1. Try to add any available node.
    2. Try 1-for-k swaps (remove u, greedily refill).
    3. Try 2-for-k swaps (remove u,v, greedily refill).
    """
    S = list(S)
    S_set = set(S)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Add any node that fits
        for x in list(complement):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        
        if time.time() >= time_limit:
            break
            
        # Phase 2: Try swaps (only if no additions in Phase 1)
        if not improved:
            best_size = len(S)
            
            # --- 1-for-k swaps ---
            for u in list(S):
                if time.time() >= time_limit:
                    break
                
                S_set.remove(u)
                complement.add(u)
                
                new_S_set, new_complement = _greedy_refill(S_set, complement, neighbors)
                
                if len(new_S_set) > best_size:
                    S = list(new_S_set)
                    S_set = new_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    best_size = len(new_S_set)
                    improved = True
                    break
            
            if time.time() >= time_limit:
                break
            
            # --- 2-for-k swaps (if still stuck) ---
            if not improved:
                S_list = list(S)
                n = len(S_list)
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    for j in range(i + 1, n):
                        if time.time() >= time_limit:
                            break
                        
                        u, v = S_list[i], S_list[j]
                        S_set.remove(u)
                        S_set.remove(v)
                        complement.add(u)
                        complement.add(v)
                        
                        new_S_set, new_complement = _greedy_refill(S_set, complement, neighbors)
                        
                        if len(new_S_set) > best_size:
                            S = list(new_S_set)
                            S_set = new_S_set
                            complement = set(range(1, _N + 1)) - S_set
                            best_size = len(new_S_set)
                            improved = True
                            break
                    if improved:
                        break

    return S

def _validate(S):
    """Check that all distinct pairs in S satisfy (a*b) % (a+b) != 0."""
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = S_list[i], S_list[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    neighbors = _construct_neighbors()
    
    best_S = []
    best_size = 0
    
    # Try multiple random restarts of greedy init with varied heuristics
    for i in range(60):
        if time.time() - t0 > _DEADLINE - 8:
            break
        
        if i % 3 == 0:
            # Degree ascending
            S, S_set = _greedy_init(neighbors, rng)
        elif i % 3 == 1:
            # Random order
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S = []
            S_set = set()
            for x in order:
                if _can_add(x, S_set, neighbors):
                    S.append(x)
                    S_set.add(x)
        else:
            # Most-conflicting-first: pick node with most neighbors in current S
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S = []
            S_set = set()
            for x in order:
                if _can_add(x, S_set, neighbors):
                    S.append(x)
                    S_set.add(x)
            # Refine: greedily add remaining nodes that fit
            remaining = [x for x in range(1, _N + 1) if x not in S_set]
            for x in remaining:
                if _can_add(x, S_set, neighbors):
                    S.append(x)
                    S_set.add(x)
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 3:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 3)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Final validation safety net
    if not _validate(best_S):
        # Fallback to all odds (guaranteed valid)
        best_S = list(range(1, _N + 1, 2))
    
    return best_S
