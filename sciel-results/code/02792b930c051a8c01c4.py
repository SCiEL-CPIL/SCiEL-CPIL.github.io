import time
import random

_N = 256
_DEADLINE = 85.0

def _construct_neighbors():
    """
    Precompute neighbors for each node 1..256.
    i and j are neighbors if (i*j) % (i+j) == 0.
    """
    neighbors = [set() for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors

def _can_add(x, S_set, neighbors):
    """Check if x can be added to S_set."""
    # Using set intersection is faster than iterating over neighbors[x] and checking membership in S_set
    # if S_set is a set
    return not (neighbors[x] & S_set)

def _greedy_init(neighbors, rng, mode='degree'):
    """
    Greedy initialization.
    mode: 'degree', 'random', 'least_conflicting'
    """
    if mode == 'degree':
        degrees = [len(neighbors[i]) for i in range(_N + 1)]
        order = list(range(1, _N + 1))
        order.sort(key=lambda x: (degrees[x], x))
    elif mode == 'random':
        order = list(range(1, _N + 1))
        rng.shuffle(order)
    else:
        order = list(range(1, _N + 1))

    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _local_search(S, neighbors, rng, time_limit):
    """
    Local search to improve S.
    1. Try to add any available node.
    2. Try 1-for-k swaps (remove u, greedily refill).
    """
    S = list(S)
    S_set = set(S)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Add any node that fits
        # Iterate through complement. To make it more effective, we can sort by degree in S_set?
        # Let's just iterate.
        for x in list(complement):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        
        if time.time() >= time_limit:
            break
            
        # Phase 2: Try swaps
        # Only try swaps if we are stuck (no additions in Phase 1)
        if not improved:
            # Try 1-for-k
            best_S = S
            best_S_set = S_set
            best_size = len(S)
            
            # Shuffle S to randomize which u we try first?
            # Or just iterate.
            for u in list(S):
                if time.time() >= time_limit:
                    break
                
                # Remove u
                S_set.remove(u)
                complement.add(u)
                
                # Greedily refill from complement
                new_S_set = set(S_set)
                new_complement = set(complement)
                
                # Simple greedy refill
                # To improve refill quality, we can sort new_complement by 
                # number of neighbors in new_S_set.
                # This is O(|C| log |C|) but |C| is small.
                # Let's try sorting by conflicts with new_S_set.
                
                def sort_key(x):
                    return len(neighbors[x] & new_S_set)
                
                # Sorting every time is expensive. 
                # Let's just iterate.
                for x in list(new_complement):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        new_complement.discard(x)
                        
                if len(new_S_set) > best_size:
                    best_S = list(new_S_set)
                    best_S_set = new_S_set
                    best_size = len(new_S_set)
                    improved = True
                    S = best_S
                    S_set = best_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    break 

    return S

def _two_swap_search(S, neighbors, rng, time_limit):
    """
    Try 2-for-3 swaps: remove two elements from S, add up to three from remaining.
    """
    S = list(S)
    S_set = set(S)
    remaining = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                
                # Greedily add from remaining
                new_S_set = set(S_set)
                added = 0
                for x in list(remaining):
                    if added >= 3:
                        break
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        added += 1
                
                if added > 2:
                    # Net gain is positive
                    S = list(new_S_set)
                    S_set = new_S_set
                    remaining = set(range(1, _N + 1)) - S_set
                    improved = True
                    break
                else:
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    neighbors = _construct_neighbors()
    
    best_S = []
    best_size = 0
    
    # Try multiple random restarts of greedy init
    # Greedy with different orderings
    for i in range(100):
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        # Variation 1: Degree sorted
        if i % 3 == 0:
            S, S_set = _greedy_init(neighbors, rng, mode='degree')
        elif i % 3 == 1:
            S, S_set = _greedy_init(neighbors, rng, mode='random')
        else:
            # Least conflicting greedy (dynamic)
            S, S_set = [], set()
            remaining = set(range(1, _N + 1))
            while remaining:
                best_x = None
                best_c = None
                for x in remaining:
                    c = len(neighbors[x] & S_set)
                    if best_c is None or c < best_c or (c == best_c and x < best_x):
                        best_x = x
                        best_c = c
                    if c == 0:
                        break
                if best_c == 0:
                    S.append(best_x)
                    S_set.add(best_x)
                    remaining.discard(best_x)
                else:
                    break
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 5:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 5)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Two-swap search
    if time.time() - t0 < _DEADLINE - 2:
        S_final = _two_swap_search(best_S, neighbors, rng, _DEADLINE - 2)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Final local search polish
    if time.time() - t0 < _DEADLINE - 1:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 1)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
          
    return best_S
