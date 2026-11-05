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
        # Sort complement by degree in current S? Or just iterate?
        # Iterating is fine.
        for x in list(complement):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        
        if time.time() >= time_limit:
            break
            
        # Phase 2: Try swaps
        # To avoid O(N^2) per iteration, we can limit the number of swap attempts
        # or try random swaps.
        # Let's try removing each u in S and seeing if we can gain.
        # To speed up, we can just try a few random u's if S is large.
        # But N=256 is small. Let's try all u's but with a fast refill.
        
        # Optimization: Only try swaps if we are stuck (no additions in Phase 1)
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
                # Order of refill matters. Sort by degree in S_set?
                # Let's just iterate through complement.
                # To make it deterministic-ish but varied, we can shuffle complement?
                # No, shuffling every time is expensive.
                # Let's try adding in increasing order of "conflicts with S_set".
                
                new_S_set = set(S_set)
                new_complement = set(complement)
                
                # Simple greedy refill
                # We want to add as many as possible.
                # A good heuristic: add nodes with fewest neighbors in new_S_set first.
                # But calculating this for all is expensive.
                # Let's just iterate through new_complement and add if possible.
                # This is O(|C| * |S|).
                
                added_any = False
                # To improve refill quality, we can sort new_complement by 
                # number of neighbors in new_S_set.
                # But that's O(|C| * |S|) per sort.
                # Let's try a simple pass first.
                
                for x in list(new_complement):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        new_complement.discard(x)
                        added_any = True
                        
                # If we added more than 1, it's an improvement
                # (We removed 1, added k. Net gain k-1. If k>=2, gain >=1)
                if len(new_S_set) > best_size:
                    best_S = list(new_S_set)
                    best_S_set = new_S_set
                    best_size = len(new_S_set)
                    improved = True
                    # Update global state immediately?
                    # Let's wait until end of loop to pick best, or update now?
                    # Updating now allows subsequent swaps to build on it.
                    S = best_S
                    S_set = best_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    # Break out of u-loop to restart with new S?
                    # Yes, because S changed.
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
    for i in range(50):
        if time.time() - t0 > _DEADLINE - 5:
            break
        
        # Variation 1: Degree sorted
        if i % 2 == 0:
            S, S_set = _greedy_init(neighbors, rng)
        else:
            # Random order
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S = []
            S_set = set()
            for x in order:
                if _can_add(x, S_set, neighbors):
                    S.append(x)
                    S_set.add(x)
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 2:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 2)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Final validation
    # (Optional, but good for safety)
    # for i in range(len(best_S)):
    #    for j in range(i+1, len(best_S)):
    #       if (best_S[i]*best_S[j]) % (best_S[i]+best_S[j]) == 0:
    #          # Invalid, fallback to odds
    #          return list(range(1, 257, 2))
          
    return best_S
