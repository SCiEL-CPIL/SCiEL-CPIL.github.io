import time
import random

_N = 256
_DEADLINE = 82.0  # Safety margin from 90s kill

def _build_conflict_bitsets():
    """
    Build conflict bitsets for fast membership testing.
    conflict[i] is a bitset where bit j is set if (i*j) % (i+j) == 0.
    """
    # conflict[i] is an integer whose bit j is set if i and j conflict
    conflict = [0] * (_N + 1)
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    return conflict

def _can_add_fast(x, S_bitset, conflict):
    """Check if x can be added to S using bitset operations."""
    return (conflict[x] & S_bitset) == 0

def _greedy_build(conflict, order):
    """Greedy build given an order of nodes."""
    S = []
    S_bitset = 0
    S_set = set()
    for x in order:
        if _can_add_fast(x, S_bitset, conflict):
            S.append(x)
            S_bitset |= (1 << x)
            S_set.add(x)
    return S, S_bitset, S_set

def _greedy_refill_fast(S_bitset, S_set, complement, conflict, order=None):
    """Greedily add nodes from complement to S using bitsets."""
    if order is None:
        order = sorted(complement)
    new_S_bitset = S_bitset
    new_S_set = set(S_set)
    new_complement = set(complement)
    for x in order:
        if x not in new_complement:
            continue
        if _can_add_fast(x, new_S_bitset, conflict):
            new_S_set.add(x)
            new_S_bitset |= (1 << x)
            new_complement.discard(x)
    return new_S_bitset, new_S_set, new_complement

def _local_search(S, S_bitset, S_set, conflict, rng, time_limit):
    """
    Local search with bitset-accelerated checks.
    Tries: 1) additions, 2) 1-for-k swaps, 3) 2-for-k swaps.
    """
    S = list(S)
    S_bitset = S_bitset
    S_set = set(S_set)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Add any node that fits
        for x in list(complement):
            if _can_add_fast(x, S_bitset, conflict):
                S.append(x)
                S_bitset |= (1 << x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        
        if time.time() >= time_limit:
            break
            
        # Phase 2: Try 1-for-k swaps
        if not improved:
            best_size = len(S)
            S_list = list(S)
            for u in S_list:
                if time.time() >= time_limit:
                    break
                
                S_bitset_no_u = S_bitset & ~(1 << u)
                S_set_no_u = S_set - {u}
                comp_with_u = complement | {u}
                
                new_S_bitset, new_S_set, new_complement = _greedy_refill_fast(
                    S_bitset_no_u, S_set_no_u, comp_with_u, conflict
                )
                
                if len(new_S_set) > best_size:
                    S = list(new_S_set)
                    S_bitset = new_S_bitset
                    S_set = new_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    best_size = len(new_S_set)
                    improved = True
                    break
            
            if time.time() >= time_limit:
                break
            
        # Phase 3: Try 2-for-k swaps
        if not improved:
            best_size = len(S)
            S_list = list(S)
            n = len(S_list)
            for i in range(n):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() >= time_limit:
                        break
                    
                    u, v = S_list[i], S_list[j]
                    S_bitset_no_uv = S_bitset & ~(1 << u) & ~(1 << v)
                    S_set_no_uv = S_set - {u, v}
                    comp_with_uv = complement | {u, v}
                    
                    new_S_bitset, new_S_set, new_complement = _greedy_refill_fast(
                        S_bitset_no_uv, S_set_no_uv, comp_with_uv, conflict
                    )
                    
                    if len(new_S_set) > best_size:
                        S = list(new_S_set)
                        S_bitset = new_S_bitset
                        S_set = new_S_set
                        complement = set(range(1, _N + 1)) - S_set
                        best_size = len(new_S_set)
                        improved = True
                        break
                if improved:
                    break

    return S, S_bitset, S_set

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
    
    conflict = _build_conflict_bitsets()
    
    best_S = []
    best_size = 0
    best_bitset = 0
    best_set = set()
    
    # Multiple initialization strategies
    for i in range(80):
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        if i % 4 == 0:
            # Degree ascending: pick nodes with fewest conflicts first
            degrees = [bin(conflict[x]).count('1') for x in range(1, _N + 1)]
            order = list(range(1, _N + 1))
            order.sort(key=lambda x: (degrees[x], x))
        elif i % 4 == 1:
            # Random order
            order = list(range(1, _N + 1))
            rng.shuffle(order)
        elif i % 4 == 2:
            # Reverse degree: pick high-degree nodes first (risky but may find dense independent sets)
            degrees = [bin(conflict[x]).count('1') for x in range(1, _N + 1)]
            order = list(range(1, _N + 1))
            order.sort(key=lambda x: (-degrees[x], x))
        else:
            # Weighted random: prefer low-degree nodes
            weights = [1.0 / (1 + bin(conflict[x]).count('1')) for x in range(1, _N + 1)]
            order = list(range(1, _N + 1))
            # Custom weighted shuffle
            for idx in range(len(order) - 1, 0, -1):
                total_w = sum(weights[i] for i in range(idx + 1))
                r = rng.random() * total_w
                cum = 0
                for j in range(idx + 1):
                    cum += weights[j]
                    if cum >= r:
                        order[idx], order[j] = order[j], order[idx]
                        break
        
        S, S_bitset, S_set = _greedy_build(conflict, order)
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            best_bitset = S_bitset
            best_set = S_set
    
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 3:
        S_final, bitset_final, set_final = _local_search(
            best_S, best_bitset, best_set, conflict, rng, _DEADLINE - 3
        )
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            best_bitset = bitset_final
            best_set = set_final
    
    # Final validation safety net
    if not _validate(best_S):
        # Fallback to all odds (guaranteed valid: for odd a,b, a+b is even, a*b is odd, so (a*b)%(a+b) != 0)
        best_S = list(range(1, _N + 1, 2))
    
    return best_S
