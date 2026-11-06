"""Simulated annealing approach for maximum subset avoiding reciprocal Schur triples."""
import time
import random
import math

def _precompute_conflicts(N=180):
    """
    Precompute all forbidden triples. For each element x, store the list of pairs
    (y, z) such that {x, y, z} is a forbidden triple.
    """
    N = 180
    adj = {i: [] for i in range(1, N + 1)}
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if a >= 1 and a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    for elem in triple:
                        other1, other2 = [x for x in triple if x != elem]
                        adj[elem].append((other1, other2))
    
    return adj

def _can_add(x, S_set, adj):
    """Check if element x can be added to set S_set without creating a forbidden triple."""
    for (y, z) in adj[x]:
        if y in S_set and z in S_set:
            return False
    return True

def _initial_greedy(adj, N=180):
    """Build an initial valid set using greedy approach."""
    S_set = set()
    for x in range(1, N + 1):
        if _can_add(x, S_set, adj):
            S_set.add(x)
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 180
    adj = _precompute_conflicts(N)
    
    best_set = None
    best_size = 0
    
    # Try multiple random restarts
    num_restarts = 40
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        
        # Start from greedy or random initial set
        if restart == 0:
            S_set = _initial_greedy(adj, N)
        else:
            # Random initial: start with greedy, then randomly remove some elements
            S_set = _initial_greedy(adj, N)
            # Randomly remove 5-25% of elements
            elems = list(S_set)
            num_remove = max(1, len(elems) * rng.randint(5, 25) // 100)
            for _ in range(num_remove):
                if elems:
                    idx = rng.randint(0, len(elems) - 1)
                    S_set.discard(elems[idx])
                    elems[idx] = elems[-1]
                    elems.pop()
        
        if len(S_set) > best_size:
            best_set = set(S_set)
            best_size = len(S_set)
        
        # Simulated annealing with improved moves
        temp = 1.0
        cooling = 0.998
        min_temp = 0.001
        
        while temp > min_temp and time.time() < deadline - 3.0:
            # Maintain a list of elements not in S for efficient sampling
            not_in_S = [x for x in range(1, N + 1) if x not in S_set]
            in_S = list(S_set)
            
            if not not_in_S and not in_S:
                break
            
            # Do a batch of moves at this temperature
            steps = 200
            for _ in range(steps):
                if time.time() > deadline - 3.0:
                    break
                
                move_type = rng.random()
                
                if move_type < 0.35:
                    # Try to add a random element not in S
                    if not not_in_S:
                        continue
                    x = rng.choice(not_in_S)
                    if _can_add(x, S_set, adj):
                        S_set.add(x)
                        not_in_S.remove(x)
                        in_S.append(x)
                elif move_type < 0.6:
                    # Try to remove a random element from S
                    if not in_S:
                        continue
                    x = rng.choice(in_S)
                    S_set.discard(x)
                    in_S.remove(x)
                    not_in_S.append(x)
                else:
                    # Swap: remove x from S, add y not in S
                    if not in_S or not not_in_S:
                        continue
                    x = rng.choice(in_S)
                    y = rng.choice(not_in_S)
                    
                    # Check if swap is valid: remove x, then add y
                    S_set.discard(x)
                    in_S.remove(x)
                    not_in_S.append(x)
                    
                    if _can_add(y, S_set, adj):
                        S_set.add(y)
                        not_in_S.remove(y)
                        in_S.append(y)
                    else:
                        # Revert
                        S_set.add(x)
                        in_S.append(x)
                        not_in_S.remove(x)
                
                # Update best
                if len(S_set) > best_size:
                    best_set = set(S_set)
                    best_size = len(S_set)
            
            temp *= cooling
        
        # Local improvement: greedily add any remaining elements
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S_set and _can_add(x, S_set, adj):
                    S_set.add(x)
                    changed = True
        
        if len(S_set) > best_size:
            best_set = set(S_set)
            best_size = len(S_set)
    
    if best_set is None or not best_set:
        # Fallback: return the simple range
        return list(range(91, 181))
    
    return sorted(best_set)
