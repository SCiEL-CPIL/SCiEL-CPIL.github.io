"""Simulated annealing approach for maximum subset avoiding reciprocal Schur triples."""
import time
import random
import math

def _precompute_conflicts(N=180):
    """
    Precompute for each element x in [1,N], the set of elements y such that
    there exists a third distinct element z with 1/x = 1/y + 1/z (or any permutation).
    Actually, we need: for each pair (a, b), determine if adding both creates a conflict
    with some c. Better: precompute all forbidden triples, then for each element,
    store which other elements it 'conflicts' with (i.e., cannot be in the same set
    because there's a third element that would complete a triple).
    
    Wait, the constraint is: no THREE DISTINCT elements a,b,c with 1/a=1/b+1/c.
    So a set S is valid iff for every triple of distinct elements in S, the reciprocal
    equation doesn't hold.
    
    We precompute: for each pair (i,j) with i<j, the set of k such that {i,j,k} is a
    forbidden triple. Then a set S is valid iff for every pair (i,j) in S, no k in S
    is in the forbidden set for (i,j).
    
    More efficiently: for each element x, precompute conflicts[x] = set of all y such
    that there exists z (distinct from x,y) with {x,y,z} forming a forbidden triple.
    But that's not quite right either - the conflict depends on the third element being
    in the set.
    
    Let me think again. The forbidden condition: there exist distinct a,b,c in S with
    1/a = 1/b + 1/c. This is equivalent to: bc = a(b+c), so a = bc/(b+c).
    
    For a given pair (b,c), a is determined: a = bc/(b+c). If a is integer and distinct
    from b,c, then {a,b,c} is a forbidden triple.
    
    So: precompute all forbidden triples. For each element x, store the set of pairs
    (y,z) such that {x,y,z} is forbidden. Then when checking if we can add element x
    to set S, we need to check: for each pair (y,z) in S where {x,y,z} is forbidden,
    reject.
    
    For O(1) conflict checking during annealing, let's precompute for each x, the set
    of elements y such that there exists z with {x,y,z} forbidden. But that's not
    sufficient because we also need z to be in the set.
    
    Alternative: For each x, precompute forbidden_pairs_with_x[x] = set of (y,z) tuples
    where {x,y,z} is a forbidden triple. Then to check if x can be added to S:
    for each (y,z) in forbidden_pairs_with_x[x], if y in S and z in S, conflict.
    
    But this could be slow if there are many such pairs. Let's just precompute the full
    list of forbidden triples and use a different approach.
    
    Actually, for simulated annealing, we'll be doing single-element additions/removals.
    When adding element x to set S: check if any forbidden triple contains x and two
    other elements both in S.
    When removing element x from set S: always valid (removing can't create a new triple).
    
    So we need: for each x, the list of pairs (y,z) such that {x,y,z} is a forbidden
    triple. Let's call this adj[x]. To check if x can be added to S:
    for (y,z) in adj[x]: if y in S and z in S: return False
    return True
    
    This should be fast enough if adj[x] is small.
    """
    N = 180
    # adj[x] = list of (y, z) such that {x, y, z} is a forbidden triple
    adj = {i: [] for i in range(1, N + 1)}
    
    # Generate all forbidden triples
    # 1/a = 1/b + 1/c => a = bc/(b+c)
    # Iterate over all pairs (b, c) with b < c, compute a, check if valid
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if a >= 1 and a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    # Add to adj for each element in the triple
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
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 180
    adj = _precompute_conflicts(N)
    
    best_set = None
    best_size = 0
    
    # Try multiple random restarts
    num_restarts = 20
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Start from greedy or random initial set
        if restart == 0:
            S_set = _initial_greedy(adj, N)
        else:
            # Random initial: start with greedy, then randomly remove some elements
            S_set = _initial_greedy(adj, N)
            # Randomly remove 10-30% of elements
            elems = list(S_set)
            num_remove = max(1, len(elems) * rng.randint(10, 30) // 100)
            for _ in range(num_remove):
                if elems:
                    idx = rng.randint(0, len(elems) - 1)
                    S_set.discard(elems[idx])
                    elems[idx] = elems[-1]
                    elems.pop()
        
        if len(S_set) > best_size:
            best_set = set(S_set)
            best_size = len(S_set)
        
        # Simulated annealing
        temp = 1.0
        cooling = 0.995
        min_temp = 0.001
        steps_per_temp = 500
        
        while temp > min_temp and time.time() < deadline - 2.0:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 2.0:
                    break
                
                # Choose a random move: add, remove, or swap
                move_type = rng.random()
                all_elems = list(range(1, N + 1))
                
                if move_type < 0.4:
                    # Try to add a random element not in S
                    candidates = [x for x in all_elems if x not in S_set]
                    if not candidates:
                        continue
                    x = rng.choice(candidates)
                    if _can_add(x, S_set, adj):
                        S_set.add(x)
                    # Always accept adds (they increase size or keep same)
                elif move_type < 0.7:
                    # Try to remove a random element from S
                    if not S_set:
                        continue
                    x = rng.choice(list(S_set))
                    S_set.discard(x)
                    # Always accept removes
                else:
                    # Swap: remove x from S, add y not in S
                    if not S_set:
                        continue
                    x = rng.choice(list(S_set))
                    candidates = [z for z in all_elems if z not in S_set]
                    if not candidates:
                        continue
                    y = rng.choice(candidates)
                    
                    # Check if swap is valid: remove x, then add y
                    S_set.discard(x)
                    if _can_add(y, S_set, adj):
                        S_set.add(y)
                    else:
                        # Revert
                        S_set.add(x)
                    
                    if len(S_set) > best_size:
                        best_set = set(S_set)
                        best_size = len(S_set)
                
                # Update best
                if len(S_set) > best_size:
                    best_set = set(S_set)
                    best_size = len(S_set)
            
            temp *= cooling
        
        # Local improvement: try to add any remaining elements
        for x in range(1, N + 1):
            if x not in S_set and _can_add(x, S_set, adj):
                S_set.add(x)
        
        if len(S_set) > best_size:
            best_set = set(S_set)
            best_size = len(S_set)
    
    if best_set is None or not best_set:
        # Fallback: return the simple range
        return list(range(91, 181))
    
    return sorted(best_set)
