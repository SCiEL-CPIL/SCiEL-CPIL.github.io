"""
Maximize cardinality of a 4-term-AP-free subset of [1, 512].

Strategy:
1. Greedy seed construction (fast baseline).
2. Simulated annealing with bit-parallel conflict detection to escape local optima.
3. Deadline guard to ensure we return within 90s wall-clock.
"""

import time
import random

N = 512

def _precompute_conflict_masks():
    """
    For each vertex x in [1, N], precompute a 512-bit integer mask where bit i is set
    if adding x to a set containing i would create a nonconstant 4-AP.
    
    A 4-AP is (a, a+d, a+2d, a+3d) with d != 0.
    If we add x, conflicts arise if there exist d != 0 such that:
      - x = a+3d, and a, a+d, a+2d are all in the set
      - x = a+2d, and a, a+d, a+3d are all in the set
      - x = a+d, and a, a+2d, a+3d are all in the set
      - x = a, and a+d, a+2d, a+3d are all in the set
    
    For the mask of x, we mark all y such that {x, y} is part of some 4-AP pattern
    that could be completed. Actually, for simulated annealing, we need to check
    if adding x to current set S creates any 4-AP. This is equivalent to checking
    if there exist y1, y2 in S such that {x, y1, y2, y3} forms a 4-AP for some y3.
    
    A simpler approach: for each x, precompute all pairs/triples that would form a 4-AP
    with x. But for SA, we need to check conflicts efficiently.
    
    Let's use a different approach: represent the set as a bitmask (512 bits).
    For each x, precompute a list of "forbidden patterns" - triples (a, b, c) such that
    {x, a, b, c} is a 4-AP. Then x can be added to S iff no such triple is fully in S.
    
    Actually, let's think differently. For simulated annealing, we want to:
    - Start with a valid set
    - Try to add/remove elements
    - Check validity quickly
    
    Let's precompute for each x the set of y such that x and y are in the same 4-AP.
    But that's not quite right either.
    
    Better approach: For each x, precompute a list of all 4-APs that contain x.
    When checking if we can add x to S, we check if any 4-AP containing x has its
    other 3 elements all in S.
    """
    # For each x in 1..N, find all 4-APs containing x
    # A 4-AP is (a, a+d, a+2d, a+3d) with d>0, 1<=a, a+3d<=N
    aps_containing = [[] for _ in range(N+1)]  # 1-indexed
    for a in range(1, N+1):
        for d in range(1, (N - a) // 3 + 1):
            ap = (a, a+d, a+2*d, a+3*d)
            for x in ap:
                # Store the other three elements
                others = tuple(y for y in ap if y != x)
                aps_containing[x].append(others)
    return aps_containing

def _greedy_construct():
    """Greedy forward construction: add x if it doesn't create a 4-AP with existing elements."""
    selected = set()
    for x in range(1, N+1):
        # Check if adding x creates any 4-AP
        conflict = False
        for d in range(1, (x-1)//3 + 1):
            # x could be the 4th term: x-3d, x-2d, x-d
            if x-3*d in selected and x-2*d in selected and x-d in selected:
                conflict = True
                break
            # x could be the 3rd term: x-2d, x-d, x+d
            if x-2*d in selected and x-d in selected and x+d in selected:
                conflict = True
                break
            # x could be the 2nd term: x-d, x+d, x+2d
            if x-d in selected and x+d in selected and x+2*d in selected:
                conflict = True
                break
            # x could be the 1st term: x+d, x+2d, x+3d
            if x+d in selected and x+2*d in selected and x+3*d in selected:
                conflict = True
                break
        if not conflict:
            selected.add(x)
    return selected

def _check_validity(selected):
    """Verify that selected contains no nonconstant 4-AP."""
    sel = set(selected)
    for x in sel:
        for d in range(1, (x-1)//3 + 1):
            if x-3*d in sel and x-2*d in sel and x-d in sel:
                return False
            if x-2*d in sel and x-d in sel and x+d in sel:
                return False
            if x-d in sel and x+d in sel and x+2*d in sel:
                return False
            if x+d in sel and x+2*d in sel and x+3*d in sel:
                return False
    return True

def _simulated_annealing(initial_set, aps_containing, time_limit=80.0):
    """
    Simulated annealing to maximize |S| subject to no 4-AP.
    
    Moves:
    - Add an element not in S (if it doesn't create a 4-AP)
    - Remove an element from S
    - Swap: remove one, add another
    
    We track the set as a Python set for simplicity, but use precomputed APs for conflict checks.
    """
    start_time = time.time()
    S = set(initial_set)
    best_S = set(S)
    best_size = len(S)
    
    # Temperature schedule
    T = 5.0
    cooling = 0.9995
    
    random.seed(42)
    
    def can_add(x, S):
        """Check if x can be added to S without creating a 4-AP."""
        for others in aps_containing[x]:
            if all(y in S for y in others):
                return False
        return True
    
    def try_add(x):
        """Try to add x to S. Return True if successful."""
        if x in S:
            return False
        if can_add(x, S):
            S.add(x)
            return True
        return False
    
    def try_remove(x):
        """Remove x from S."""
        if x in S:
            S.discard(x)
            return True
        return False
    
    num_iters = 0
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Random move
        move_type = random.random()
        if move_type < 0.5:
            # Try to add a random element
            candidates = [x for x in range(1, N+1) if x not in S]
            if candidates:
                x = random.choice(candidates)
                if can_add(x, S):
                    S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    # Accept with probability exp(-1/T) since this is a "worse" move in terms of size
                    if random.random() < (0.3 / T):  # scaled probability
                        # Actually, adding when it conflicts is invalid, so we can't do that.
                        # Let's just skip this.
                        pass
        elif move_type < 0.8:
            # Try to remove a random element
            if S:
                x = random.choice(list(S))
                S.discard(x)
        else:
            # Swap: remove one, try to add another
            if S:
                x = random.choice(list(S))
                S.discard(x)
                candidates = [y for y in range(1, N+1) if y not in S and y != x]
                if candidates:
                    y = random.choice(candidates)
                    if can_add(y, S):
                        S.add(y)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
        
        num_iters += 1
        if num_iters % 10000 == 0:
            # Periodic check
            pass
    
    return best_S

def construct():
    """
    Construct a 4-term-AP-free subset of [1, 512] with maximum cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Step 1: Greedy construction
    S = _greedy_construct()
    best_S = set(S)
    best_size = len(S)
    
    # Step 2: Precompute APs containing each element
    aps_containing = _precompute_conflict_masks()
    
    # Step 3: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_result = _simulated_annealing(S, aps_containing, time_limit=deadline - time.time() - 5.0)
        if len(sa_result) > best_size:
            best_S = sa_result
            best_size = len(sa_result)
    
    # Step 4: Local search polish - try to add any element that doesn't conflict
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N+1):
            if x not in best_S and can_add_x(x, best_S, aps_containing):
                best_S.add(x)
                best_size = len(best_S)
                changed = True
    
    # Verify validity
    if not _check_validity(best_S):
        # Fallback to greedy
        best_S = _greedy_construct()
    
    return sorted(best_S)

def can_add_x(x, S, aps_containing):
    """Check if x can be added to S without creating a 4-AP."""
    for others in aps_containing[x]:
        if all(y in S for y in others):
            return False
    return True
