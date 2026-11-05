"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _build_conflict_map():
    """conflict[v] = frozenset of all u in 1..256 (u != v) such that v+u is a perfect square.
       Also include v itself if 2v is a perfect square (diagonal constraint)."""
    conflict = {}
    for v in range(1, 257):
        s = set()
        for u in range(1, 257):
            if u != v and _is_square(v + u):
                s.add(u)
        # Diagonal: v+v = 2v must not be a square
        if _is_square(2 * v):
            s.add(v)
        conflict[v] = frozenset(s)
    return conflict

def _can_add(v, current_set, conflict):
    """Check if v can be added to current_set without violating constraints."""
    # v must not conflict with any element in current_set
    # Also v itself must not be a self-conflict (2v is square)
    if v in conflict[v]:
        # This means 2v is a square, so v can never be in any valid set
        return False
    return len(conflict[v] & current_set) == 0

def _greedy_construct(conflict, order):
    result = set()
    for v in order:
        if _can_add(v, result, conflict):
            result.add(v)
    return result

def _local_search(initial_set, conflict, time_limit, rng):
    current = set(initial_set)
    best = set(initial_set)
    best_size = len(best)
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 10000
    
    all_values = list(range(1, 257))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try add moves
        add_candidates = [v for v in all_values if v not in current and _can_add(v, current, conflict)]
        if add_candidates:
            v = rng.choice(add_candidates)
            current.add(v)
            improved = True
            if len(current) > best_size:
                best = set(current)
                best_size = len(best)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Try swap moves
            if len(current) > 0:
                to_remove = rng.choice(list(current))
                temp = current - {to_remove}
                add_candidates = [v for v in all_values if v not in current and _can_add(v, temp, conflict)]
                if add_candidates:
                    v = rng.choice(add_candidates)
                    current.discard(to_remove)
                    current.add(v)
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(best)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Try remove move
                    if len(current) > 1:
                        to_remove = rng.choice(list(current))
                        current.discard(to_remove)
                        no_improve += 1
                    else:
                        no_improve += 100
            else:
                no_improve += 100
        
        if not improved:
            no_improve += 50
        
        if no_improve > max_no_improve:
            break
    
    return list(best)

def construct():
    deadline = time.time() + 80
    
    conflict = _build_conflict_map()
    
    # Initial greedy
    order = list(range(1, 257))
    best_result = _greedy_construct(conflict, order)
    best_size = len(best_result)
    
    rng = random.Random(12345)
    
    # Multiple restarts
    while time.time() < deadline - 10:
        order = list(range(1, 257))
        rng.shuffle(order)
        
        greedy_set = _greedy_construct(conflict, order)
        
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        
        ls_result = _local_search(greedy_set, conflict, min(remaining, 15), rng)
        
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_result = ls_result
    
    # Final polish on best
    remaining = deadline - time.time() - 2
    if remaining > 1:
        final_result = _local_search(best_result, conflict, remaining, rng)
        if len(final_result) > best_size:
            best_result = final_result
    
    if not best_result:
        best_result = [1]
    
    return list(best_result)
# EVOLVE-BLOCK-END
