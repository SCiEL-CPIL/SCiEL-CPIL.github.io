"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import itertools
import math

# EVOLVE-BLOCK-START
ALL_POINTS = list(itertools.product(range(N), repeat=2))
ALL_POINTS_SET = set(ALL_POINTS)

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    # chosen is a list
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i + 1, n):
            b = chosen[j]
            if _is_iso(p, a, b):
                return False
    return True

def _count_addable(chosen):
    chosen_set = set(chosen)
    count = 0
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            count += 1
    return count

def _is_maximal(chosen):
    return _count_addable(chosen) == 0

def _is_valid_no_iso(chosen):
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_iso(chosen[i], chosen[j], chosen[k]):
                    return False
    return True

def _greedy_init(rng):
    pts = ALL_POINTS[:]
    rng.shuffle(pts)
    chosen = []
    for p in pts:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Try several initial greedy solutions
    for _ in range(10):
        if time.time() > deadline - 10: break
        S = _greedy_init(rng)
        addable = _count_addable(S)
        if addable == 0:
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
    
    # Simulated Annealing
    # We want to minimize: size + penalty * addable
    # If addable > 0, it's not a valid maximal set.
    # We prioritize reducing addable to 0.
    
    # Pick the best initial set found, or a random one if none maximal
    if best_set is not None:
        current_S = best_set[:]
    else:
        current_S = _greedy_init(rng)
        
    current_addable = _count_addable(current_S)
    current_size = len(current_S)
    
    # If we already have a maximal set, we can try to shrink it
    # If not, we need to make it maximal.
    
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    
    while time.time() < deadline:
        if T < min_T:
            # Restart or break?
            # Let's restart with a new random seed if stuck
            if rng.random() < 0.1:
                current_S = _greedy_init(rng)
                current_addable = _count_addable(current_S)
                current_size = len(current_S)
                T = 50.0
            else:
                break
                
        # Generate a move
        if current_size > 0 and rng.random() < 0.5:
            # Remove a point
            idx = rng.randint(0, current_size - 1)
            p_remove = current_S[idx]
            new_S = current_S[:idx] + current_S[idx+1:]
            # new_S is always no-isosceles if current_S was
            new_addable = _count_addable(new_S)
            new_size = len(new_S)
            
            # Score: We want to minimize size, but MUST have addable=0 eventually.
            # Penalty for addable points is high.
            score_new = new_size + 100 * new_addable
            score_old = current_size + 100 * current_addable
            
            if score_new < score_old or (score_new > score_old and rng.random() < math.exp(-(score_new - score_old)/T)):
                current_S = new_S
                current_addable = new_addable
                current_size = new_size
                
                if current_addable == 0 and current_size < best_size:
                    best_size = current_size
                    best_set = current_S[:]
        else:
            # Add a point
            # Choose a random point not in S
            # To bias towards "useful" points, we could pick from addable points?
            # No, adding an addable point makes it NOT addable (it becomes part of S).
            # But it might make OTHER points un-addable.
            # Let's just pick random.
            p_add = rng.choice(ALL_POINTS)
            if p_add in current_S:
                continue
            if not _can_add(p_add, current_S):
                continue
                
            new_S = current_S + [p_add]
            new_addable = _count_addable(new_S)
            new_size = len(new_S)
            
            score_new = new_size + 100 * new_addable
            score_old = current_size + 100 * current_addable
            
            if score_new < score_old or (score_new > score_old and rng.random() < math.exp(-(score_new - score_old)/T)):
                current_S = new_S
                current_addable = new_addable
                current_size = new_size
                
                if current_addable == 0 and current_size < best_size:
                    best_size = current_size
                    best_set = current_S[:]
        
        T *= cooling

    if best_set is None:
        # Fallback: Construct a maximal set greedily
        S = []
        for p in ALL_POINTS:
            if _can_add(p, S):
                S.append(p)
        # Ensure maximal by adding any remaining addable points
        changed = True
        while changed:
            changed = False
            for p in ALL_POINTS:
                if p not in S and _can_add(p, S):
                    S.append(p)
                    changed = True
        best_set = S

    return best_set
# EVOLVE-BLOCK-END
