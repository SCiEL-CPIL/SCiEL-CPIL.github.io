"""Direct approach: construct a maximal no-isosceles set on [0,N)^2.
Uses greedy construction + local search refinement (remove redundant points, swap to escape local optima)."""

import time
import random
import itertools

N = 16
DEADLINE_SECONDS = 280  # safety margin before 300s kill

def _sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _is_iso_with_set(p, S):
    """Check if adding p to S creates any isosceles triangle with two points in S."""
    for a, b in itertools.combinations(S, 2):
        d_pa = _sqdist(p, a)
        d_pb = _sqdist(p, b)
        d_ab = _sqdist(a, b)
        if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
            return True
    return False

def _can_add(p, S):
    """True if p can be added to S without creating an isosceles triangle."""
    return not _is_iso_with_set(p, S)

def _build_greedy_order(order):
    """Simple greedy in a given order."""
    S = []
    for p in order:
        if _can_add(p, S):
            S.append(p)
    return S

def _is_maximal(S):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    for x in range(N):
        for y in range(N):
            p = (x, y)
            if p not in S_set and _can_add(p, S):
                return False
    return True

def _local_search(S, rng, deadline):
    """
    Local search refinement:
    1. Try removing each point (if result is still maximal).
    2. Try swap operations: remove one point, add another (if still valid and maximal).
    Iterates until no improvement or deadline.
    """
    S = list(S)
    S_set = set(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: try removing points
        for i in range(len(S)):
            if time.time() > deadline:
                return S
            removed = S[i]
            new_S = S[:i] + S[i+1:]
            if _is_maximal(new_S):
                S = new_S
                S_set = set(S)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: try swap operations (remove one, add one)
        all_points = list(itertools.product(range(N), repeat=2))
        candidates = [p for p in all_points if p not in S_set]
        rng.shuffle(candidates)
        
        for i in range(len(S)):
            if time.time() > deadline:
                return S
            removed = S[i]
            base_S = S[:i] + S[i+1:]
            base_set = set(base_S)
            
            for add_p in candidates:
                if time.time() > deadline:
                    return S
                if add_p in base_set:
                    continue
                if not _can_add(add_p, base_S):
                    continue
                new_S = base_S + [add_p]
                if _is_maximal(new_S):
                    if len(new_S) < len(S) or rng.random() < 0.3:
                        S = new_S
                        S_set = set(S)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try removing 2 random points
            if len(S) > 2 and rng.random() < 0.5:
                idxs = rng.sample(range(len(S)), 2)
                idxs.sort(reverse=True)
                new_S = S[:]
                for idx in idxs:
                    new_S.pop(idx)
                if _is_maximal(new_S):
                    S = new_S
                    S_set = set(S)
                    improved = True
        
        if not improved:
            break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_set = None
    best_size = float('inf')
    
    # Generate multiple greedy starting points with different orderings
    starting_sets = []
    
    row_major = list(itertools.product(range(N), repeat=2))
    S1 = _build_greedy_order(row_major)
    starting_sets.append(S1)
    
    col_major = sorted(itertools.product(range(N), repeat=2), key=lambda p: (p[1], p[0]))
    S2 = _build_greedy_order(col_major)
    starting_sets.append(S2)
    
    diag = sorted(itertools.product(range(N), repeat=2), key=lambda p: (abs(p[0]-p[1]), p[0], p[1]))
    S3 = _build_greedy_order(diag)
    starting_sets.append(S3)
    
    anti_diag = sorted(itertools.product(range(N), repeat=2), key=lambda p: (abs(p[0]+p[1]-N+1), p[0], p[1]))
    S4 = _build_greedy_order(anti_diag)
    starting_sets.append(S4)
    
    rng_global = random.Random(42)
    for _ in range(15):
        if time.time() > deadline - 10:
            break
        random_order = list(itertools.product(range(N), repeat=2))
        rng_global.shuffle(random_order)
        S = _build_greedy_order(random_order)
        starting_sets.append(S)
    
    # Local search on each starting set
    for S in starting_sets:
        if time.time() > deadline - 5:
            break
        
        for ls_seed in range(3):
            if time.time() > deadline - 3:
                break
            rng_ls = random.Random(ls_seed * 1000 + (hash(S[0]) if S else 0))
            S_refined = _local_search(S, rng_ls, deadline - 2)
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
    
    if best_set is None:
        best_set = starting_sets[0]
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
