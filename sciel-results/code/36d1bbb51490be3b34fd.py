"""Linear triple systems on 16 points with small independence number."""
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _validate(triples):
    """Check that all triples are valid and the system is linear."""
    if not triples:
        return True
    seen = set()
    for t in triples:
        if len(t) != 3:
            return False
        if any(p < 0 or p > 15 for p in t):
            return False
        if len(set(t)) != 3:
            return False
        key = tuple(sorted(t))
        if key in seen:
            return False
        seen.add(key)
    # Check linearity: any two triples share at most one point
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            inter = set(triples[i]) & set(triples[j])
            if len(inter) > 1:
                return False
    return True

def _calc_alpha_bitmask(triples):
    """
    Compute independence number via bitmask brute force.
    For 16 points, 2^16 = 65536 subsets, each checked against all triples.
    This is fast enough for our purposes.
    """
    if not triples:
        return 16
    
    triple_masks = []
    for t in triples:
        mm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mm)
    
    best = 0
    for mask in range(1 << 16):
        independent = True
        for tm in triple_masks:
            if (tm & mask) == tm:
                independent = False
                break
        if independent:
            cnt = bin(mask).count('1')
            if cnt > best:
                best = cnt
    return best

def _calc_alpha_ilp(triples):
    """
    Compute independence number via ILP (scipy.optimize.milp).
    Maximizes sum(x_i) subject to x_i + x_j + x_k <= 2 for each triple {i,j,k}.
    """
    if not triples:
        return 16
    
    n = 16
    m = len(triples)
    
    c = -np.ones(n)
    
    A = np.zeros((m, n), dtype=int)
    for idx, t in enumerate(triples):
        A[idx, t[0]] = 1
        A[idx, t[1]] = 1
        A[idx, t[2]] = 1
    
    constraint = LinearConstraint(A, lb=-np.inf, ub=np.ones(m))
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)
    
    result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
    
    if result.success:
        return int(round(-result.fun))
    else:
        return _calc_alpha_bitmask(triples)

def _calc_alpha(triples):
    """Compute independence number: largest set of points containing no triple."""
    # Use bitmask for speed when number of triples is moderate
    if len(triples) <= 50:
        return _calc_alpha_bitmask(triples)
    else:
        return _calc_alpha_ilp(triples)

def _max_triples_bound():
    """
    Upper bound on number of triples in a linear 3-graph on 16 points.
    Each triple uses 3 pairs, and there are C(16,2)=120 pairs total.
    So at most 120/3 = 40 triples.
    """
    return 40

def _build_linear_graph_backtrack(n_points=16, deadline=None, seed=42, max_triples_target=40):
    """
    CSP-style backtracking with forward checking to build a linear 3-graph
    that maximizes the number of triples (to minimize independence number).
    
    Key insight: to minimize alpha, we want as many triples as possible,
    and we want them distributed to hit all large independent sets.
    """
    rng = random.Random(seed)
    all_triples = list(combinations(range(n_points), 3))
    
    best_triples = []
    best_count = 0
    
    def backtrack(remaining_triples, current_triples, pair_used, depth=0):
        nonlocal best_triples, best_count
        
        if deadline is not None and time.time() > deadline:
            return
        
        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_triples = [list(t) for t in current_triples]
        
        if len(current_triples) >= max_triples_target:
            return
        
        if not remaining_triples:
            return
        
        # Forward checking: filter out triples that conflict with current
        valid = []
        for t in remaining_triples:
            ok = True
            for a, b in combinations(t, 2):
                if (min(a,b), max(a,b)) in pair_used:
                    ok = False
                    break
            if ok:
                valid.append(t)
        
        if not valid:
            return
        
        # Order by number of conflicts (fewest first) for better pruning
        # Actually, let's try a different heuristic: prioritize triples that
        # cover "rare" pairs
        
        # Shuffle for diversity
        rng.shuffle(valid)
        
        # Try each valid triple
        for t in valid:
            if deadline is not None and time.time() > deadline:
                return
            
            # Add triple
            new_pair_used = set(pair_used)
            for a, b in combinations(t, 2):
                new_pair_used.add((min(a,b), max(a,b)))
            
            current_triples.append(t)
            
            # Recurse
            backtrack(valid[valid.index(t)+1:], current_triples, new_pair_used, depth+1)
            
            # Undo
            current_triples.pop()
    
    # Initial call
    backtrack(all_triples, [], set())
    
    return best_triples

def _local_search_reduce_alpha(triples, n_points=16, deadline=None, max_iter=500):
    """
    Local search to reduce independence number by adding/removing triples.
    Focus on adding triples that are "most useful" for reducing alpha.
    """
    if not triples:
        return triples, 16
    
    current = [list(t) for t in triples]
    best_alpha = _calc_alpha(current)
    best_triples = [list(t) for t in current]
    
    all_triples = list(combinations(range(n_points), 3))
    rng = random.Random(12345)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if best_alpha <= 5:
            break
        
        # Build pair_used set
        pair_used = set()
        for t in current:
            for a, b in combinations(t, 2):
                pair_used.add((min(a,b), max(a,b)))
        
        # Find all valid triples that can be added
        valid_adds = []
        for t in all_triples:
            ok = True
            for a, b in combinations(t, 2):
                if (min(a,b), max(a,b)) in pair_used:
                    ok = False
                    break
            if ok:
                valid_adds.append(t)
        
        if not valid_adds:
            # No more triples can be added; try removing some and re-adding
            if len(current) > 5:
                # Remove a random triple and try to find better ones
                idx = rng.randrange(len(current))
                removed = current.pop(idx)
                for a, b in combinations(removed, 2):
                    pair_used.discard((min(a,b), max(a,b)))
                
                # Try to add different triples
                valid_adds = []
                for t in all_triples:
                    ok = True
                    for a, b in combinations(t, 2):
                        if (min(a,b), max(a,b)) in pair_used:
                            ok = False
                            break
                    if ok:
                        valid_adds.append(t)
                
                if valid_adds:
                    # Add a few random valid triples
                    for _ in range(min(3, len(valid_adds))):
                        t = rng.choice(valid_adds)
                        current.append(t)
                        for a, b in combinations(t, 2):
                            pair_used.add((min(a,b), max(a,b)))
                        valid_adds.remove(t)
                
                new_alpha = _calc_alpha(current)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = [list(t) for t in current]
                else:
                    # Revert
                    for _ in range(3):
                        if current:
                            t = current.pop()
                            for a, b in combinations(t, 2):
                                pair_used.discard((min(a,b), max(a,b)))
                    current.append(removed)
                    for a, b in combinations(removed, 2):
                        pair_used.add((min(a,b), max(a,b)))
            continue
        
        # Try adding one valid triple
        # Sort by some heuristic: prefer triples that cover points with high degree
        point_degree = [0] * n_points
        for t in current:
            for p in t:
                point_degree[p] += 1
        
        # Score each valid triple by sum of degrees of its points
        scored = []
        for t in valid_adds:
            score = point_degree[t[0]] + point_degree[t[1]] + point_degree[t[2]]
            scored.append((score, t))
        
        scored.sort(reverse=True)
        
        # Try top few
        tried = 0
        for score, t in scored[:10]:
            if deadline is not None and time.time() > deadline:
                break
            
            current.append(t)
            for a, b in combinations(t, 2):
                pair_used.add((min(a,b), max(a,b)))
            
            new_alpha = _calc_alpha(current)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = [list(t) for t in current]
                tried += 1
            else:
                current.pop()
                for a, b in combinations(t, 2):
                    pair_used.discard((min(a,b), max(a,b)))
            
            if tried >= 3:
                break
        
        if best_alpha <= 5:
            break
    
    return best_triples, best_alpha

def _greedy_packed(n_points=16, n_iter=200, seed=42, deadline=None):
    """
    Greedy construction that packs triples densely.
    Uses a priority queue approach: always add the triple that covers
    the most "under-covered" pairs.
    """
    rng = random.Random(seed)
    all_triples = list(combinations(range(n_points), 3))
    
    best_triples = []
    best_alpha = 16
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        triples = []
        pair_used = set()
        
        # Shuffle
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        for t in shuffled:
            ok = True
            for a, b in combinations(t, 2):
                if (min(a,b), max(a,b)) in pair_used:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
                for a, b in combinations(t, 2):
                    pair_used.add((min(a,b), max(a,b)))
        
        alpha = _calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
            if alpha <= 5:
                break
    
    return best_triples, best_alpha

def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_triples = []
    best_alpha = 16
    
    # Phase 1: Backtracking to build dense linear graph
    bt_triples = _build_linear_graph_backtrack(
        n_points=16,
        deadline=deadline,
        seed=42,
        max_triples_target=40
    )
    
    if bt_triples:
        bt_alpha = _calc_alpha(bt_triples)
        if bt_alpha < best_alpha:
            best_alpha = bt_alpha
            best_triples = bt_triples
    
    if time.time() > deadline:
        return best_triples
    
    # Phase 2: Local search to reduce alpha further
    improved_triples, improved_alpha = _local_search_reduce_alpha(
        best_triples,
        n_points=16,
        deadline=deadline,
        max_iter=500
    )
    
    if improved_alpha < best_alpha:
        best_alpha = improved_alpha
        best_triples = improved_triples
    
    if time.time() > deadline:
        return best_triples
    
    # Phase 3: More greedy restarts with different seeds
    for seed in [7, 42, 99, 2024, 31415, 1234, 5678, 9999, 11111, 22222]:
        if time.time() > deadline:
            break
        
        t, a = _greedy_packed(
            n_points=16,
            n_iter=100,
            seed=seed,
            deadline=deadline
        )
        if a < best_alpha:
            best_alpha = a
            best_triples = t
            
            # Local search on this
            t2, a2 = _local_search_reduce_alpha(
                best_triples,
                n_points=16,
                deadline=deadline,
                max_iter=200
            )
            if a2 < best_alpha:
                best_alpha = a2
                best_triples = t2
        
        if best_alpha <= 5:
            break
    
    if time.time() > deadline:
        return best_triples
    
    # Phase 4: Final local search
    final_triples, final_alpha = _local_search_reduce_alpha(
        best_triples,
        n_points=16,
        deadline=deadline,
        max_iter=300
    )
    
    if final_alpha < best_alpha:
        best_triples = final_triples
    
    return best_triples

# EVOLVE-BLOCK-END
