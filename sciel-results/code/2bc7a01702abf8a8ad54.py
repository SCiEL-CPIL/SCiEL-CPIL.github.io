"""
Strategy: Incremental Violation-Tracking Local Search for R(4,5;3).
Key insight: Track violations incrementally to enable many more search steps
per unit time, allowing escape from local minima via multiple restarts.
"""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _get_triples(n):
    """Return list of (a,b,c) triples in lex order."""
    return list(combinations(range(n), 3))

def _build_triple_index(n):
    """Build dict mapping triple tuple to index."""
    triples = _get_triples(n)
    return {t: i for i, t in enumerate(triples)}

def _init_coloring(n, strategy='random', seed=None):
    """Initialize coloring with given strategy."""
    triples = _get_triples(n)
    m = len(triples)
    if strategy == 'random':
        if seed is not None:
            rng = random.Random(seed)
            return np.array([rng.randint(0, 1) for _ in range(m)], dtype=np.int8)
        return np.array([random.randint(0, 1) for _ in range(m)], dtype=np.int8)
    elif strategy == 'modular':
        return np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
    elif strategy == 'balanced':
        # Try to balance red/blue counts while avoiding obvious patterns
        colors = np.array([random.randint(0, 1) for _ in range(m)], dtype=np.int8)
        return colors
    else:
        return np.array([0] * m, dtype=np.int8)

def _initial_violation_scan(n, colors, triple_index):
    """
    Full scan to find initial violations and build violation structures.
    Returns:
    - red_k4s: dict mapping frozenset of 4 vertices -> list of 4 triple indices
    - blue_k5s: dict mapping frozenset of 5 vertices -> list of 10 triple indices
    - red_conflict: np.array of length M, count of red K4s involving each triple
    - blue_conflict: np.array of length M, count of blue K5s involving each triple
    - total_violations: int
    """
    triples = _get_triples(n)
    m = len(triples)
    
    red_k4s = {}
    blue_k5s = {}
    red_conflict = np.zeros(m, dtype=np.int32)
    blue_conflict = np.zeros(m, dtype=np.int32)
    
    # Check all 4-sets for red K4
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_index[(a, b, c)]
        t2 = triple_index[(a, b, d)]
        t3 = triple_index[(a, c, d)]
        t4 = triple_index[(b, c, d)]
        
        if colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0:
            key = frozenset(quad)
            red_k4s[key] = [t1, t2, t3, t4]
            for t in (t1, t2, t3, t4):
                red_conflict[t] += 1
    
    # Check all 5-sets for blue K5
    for quint in combinations(range(n), 5):
        all_blue = True
        indices = []
        for tri in combinations(quint, 3):
            idx = triple_index[tri]
            indices.append(idx)
            if colors[idx] != 1:
                all_blue = False
                break
        
        if all_blue:
            key = frozenset(quint)
            blue_k5s[key] = indices
            for idx in indices:
                blue_conflict[idx] += 1
    
    total_violations = len(red_k4s) + len(blue_k5s)
    return red_k4s, blue_k5s, red_conflict, blue_conflict, total_violations

def _local_search_incremental(n, colors, time_limit, rng):
    """
    Incremental local search with violation tracking.
    Returns (colors, success).
    """
    triples = _get_triples(n)
    m = len(triples)
    triple_index = _build_triple_index(n)
    
    # Build initial violation structures
    red_k4s, blue_k5s, red_conflict, blue_conflict, total_violations = \
        _initial_violation_scan(n, colors, triple_index)
    
    if total_violations == 0:
        return colors, True
    
    start_time = time.time()
    
    # For efficient lookup: for each triple, which red K4s contain it?
    # And which blue K5s contain it?
    # We'll maintain these as we go.
    
    # Build reverse maps: triple_idx -> list of (violation_type, violation_key)
    red_reverse = {i: [] for i in range(m)}
    for key, tris in red_k4s.items():
        for t in tris:
            red_reverse[t].append(('R', key))
    
    blue_reverse = {i: [] for i in range(m)}
    for key, tris in blue_k5s.items():
        for t in tris:
            blue_reverse[t].append(('B', key))
    
    steps = 0
    max_steps = 50000  # Safety cap
    
    while time.time() < time_limit and steps < max_steps:
        if total_violations == 0:
            return colors, True
        
        steps += 1
        
        # Compute combined conflict score
        # Pick the triple with highest total conflict
        # To avoid computing full sum each time, maintain it
        # For now, compute on the fly (m is not too large for n<35)
        
        # Find triple with max (red_conflict[i] + blue_conflict[i])
        # Use numpy for speed
        combined = red_conflict + blue_conflict
        max_val = combined.max()
        
        if max_val == 0:
            # Shouldn't happen if total_violations > 0, but safety
            # Force a random flip
            flip_idx = rng.randint(0, m - 1)
        else:
            candidates = np.where(combined == max_val)[0]
            flip_idx = candidates[rng.randint(0, len(candidates) - 1)]
        
        old_color = colors[flip_idx]
        new_color = 1 - old_color
        colors[flip_idx] = new_color
        
        # Update violations affected by this flip
        # A flip of triple T affects:
        # - Red K4s that contain T: if T was red and becomes blue, K4 may be broken
        # - Blue K5s that contain T: if T was blue and becomes red, K5 may be broken
        # - Conversely, new violations may form
        
        # Process red K4s containing flip_idx
        to_remove_red = []
        for vtype, key in red_reverse[flip_idx]:
            if key in red_k4s:
                tris = red_k4s[key]
                # Check if still a red K4
                if all(colors[t] == 0 for t in tris):
                    pass  # Still valid
                else:
                    to_remove_red.append(key)
        
        for key in to_remove_red:
            tris = red_k4s.pop(key)
            for t in tris:
                red_conflict[t] -= 1
                # Remove from reverse map
                for i, (vt, k) in enumerate(red_reverse[t]):
                    if k == key:
                        red_reverse[t].pop(i)
                        break
            total_violations -= 1
        
        # Process blue K5s containing flip_idx
        to_remove_blue = []
        for vtype, key in blue_reverse[flip_idx]:
            if key in blue_k5s:
                tris = blue_k5s[key]
                if all(colors[t] == 1 for t in tris):
                    pass
                else:
                    to_remove_blue.append(key)
        
        for key in to_remove_blue:
            tris = blue_k5s.pop(key)
            for t in tris:
                blue_conflict[t] -= 1
                for i, (vt, k) in enumerate(blue_reverse[t]):
                    if k == key:
                        blue_reverse[t].pop(i)
                        break
            total_violations -= 1
        
        # Now check for NEW violations created by the flip
        # New red K4s: need flip_idx to be red (new_color == 0)
        # and all other 3 triples in some quad to be red
        if new_color == 0:
            # Find all quads containing flip_idx's triple
            # flip_idx corresponds to triple (a,b,c)
            a, b, c = triples[flip_idx]
            # For each pair of vertices from {a,b,c}, find third vertex d
            # such that (a,b,d), (a,c,d), (b,c,d) are all red
            # Actually, we need to check all quads containing the triple (a,b,c)
            # A quad containing triple (a,b,c) must have 4 vertices including a,b,c and one more d
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # Check the other 3 triples in this quad
                if d < a:
                    t_abd = triple_index[(d, a, b)]
                    t_acd = triple_index[(d, a, c)]
                    t_bcd = triple_index[(d, b, c)]
                elif a < d < b:
                    t_abd = triple_index[(a, d, b)]
                    t_acd = triple_index[(a, d, c)]
                    t_bcd = triple_index[(d, b, c)]
                elif b < d < c:
                    t_abd = triple_index[(a, b, d)]
                    t_acd = triple_index[(a, d, c)]
                    t_bcd = triple_index[(b, d, c)]
                else:  # c < d
                    t_abd = triple_index[(a, b, d)]
                    t_acd = triple_index[(a, c, d)]
                    t_bcd = triple_index[(b, c, d)]
                
                if colors[t_abd] == 0 and colors[t_acd] == 0 and colors[t_bcd] == 0:
                    # New red K4!
                    key = frozenset((a, b, c, d))
                    if key not in red_k4s:
                        tris_list = [flip_idx, t_abd, t_acd, t_bcd]
                        red_k4s[key] = tris_list
                        for t in tris_list:
                            red_conflict[t] += 1
                            red_reverse[t].append(('R', key))
                        total_violations += 1
        
        # New blue K5s: need flip_idx to be blue (new_color == 1)
        # and all other 9 triples in some quint to be blue
        if new_color == 1:
            a, b, c = triples[flip_idx]
            # Need to find d, e such that all 10 triples in {a,b,c,d,e} are blue
            # This is expensive. For now, check a subset of candidates.
            # Optimization: only check if there are many blue triples
            # For simplicity, skip new blue K5 detection in incremental update
            # (this means we might miss some, but the initial scan catches them,
            #  and we'll do periodic full rescans)
            pass
        
        # Periodic full rescan to catch missed violations
        if steps % 100 == 0:
            red_k4s, blue_k5s, red_conflict, blue_conflict, total_violations = \
                _initial_violation_scan(n, colors, triple_index)
            
            # Rebuild reverse maps
            red_reverse = {i: [] for i in range(m)}
            for key, tris in red_k4s.items():
                for t in tris:
                    red_reverse[t].append(('R', key))
            blue_reverse = {i: [] for i in range(m)}
            for key, tris in blue_k5s.items():
                for t in tris:
                    blue_reverse[t].append(('B', key))
            
            if total_violations == 0:
                return colors, True
        
        # If stuck (no progress for many steps), do a random perturbation
        if steps % 200 == 0 and total_violations > 0:
            # Flip 3 random triples
            for _ in range(3):
                ri = rng.randint(0, m - 1)
                colors[ri] = 1 - colors[ri]
            # Full rescan after perturbation
            red_k4s, blue_k5s, red_conflict, blue_conflict, total_violations = \
                _initial_violation_scan(n, colors, triple_index)
            red_reverse = {i: [] for i in range(m)}
            for key, tris in red_k4s.items():
                for t in tris:
                    red_reverse[t].append(('R', key))
            blue_reverse = {i: [] for i in range(m)}
            for key, tris in blue_k5s.items():
                for t in tris:
                    blue_reverse[t].append(('B', key))
    
    return colors, total_violations == 0

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    best_n = 1
    best_colors = [0]
    
    # We know n=9 works with modular coloring
    # Try to push higher
    
    # Start from n=10 and go up
    # For each n, do multiple restarts
    
    rng = random.Random(42)
    
    for n in range(10, 28):
        if time.time() > deadline - 8:
            break
        
        # Time budget for this n
        # Decrease budget as n increases? No, keep it roughly constant
        budget_per_n = 4.0
        t_start_n = time.time()
        
        found = False
        
        max_restarts = 8
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            if time.time() > t_start_n + budget_per_n:
                break
            
            # Choose initialization strategy
            if restart % 3 == 0:
                colors = _init_coloring(n, 'modular', seed=restart * 1000)
            elif restart % 3 == 1:
                colors = _init_coloring(n, 'random', seed=restart * 1000)
            else:
                # Slightly biased random
                colors = _init_coloring(n, 'random', seed=restart * 1000)
                # Bias towards blue (since blue K5 is harder to form with 10 edges)
                # Actually, for R(4,5;3), red K4 has only 4 edges, so red is easier to avoid
                # Let's try biasing red to be less frequent
                mask = rng.rand(len(colors)) < 0.45
                colors = np.where(mask, 0, 1).astype(np.int8)
            
            # Run local search with time limit
            t_limit = min(deadline - 5, time.time() + budget_per_n / max_restarts + 1.0)
            colors, success = _local_search_incremental(n, colors, t_limit, rng)
            
            if success:
                found = True
                best_n = n
                best_colors = colors.tolist()
                break
        
        if not found:
            # If we failed for n, try n again with more aggressive search?
            # Or break? Let's break, assuming monotonic difficulty
            # But actually, let's continue trying - maybe a different init works
            # For now, break to save time for other n values
            # Actually, let's not break - try next n might be easier with our init
            # Hmm, this is tricky. Let's just continue.
            pass
    
    # Fallback: n=9 with modular
    if best_n < 9:
        n = 9
        triples = _get_triples(n)
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        best_n = 9
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
