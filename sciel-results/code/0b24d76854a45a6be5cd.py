import time
import random
import numpy as np
from itertools import combinations

def _edge_idx(i, j):
    """Map (i,j) with i<j to a linear index 0..119."""
    return i * 15 - i * (i + 1) // 2 + (j - i - 1)

def _k4_colours(colours, a, b, c, d):
    """Return the set of colours on the 6 edges of K4 {a,b,c,d}."""
    idx = [
        _edge_idx(a, b), _edge_idx(a, c), _edge_idx(a, d),
        _edge_idx(b, c), _edge_idx(b, d), _edge_idx(c, d)
    ]
    return set(colours[i] for i in idx)

def _count_bad_k4s(colours, n=16):
    """Count K4s with fewer than 5 distinct colours."""
    bad = 0
    for quad in combinations(range(n), 4):
        if len(_k4_colours(colours, *quad)) < 5:
            bad += 1
    return bad

def _count_colors_used(colours):
    return len(set(colours))

def _try_random_search(k_colors, n=16, max_iter=50000, seed=42):
    """Randomly assign colours and try to fix bad K4s."""
    rng = random.Random(seed)
    colors = [rng.randrange(k_colors) for _ in range(120)]
    
    for it in range(max_iter):
        bad = _count_bad_k4s(colors, n)
        if bad == 0:
            return colors
        # Pick a random bad K4 and change one of its edges
        quads = list(combinations(range(n), 4))
        rng.shuffle(quads)
        found_bad = False
        for quad in quads[:50]:  # Check a few
            if len(_k4_colours(colors, *quad)) < 5:
                a, b, c, d = quad
                edges = [
                    _edge_idx(a, b), _edge_idx(a, c), _edge_idx(a, d),
                    _edge_idx(b, c), _edge_idx(b, d), _edge_idx(c, d)
                ]
                ei = rng.choice(edges)
                colors[ei] = rng.randrange(k_colors)
                found_bad = True
                break
        if not found_bad:
            break
    
    if _count_bad_k4s(colors, n) == 0:
        return colors
    return None

def _try_structured_13():
    """Try a structured 13-colouring using modular arithmetic."""
    # Try various moduli and linear forms
    for p in [13, 14, 15, 16, 17, 19, 23]:
        for a in range(p):
            for b in range(p):
                if a == 0 and b == 0:
                    continue
                colors = [(a * i + b * j) % p for i in range(16) for j in range(i + 1, 16)]
                if _count_bad_k4s(colors) == 0 and _count_colors_used(colors) <= 13:
                    return colors
    return None

def _try_structured_14():
    """Try a structured 14-colouring."""
    for p in [13, 14, 15, 16, 17, 19, 23]:
        for a in range(p):
            for b in range(p):
                if a == 0 and b == 0:
                    continue
                colors = [(a * i + b * j) % p for i in range(16) for j in range(i + 1, 16)]
                if _count_bad_k4s(colors) == 0 and _count_colors_used(colors) <= 14:
                    return colors
    return None

def _try_structured_15():
    """Try a structured 15-colouring."""
    for p in [13, 14, 15, 16, 17, 19, 23]:
        for a in range(p):
            for b in range(p):
                if a == 0 and b == 0:
                    continue
                colors = [(a * i + b * j) % p for i in range(16) for j in range(i + 1, 16)]
                if _count_bad_k4s(colors) == 0 and _count_colors_used(colors) <= 15:
                    return colors
    return None

def _try_local_search(k_colors, n=16, max_iter=20000, seed=42):
    """Start from a random assignment and do local search."""
    rng = random.Random(seed)
    colors = [rng.randrange(k_colors) for _ in range(120)]
    
    best_bad = _count_bad_k4s(colors, n)
    
    for it in range(max_iter):
        if best_bad == 0:
            return colors
        
        # Find a bad K4
        quads = list(combinations(range(n), 4))
        rng.shuffle(quads)
        bad_quad = None
        for quad in quads:
            if len(_k4_colours(colors, *quad)) < 5:
                bad_quad = quad
                break
        
        if bad_quad is None:
            return colors
        
        a, b, c, d = bad_quad
        edges = [
            _edge_idx(a, b), _edge_idx(a, c), _edge_idx(a, d),
            _edge_idx(b, c), _edge_idx(b, d), _edge_idx(c, d)
        ]
        
        # Try changing each edge to each colour
        best_change = None
        best_bad_after = best_bad
        for ei in edges:
            old = colors[ei]
            for new_c in range(k_colors):
                if new_c == old:
                    continue
                colors[ei] = new_c
                new_bad = _count_bad_k4s(colors, n)
                if new_bad < best_bad_after:
                    best_bad_after = new_bad
                    best_change = (ei, new_c)
                colors[ei] = old
            
            if best_change is not None and best_change[0] == ei:
                break
        
        if best_change is not None:
            ei, new_c = best_change
            colors[ei] = new_c
            best_bad = best_bad_after
        else:
            # Random restart
            colors = [rng.randrange(k_colors) for _ in range(120)]
            best_bad = _count_bad_k4s(colors, n)
    
    if _count_bad_k4s(colors, n) == 0:
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try the known 17-colour solution as fallback
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    best_result = fallback
    best_colors = 17
    
    # Try to find 13-colour solutions
    if time.time() < deadline:
        result = _try_structured_13()
        if result is not None:
            return result
        best_result = result if result is not None else fallback
    
    # Try random search for 13 colours
    if time.time() < deadline:
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _try_random_search(13, max_iter=10000, seed=seed)
            if result is not None:
                return result
    
    # Try 14-colour solutions
    if time.time() < deadline:
        result = _try_structured_14()
        if result is not None:
            return result
    
    if time.time() < deadline:
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _try_random_search(14, max_iter=10000, seed=seed)
            if result is not None:
                return result
    
    # Try 15-colour solutions
    if time.time() < deadline:
        result = _try_structured_15()
        if result is not None:
            return result
    
    if time.time() < deadline:
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _try_random_search(15, max_iter=10000, seed=seed)
            if result is not None:
                return result
    
    # Return the best we found (fallback is always valid)
    return fallback
