import time
import random
import numpy as np

def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in s."""
    s = set(s)
    distinct_d = set()
    for a in s:
        for b in s:
            if b > a:
                d = b - a
                if a + 2 * d in s:
                    distinct_d.add(d)
    return len(distinct_d)

def _greedy_construct():
    """Greedy set-cover style construction."""
    n = 32
    universe = set(range(1, 513))
    # Start with a random seed element
    s = set()
    # Pick a starting element
    start = random.randint(1, 512)
    s.add(start)
    
    # Greedily add elements that maximize new distinct differences
    while len(s) < n:
        best_elem = None
        best_score = -1
        # Try all remaining elements
        for elem in universe - s:
            trial = s | {elem}
            # Quick score computation
            sc = _score_set(trial)
            if sc > best_score:
                best_score = sc
                best_elem = elem
        s.add(best_elem)
    
    return list(s)

def _swap_polish(s, max_iters=200):
    """Local search via swaps: replace one element with another to improve score."""
    s = set(s)
    current_score = _score_set(s)
    universe = set(range(1, 513))
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for a in list(s):
            for b in universe - s:
                trial = (s - {a}) | {b}
                sc = _score_set(trial)
                if sc > current_score:
                    s = trial
                    current_score = sc
                    improved = True
                    break
            if improved:
                break
    return list(s), current_score

def _simulated_annealing(s, max_iters=5000, t_start=1.0, t_end=0.01):
    """Simulated annealing with swap moves."""
    s = set(s)
    current_score = _score_set(s)
    best_s = set(s)
    best_score = current_score
    universe = set(range(1, 513))
    n = len(s)
    
    for it in range(max_iters):
        t = t_start * (t_end / t_start) ** (it / max_iters)
        # Pick a random element to remove and a random element to add
        a = random.choice(list(s))
        b = random.choice(list(universe - s))
        trial = (s - {a}) | {b}
        sc = _score_set(trial)
        delta = sc - current_score
        if delta >= 0 or random.random() < np.exp(delta / max(t, 1e-10)):
            s = trial
            current_score = sc
            if current_score > best_score:
                best_score = current_score
                best_s = set(s)
    
    return list(best_s), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_set = None
    best_score = 0
    
    # Strategy 1: Multiple greedy restarts with different seeds
    for restart in range(50):
        if time.time() > deadline:
            break
        random.seed()
        s = _greedy_construct()
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s
        # Polish
        s_polished, sc_polished = _swap_polish(s, max_iters=100)
        if sc_polished > best_score:
            best_score = sc_polished
            best_set = s_polished
    
    # Strategy 2: Simulated annealing from best found
    if time.time() < deadline:
        s_sa, sc_sa = _simulated_annealing(best_set, max_iters=3000)
        if sc_sa > best_score:
            best_score = sc_sa
            best_set = s_sa
        # Polish the SA result
        s_polished, sc_polished = _swap_polish(s_sa, max_iters=150)
        if sc_polished > best_score:
            best_score = sc_polished
            best_set = s_polished
    
    # Strategy 3: More greedy restarts with different initial structures
    for restart in range(30):
        if time.time() > deadline:
            break
        # Try different starting points
        s = set()
        # Start with a cluster
        base = random.randint(1, 100)
        for i in range(5):
            s.add(base + i * random.randint(1, 20))
        # Then greedy fill
        universe = set(range(1, 513))
        while len(s) < 32:
            best_elem = None
            best_score_local = -1
            for elem in universe - s:
                trial = s | {elem}
                sc = _score_set(trial)
                if sc > best_score_local:
                    best_score_local = sc
                    best_elem = elem
            s.add(best_elem)
        
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s
        s_polished, sc_polished = _swap_polish(s, max_iters=80)
        if sc_polished > best_score:
            best_score = sc_polished
            best_set = s_polished
    
    # Final polish
    if time.time() < deadline - 10:
        s_polished, sc_polished = _swap_polish(best_set, max_iters=300)
        if sc_polished > best_score:
            best_score = sc_polished
            best_set = s_polished
    
    return list(best_set)
