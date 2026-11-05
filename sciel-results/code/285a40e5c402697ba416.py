"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
    s_set = set(s)
    diffs = set()
    for a in s:
        for d in range(1, 256):
            if a + d in s_set and a + 2*d in s_set:
                diffs.add(d)
                # No need to break; we want all distinct d
    return len(diffs)

def _score_set_fast(s):
    """Faster scoring using sorted array and set lookup."""
    s_set = set(s)
    diffs = set()
    for a in s:
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if a + d in s_set and a + 2*d in s_set:
                diffs.add(d)
    return len(diffs)

def _greedy_polish(s, max_iters=200):
    """Local search: try swapping elements to improve score."""
    s = list(s)
    s_set = set(s)
    current = _score_set_fast(s)
    used = set(s)
    
    for _ in range(max_iters):
        improved = False
        random.shuffle(s)
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                # Try replacing s[i] with some new value
                # Try small perturbations
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -7, 7, -10, 10]:
                    new_val = s[i] + delta
                    if 1 <= new_val <= 512 and new_val not in used:
                        s[i] = new_val
                        used.remove(s[i] - delta)
                        used.add(new_val)
                        new_score = _score_set_fast(s)
                        if new_score > current:
                            current = new_score
                            improved = True
                        else:
                            s[i] = s[i] - delta
                            used.remove(new_val)
                            used.add(s[i])
                        if improved:
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return s, current

def _simulated_annealing(s, time_limit=10.0):
    """SA with deadline guard."""
    s = list(s)
    s_set = set(s)
    current = _score_set_fast(s)
    best = list(s)
    best_score = current
    t0 = time.time()
    
    T = 50.0
    cooling = 0.995
    
    while time.time() - t0 < time_limit:
        if T < 0.01:
            break
        # Move: pick random element, change by random amount
        i = random.randint(0, len(s)-1)
        old_val = s[i]
        new_val = old_val + random.randint(-15, 15)
        if new_val < 1 or new_val > 512 or new_val in s_set:
            T *= cooling
            continue
        
        s[i] = new_val
        s_set.remove(old_val)
        s_set.add(new_val)
        new_score = _score_set_fast(s)
        
        if new_score >= current or random.random() < np.exp((new_score - current) / T):
            current = new_score
            if current > best_score:
                best_score = current
                best = list(s)
        else:
            s[i] = old_val
            s_set.remove(new_val)
            s_set.add(old_val)
        
        T *= cooling
    
    return best, best_score

def _structured_init(seed_val=0):
    """Generate a structured initial set."""
    rng = random.Random(seed_val)
    # Use a geometric-like spread
    base = rng.choice([3, 4, 5, 6, 7, 8, 10])
    s = set()
    # Add a geometric progression
    x = rng.randint(1, 20)
    for _ in range(16):
        if x <= 512:
            s.add(x)
        x = int(x * base / (base - 1)) + rng.randint(-2, 2)
        x = max(1, min(512, x))
    
    # Fill remaining with random
    while len(s) < 32:
        v = rng.randint(1, 512)
        s.add(v)
    
    return list(s)[:32]

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    best_set = list(range(1, 33))
    best_score = _score_set_fast(best_set)
    
    # Multi-restart SA with different initializations
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if r == 0:
            init = list(range(1, 33))
        elif r == 1:
            # Geometric spread
            init = [int(512 * (i+1) / 33) for i in range(32)]
            init = list(set(init))
            while len(init) < 32:
                init.append(random.randint(1, 512))
            init = list(set(init))[:32]
        elif r == 2:
            # Two clusters
            init = list(range(1, 17)) + list(range(400, 416))
        elif r == 3:
            # Random
            init = random.sample(range(1, 513), 32)
        elif r == 4:
            init = _structured_init(r)
        elif r == 5:
            # Arithmetic with large step
            init = [1 + i * 16 for i in range(32)]
            init = [x for x in init if x <= 512]
            while len(init) < 32:
                init.append(random.randint(1, 512))
            init = list(set(init))[:32]
        elif r == 6:
            # Mix of small and large
            init = list(range(1, 13)) + list(range(480, 492)) + [random.randint(1, 512) for _ in range(8)]
            init = list(set(init))[:32]
        elif r == 7:
            # Power of 2 based
            init = [2**i for i in range(0, 9)]
            init += [2**i * 3 for i in range(0, 8)]
            init += [2**i * 5 for i in range(0, 7)]
            init += [random.randint(1, 512) for _ in range(32 - len(init))]
            init = list(set(x for x in init if 1 <= x <= 512))[:32]
        elif r == 8:
            # Fibonacci-like
            init = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
            init += [random.randint(1, 512) for _ in range(32 - len(init))]
            init = list(set(x for x in init if 1 <= x <= 512))[:32]
        elif r == 9:
            # Dense small + sparse large
            init = list(range(1, 21))
            init += [random.randint(250, 512) for _ in range(12)]
            init = list(set(init))[:32]
        elif r == 10:
            # Quadratic
            init = [i*i + i + 1 for i in range(22)]
            init = [x for x in init if x <= 512]
            init += [random.randint(1, 512) for _ in range(32 - len(init))]
            init = list(set(init))[:32]
        else:
            init = random.sample(range(1, 513), 32)
        
        # Ensure exactly 32 distinct
        init = list(set(init))
        while len(init) < 32:
            v = random.randint(1, 512)
            if v not in init:
                init.append(v)
        init = init[:32]
        
        # SA phase
        sa_time = min(8.0, max(2.0, (deadline - time.time()) / (n_restarts - r)))
        sa_result, sa_score = _simulated_annealing(init, time_limit=sa_time)
        
        # Greedy polish
        if time.time() < deadline - 5:
            polished, pol_score = _greedy_polish(sa_result, max_iters=100)
            if pol_score > sa_score:
                sa_result, sa_score = polished, pol_score
        
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_result
    
    return best_set
# EVOLVE-BLOCK-END
