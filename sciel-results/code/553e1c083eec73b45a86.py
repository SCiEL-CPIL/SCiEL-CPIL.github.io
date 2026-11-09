"""Snake-in-the-box in Q_9: randomized multi-start greedy with diversity."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N
DEADLINE = 145.0

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _randomized_greedy(start, rng, deadline, bias=0.7):
    """
    Greedy extension where at each step, we pick a valid next vertex.
    With probability `bias`, pick the first valid one; otherwise pick a random valid one.
    """
    path = [start]
    used = {start}
    
    while time.time() < deadline:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if _hamming_dist(v, p) == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            break
        
        if len(candidates) == 1 or rng.random() < bias:
            v = candidates[0]
        else:
            v = rng.choice(candidates)
        
        path.append(v)
        used.add(v)
    
    return path

def _greedy_with_bit_order(start, bit_order, deadline):
    """Greedy with a specific bit ordering."""
    path = [start]
    used = {start}
    
    while time.time() < deadline:
        cur = path[-1]
        found = False
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if _hamming_dist(v, p) == 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_path = None
    best_score = 0
    
    # Phase 1: Deterministic greedy with various bit orders
    bit_orders = [list(range(N)), list(range(N-1, -1, -1))]
    rng_init = random.Random(0)
    for _ in range(30):
        order = list(range(N))
        rng_init.shuffle(order)
        bit_orders.append(order)
    
    for bo in bit_orders:
        if time.time() >= deadline:
            break
        path = _greedy_with_bit_order(0, bo, deadline)
        score = len(path) - 1
        if score > best_score:
            best_score = score
            best_path = list(path)
    
    # Phase 2: Randomized greedy from start=0
    rng = random.Random(1)
    for _ in range(200):
        if time.time() >= deadline:
            break
        bias = 0.5 + rng.random() * 0.5  # bias between 0.5 and 1.0
        path = _randomized_greedy(0, rng, deadline, bias=bias)
        score = len(path) - 1
        if score > best_score:
            best_score = score
            best_path = list(path)
    
    # Phase 3: Multi-start randomized greedy
    rng2 = random.Random(999)
    for _ in range(300):
        if time.time() >= deadline:
            break
        start = rng2.randint(0, TOTAL_VERTICES - 1)
        bias = 0.5 + rng2.random() * 0.5
        path = _randomized_greedy(start, rng2, deadline, bias=bias)
        score = len(path) - 1
        if score > best_score:
            best_score = score
            best_path = list(path)
    
    # Phase 4: Multi-start deterministic greedy with random bit orders
    rng3 = random.Random(777)
    for _ in range(200):
        if time.time() >= deadline:
            break
        start = rng3.randint(0, TOTAL_VERTICES - 1)
        bo = list(range(N))
        rng3.shuffle(bo)
        path = _greedy_with_bit_order(start, bo, deadline)
        score = len(path) - 1
        if score > best_score:
            best_score = score
            best_path = list(path)
    
    if best_path is None:
        # Fallback
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if _hamming_dist(v, p) == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path
    
    return best_path
