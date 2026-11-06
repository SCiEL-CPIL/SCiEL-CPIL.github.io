N = 9
FULL = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

import time
import random

# EVOLVE-BLOCK-START
def _neighbors(v):
    return [v ^ m for m in BIT_MASKS]

def _backtrack_optimized(start, deadline, rng, bit_order):
    best = [start]
    best_len = 1
    
    path = [start]
    used = {start}
    blocked = set()
    
    def extend():
        nonlocal best, best_len
        if time.time() > deadline:
            return
        cur = path[-1]
        
        candidates = []
        for b in bit_order:
            v = cur ^ BIT_MASKS[b]
            if v in used or v in blocked:
                continue
            candidates.append(v)
        
        if not candidates:
            if len(path) > best_len:
                best = path[:]
                best_len = len(path)
            return
        
        scored = []
        for v in candidates:
            cnt = 0
            for b2 in bit_order:
                w = v ^ BIT_MASKS[b2]
                if w in used:
                    continue
                if w in blocked:
                    continue
                if (w ^ cur).bit_count() == 1:
                    continue
                cnt += 1
            scored.append((cnt, v))
        
        scored.sort(key=lambda x: x[0])
        
        for cnt, v in scored:
            if time.time() > deadline:
                return
            path.append(v)
            used.add(v)
            for nb in _neighbors(cur):
                blocked.add(nb)
            
            extend()
            
            for nb in _neighbors(cur):
                blocked.discard(nb)
            used.discard(v)
            path.pop()
            
            if len(path) > best_len:
                best = path[:]
                best_len = len(path)
    
    extend()
    return best

def _greedy_optimized(start, bit_order):
    path = [start]
    used = {start}
    blocked = set()
    
    while True:
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ BIT_MASKS[b]
            if v in used or v in blocked:
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        for nb in _neighbors(cur):
            blocked.add(nb)
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    best_path = None
    best_len = 0
    rng = random.Random(12345)
    
    for trial in range(300):
        if time.time() > deadline:
            break
        order = list(range(N))
        rng.shuffle(order)
        start = rng.randint(0, FULL - 1)
        path = _greedy_optimized(start, order)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    for trial in range(300):
        if time.time() > deadline:
            break
        order = list(range(N))
        rng.shuffle(order)
        start = rng.randint(0, FULL - 1)
        sub_deadline = min(deadline, time.time() + 1.0)
        result = _backtrack_optimized(start, sub_deadline, rng, order)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result[:]
    
    if best_path is None:
        return [0]
    return best_path
# EVOLVE-BLOCK-END
