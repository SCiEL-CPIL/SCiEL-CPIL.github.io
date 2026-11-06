import time
import random
from collections import deque

N = 12

def _threshold(trans_a, trans_b):
    full = (1 << N) - 1
    if full & (full - 1) == 0:
        return 0
    visited = {full}
    queue = deque([(full, 0)])
    while queue:
        s, d = queue.popleft()
        sa = 0
        sb = 0
        for i in range(N):
            bit = 1 << i
            if s & bit:
                sa |= (1 << trans_a[i])
                sb |= (1 << trans_b[i])
        for ns in (sa, sb):
            if ns & (ns - 1) == 0:
                return d + 1
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, d + 1))
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    random.seed(12345)
    
    best_a = None
    best_b = None
    best_thresh = -1
    
    # Find initial synchronizing automaton
    for _ in range(2000):
        if time.time() >= deadline:
            break
        a = [random.randrange(N) for _ in range(N)]
        b = [random.randrange(N) for _ in range(N)]
        t = _threshold(a, b)
        if t != -1:
            best_a = a
            best_b = b
            best_thresh = t
            break
    
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Hill-climbing with occasional restarts
    while time.time() < deadline:
        # Hill-climb from current best
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(50):  # 50 mutations per round
                if time.time() >= deadline:
                    break
                letter = random.random() < 0.5
                state = random.randrange(N)
                new_dest = random.randrange(N)
                
                if letter:
                    ta = best_a[:]
                    tb = best_b[:]
                    ta[state] = new_dest
                else:
                    ta = best_a[:]
                    tb = best_b[:]
                    tb[state] = new_dest
                
                t = _threshold(ta, tb)
                if t != -1 and t > best_thresh:
                    best_a = ta
                    best_b = tb
                    best_thresh = t
                    improved = True
        
        # Occasional restart: try a new random automaton
        if random.random() < 0.1 and time.time() < deadline:
            for _ in range(100):
                if time.time() >= deadline:
                    break
                a = [random.randrange(N) for _ in range(N)]
                b = [random.randrange(N) for _ in range(N)]
                t = _threshold(a, b)
                if t != -1 and t > best_thresh:
                    best_a = a
                    best_b = b
                    best_thresh = t
    
    return [best_a, best_b]
