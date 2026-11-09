import time
import random
from collections import deque

N = 12
DEADLINE = 85

def _bfs_reset_length(trans_a, trans_b):
    full = frozenset(range(N))
    visited = {full: 0}
    queue = deque([(full, 0)])
    while queue:
        state, dist = queue.popleft()
        for trans in (trans_a, trans_b):
            new_state = frozenset(trans[s] for s in state)
            if len(new_state) == 1:
                return dist + 1
            if new_state not in visited:
                visited[new_state] = dist + 1
                queue.append((new_state, dist + 1))
    return -1

def construct():
    start_time = time.time()
    best_score = 0
    best_a = None
    best_b = None
    
    def try_auto(a, b):
        nonlocal best_score, best_a, best_b
        if time.time() - start_time > DEADLINE - 2:
            return
        s = _bfs_reset_length(a, b)
        if s > best_score:
            best_score = s
            best_a = a[:]
            best_b = b[:]
    
    # Classic constructions
    a_cycle = [(i + 1) % N for i in range(N)]
    a_rev = [(i - 1) % N for i in range(N)]
    
    # Various b maps
    b_decr = [0] + list(range(N - 1))       # 0->0, i->i-1
    b_incr = list(range(1, N)) + [N - 1]    # i->i+1, N-1->N-1
    b_half = [i // 2 for i in range(N)]     # fold
    b_const = [0] * N
    
    for a in (a_cycle, a_rev):
        for b in (b_decr, b_incr, b_half, b_const):
            try_auto(a, b)
    
    # Random search
    random.seed(12345)
    while time.time() - start_time < DEADLINE - 3:
        # Strategy 1: cycle + random map
        if random.random() < 0.4:
            a = a_cycle[:]
            b = [random.randint(0, N - 1) for _ in range(N)]
        # Strategy 2: random permutation + random map
        elif random.random() < 0.7:
            a = list(range(N))
            random.shuffle(a)
            b = [random.randint(0, N - 1) for _ in range(N)]
        # Strategy 3: two random maps
        else:
            a = [random.randint(0, N - 1) for _ in range(N)]
            b = [random.randint(0, N - 1) for _ in range(N)]
        
        try_auto(a, b)
    
    if best_a is None:
        best_a = a_cycle
        best_b = b_const
    
    return [best_a, best_b]
