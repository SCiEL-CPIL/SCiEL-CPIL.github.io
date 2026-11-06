import itertools
import time
import random

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute conflicts
# conflicts[i] is a list of tuples (j, k) such that POINTS[i], POINTS[j], POINTS[k] form an isosceles triangle
conflicts = [[] for _ in range(NUM_POINTS)]

def is_isosceles(p, a, b):
    d_pa = (p[0]-a[0])**2 + (p[1]-a[1])**2
    d_pb = (p[0]-b[0])**2 + (p[1]-b[1])**2
    d_ab = (a[0]-b[0])**2 + (a[1]-b[1])**2
    return d_pa == d_pb or d_pa == d_ab or d_pb == d_ab

# Build conflict graph
for i, p in enumerate(POINTS):
    for j in range(i + 1, NUM_POINTS):
        a = POINTS[j]
        for k in range(j + 1, NUM_POINTS):
            b = POINTS[k]
            if is_isosceles(p, a, b):
                conflicts[i].append((j, k))

def can_add(p_idx, S_set):
    for a, b in conflicts[p_idx]:
        if a in S_set and b in S_set:
            return False
    return True

def is_maximal(S_set):
    for p_idx in range(NUM_POINTS):
        if p_idx not in S_set:
            if can_add(p_idx, S_set):
                return False
    return True

def greedy_construct(order):
    S = []
    S_set = set()
    for p_idx in order:
        if can_add(p_idx, S_set):
            S.append(p_idx)
            S_set.add(p_idx)
    return S, S_set

def reduce_set(S, S_set):
    changed = True
    while changed:
        changed = False
        # Shuffle indices to try different removals first?
        # Or just iterate
        for i in range(len(S)):
            p_idx = S[i]
            new_S = S[:i] + S[i+1:]
            new_S_set = S_set.copy()
            new_S_set.remove(p_idx)
            
            if is_maximal(new_S_set):
                S = new_S
                S_set = new_S_set
                changed = True
                break
    return S, S_set

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    best_S = None
    best_size = float('inf')
    
    random.seed(42) # Fixed seed for consistency? Or random?
    # Let's use random seed based on time to explore different paths if run multiple times?
    # But evaluator runs once.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S, S_set = greedy_construct(order)
        S, S_set = reduce_set(S, S_set)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = S.copy()
            
        # SA Phase
        if time.time() + 10 < deadline:
            S_sa, S_set_sa = S.copy(), S_set.copy()
            T = 1.0
            sa_start = time.time()
            while time.time() - sa_start < 5 and time.time() < deadline:
                if not S_sa: break
                i = random.randint(0, len(S_sa) - 1)
                p_idx = S_sa[i]
                
                new_S = S_sa[:i] + S_sa[i+1:]
                new_S_set = S_set_sa.copy()
                new_S_set.remove(p_idx)
                
                if is_maximal(new_S_set):
                    S_sa = new_S
                    S_set_sa = new_S_set
                    T *= 0.9
                else:
                    candidates = []
                    for q_idx in range(NUM_POINTS):
                        if q_idx not in new_S_set:
                            if can_add(q_idx, new_S_set):
                                candidates.append(q_idx)
                    
                    if candidates:
                        q_idx = random.choice(candidates)
                        cand_S = new_S + [q_idx]
                        cand_S_set = new_S_set.copy()
                        cand_S_set.add(q_idx)
                        
                        if is_maximal(cand_S_set):
                            if random.random() < 0.5:
                                S_sa = cand_S
                                S_set_sa = cand_S_set
                            
                T *= 0.99
                
                if len(S_sa) < best_size:
                    best_size = len(S_sa)
                    best_S = S_sa.copy()

    if best_S is None:
        # Fallback: simple greedy
        S, _ = greedy_construct(list(range(NUM_POINTS)))
        S, _ = reduce_set(S, set(S))
        best_S = S
        
    return [POINTS[i] for i in best_S]
