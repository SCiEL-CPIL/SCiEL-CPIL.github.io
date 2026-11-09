import time
import itertools
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_iso(p, a, b):
    dp_a = _dist2(p, a)
    dp_b = _dist2(p, b)
    da_b = _dist2(a, b)
    return dp_a == dp_b or dp_a == da_b or dp_b == da_b

def construct():
    start_time = time.time()
    deadline = start_time + 320
    
    points = list(itertools.product(range(N), repeat=2))
    
    W = [[] for _ in range(N * N)]
    
    # Precompute W and D
    # D[p] will be a list of q's that are ONLY blocked by pairs involving p
    # Actually, D[p] = { q | for all pairs in W[q], p is in the pair }
    # We can compute this after W is built.
    
    for q_idx in range(N * N):
        q = points[q_idx]
        for a_idx in range(N * N):
            if a_idx == q_idx:
                continue
            a = points[a_idx]
            for b_idx in range(a_idx + 1, N * N):
                if b_idx == q_idx:
                    continue
                b = points[b_idx]
                if _is_iso(q, a, b):
                    W[q_idx].append((a_idx, b_idx))
                    
    # Compute D
    D = [[] for _ in range(N * N)]
    for q_idx in range(N * N):
        if not W[q_idx]:
            # If no pairs block q, it's never blocked. 
            # But in a grid, this is unlikely for all q.
            # If W[q] is empty, q is never blocked.
            # This case doesn't affect D logic directly, but means q is always "unblocked" if not in S.
            continue
            
        # Check for each p if p is in ALL pairs in W[q]
        # This is O(|W[q]| * N)
        # Better: intersect the sets of points in each pair?
        # Pair is (a, b). Points involved are {a, b}.
        # Intersection of {a, b} over all pairs in W[q].
        # If the intersection is {p}, then q is only blocked by pairs involving p.
        # If the intersection is empty, no single p blocks it alone.
        
        # Let's just check for each p:
        # is p in all pairs?
        for p_idx in range(N * N):
            all_involved = True
            for a, b in W[q_idx]:
                if p_idx != a and p_idx != b:
                    all_involved = False
                    break
            if all_involved:
                D[p_idx].append(q_idx)
                
    def greedy_build(seed):
        random.seed(seed)
        order = list(range(N * N))
        random.shuffle(order)
        
        S_mask = 0
        S_list = []
        
        for p_idx in order:
            p = points[p_idx]
            can_add = True
            for i in range(len(S_list)):
                a_idx = S_list[i]
                a = points[a_idx]
                for j in range(i + 1, len(S_list)):
                    b_idx = S_list[j]
                    b = points[b_idx]
                    if _is_iso(p, a, b):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)
                
        return S_mask, S_list

    def is_maximal_after_removal(S_mask, removed_p):
        # Only check q in D[removed_p]
        for q_idx in D[removed_p]:
            if S_mask & (1 << q_idx):
                continue
            # Check if q_idx is blocked by S_mask
            blocked = False
            for a, b in W[q_idx]:
                if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    best_size = N * N + 1
    best_set = None
    
    num_trials = 1000
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        S_mask, S_list = greedy_build(trial)
        
        current_mask = S_mask
        current_list = S_list[:]
        
        changed = True
        while changed:
            changed = False
            random.shuffle(current_list)
            
            for p_idx in current_list:
                new_mask = current_mask & ~(1 << p_idx)
                if is_maximal_after_removal(new_mask, p_idx):
                    current_mask = new_mask
                    current_list.remove(p_idx)
                    changed = True
                    break
                    
        if len(current_list) < best_size:
            best_size = len(current_list)
            best_set = current_list[:]
            
            if best_size <= 8:
                break
                
    if best_set is None:
        S_mask, S_list = greedy_build(0)
        best_set = S_list
        
    return best_set

# EVOLVE-BLOCK-END
