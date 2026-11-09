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
    n_pts = N * N
    
    W = [[] for _ in range(n_pts)]
    
    # Precompute W
    for q_idx in range(n_pts):
        q = points[q_idx]
        for a_idx in range(n_pts):
            if a_idx == q_idx:
                continue
            a = points[a_idx]
            for b_idx in range(a_idx + 1, n_pts):
                if b_idx == q_idx:
                    continue
                b = points[b_idx]
                if _is_iso(q, a, b):
                    W[q_idx].append((a_idx, b_idx))
                    
    # Compute InvW
    InvW = [[] for _ in range(n_pts)]
    for q_idx in range(n_pts):
        for a, b in W[q_idx]:
            InvW[a].append(q_idx)
            InvW[b].append(q_idx)
    for p in range(n_pts):
        InvW[p] = list(set(InvW[p]))

    def is_point_addable(p_idx, S_mask):
        """Check if p_idx can be added to S_mask without forming an isosceles triangle."""
        p = points[p_idx]
        # Check against all pairs in S
        # Get list of indices in S
        # Optimization: iterate over bits?
        # S_mask is an int.
        # We can iterate over set bits.
        s_indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_indices.append(idx)
            m ^= lsb
        
        for i in range(len(s_indices)):
            a_idx = s_indices[i]
            a = points[a_idx]
            for j in range(i + 1, len(s_indices)):
                b_idx = s_indices[j]
                b = points[b_idx]
                if _is_iso(p, a, b):
                    return False
        return True

    def make_maximal(S_list):
        """Greedy add points to make S_list maximal."""
        S_mask = 0
        for idx in S_list:
            S_mask |= (1 << idx)
        
        added = True
        while added:
            added = False
            for p_idx in range(n_pts):
                if S_mask & (1 << p_idx):
                    continue
                if is_point_addable(p_idx, S_mask):
                    S_mask |= (1 << p_idx)
                    added = True
        # Convert back to list
        return [idx for idx in range(n_pts) if S_mask & (1 << idx)]

    def greedy_build(seed):
        random.seed(seed)
        order = list(range(n_pts))
        random.shuffle(order)
        
        S_mask = 0
        S_list = []
        
        for p_idx in order:
            if is_point_addable(p_idx, S_mask):
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)
                
        return S_mask, S_list

    def is_maximal_after_removal(S_mask, removed_p):
        for q_idx in InvW[removed_p]:
            if S_mask & (1 << q_idx):
                continue
            blocked = False
            for a, b in W[q_idx]:
                if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    best_size = n_pts + 1
    best_set = None
    
    num_trials = 5000
    for trial in range(num_trials):
        if time.time() > deadline - 10: # Safety margin
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
            
    if best_set is None:
        S_mask, S_list = greedy_build(0)
        best_set = S_list

    # CRITICAL FIX: Ensure the returned set is maximal
    # The local search might have a bug or edge case.
    # Repair by adding any missing points.
    best_set = make_maximal(best_set)

    return [points[idx] for idx in best_set]

# EVOLVE-BLOCK-END
