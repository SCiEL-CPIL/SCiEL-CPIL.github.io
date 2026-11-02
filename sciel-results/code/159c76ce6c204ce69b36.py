import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)"""
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    l_inf = np.max(h)
    if l1 < 1e-15 or l_inf < 1e-15:
        return 0.0
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return (l2_sq) / (l1 * l_inf * max_conv)

def _find_sidon_set(deadline):
    """
    Find a maximal Sidon set in {0, ..., N-1} using greedy strategy with random restarts.
    A Sidon set has distinct pairwise differences.
    """
    best_set = []
    best_size = 0
    
    # Deterministic greedy first
    def greedy_sidon(order):
        used_diffs = set()
        current_set = []
        for i in order:
            can_add = True
            for j in current_set:
                diff = abs(i - j)
                if diff in used_diffs:
                    can_add = False
                    break
            if can_add:
                current_set.append(i)
                for j in current_set[:-1]:
                    used_diffs.add(abs(i - j))
        return current_set

    # 1. Standard order
    s1 = greedy_sidon(range(N))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1

    # 2. Reverse order
    s2 = greedy_sidon(range(N-1, -1, -1))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    # 3. Random restarts
    rng = np.random.RandomState(42)
    while time.time() < deadline - 2:
        order = list(range(N))
        rng.shuffle(order)
        s = greedy_sidon(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # If we hit the theoretical limit (~sqrt(N)), we can stop early
            if best_size >= 8: 
                break

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Find the best Sidon set
    sidon_set = _find_sidon_set(deadline)
    
    # Build the height vector
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0
        
    # Fallback if empty (should not happen)
    if np.max(h) < 1e-15:
        h[0] = 1.0
        
    return list(h)
