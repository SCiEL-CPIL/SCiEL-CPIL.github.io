"""Construct a 12-state synchronizing automaton with a long reset threshold."""
import time
from collections import deque

N = 12

def _compute_reset_length(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    visited = {full_mask}
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        new_mask_a = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
            
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
            
    return -1

def construct():
    deadline = time.time() + 80.0  # Safety margin
    best_a = None
    best_b = None
    best_len = -1
    
    # Standard cycle for 'a'
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Systematic enumeration of 'b' maps
    # We explore a structured space of 'b' maps that are likely to produce long reset words.
    # Structure: b(0) is fixed to 0. For i > 0, b(i) is chosen from a small set of candidates.
    # Candidates for b(i): {0, i-1, i} (Contract, Stay, Reset)
    # This space is 3^11 = 177,147.
    # With optimized BFS, this is feasible within 80s if we prune or if BFS is very fast.
    # To be safe, we first search a smaller subset: {0, i-1} (2^11 = 2048).
    # If time permits, we expand to {0, i-1, i}.
    
    def try_candidate(b_map):
        nonlocal best_a, best_b, best_len
        if time.time() > deadline:
            return
        # Quick check: b must not be a permutation (if it is, it can't sync unless a helps, 
        # but with a being a cycle, b must reduce rank)
        # Actually, just run BFS. It's fast enough for 2048 cases.
        length = _compute_reset_length(a_cycle, b_map)
        if length > best_len:
            best_a = list(a_cycle)
            best_b = list(b_map)
            best_len = length

    # Phase 1: Search {0, i-1} for i>0
    # b(0) = 0
    # For i in 1..11, b(i) is 0 or i-1
    for mask in range(1 << (N - 1)):
        if time.time() > deadline:
            break
        b_map = [0] * N
        for i in range(1, N):
            # bit (i-1) determines if b(i) is 0 or i-1
            # If bit is 0, b(i) = i-1 (stay/shift)
            # If bit is 1, b(i) = 0 (reset)
            if (mask >> (i - 1)) & 1:
                b_map[i] = 0
            else:
                b_map[i] = i - 1
        try_candidate(b_map)
        
    # Phase 2: If time remains, expand search to include 'i' (stay)
    # Candidates for b(i) in {0, i-1, i}
    # This is 3^11. We can iterate using base-3 counting or recursion.
    # Given the 80s limit and Phase 1 taking ~1-2s, we have plenty of time for Phase 2.
    
    def search_phase2(prefix, depth):
        nonlocal best_a, best_b, best_len
        if time.time() > deadline:
            return
        if depth == N:
            try_candidate(prefix)
            return
        
        # Options for b(depth): 0, depth-1, depth
        # Note: for depth=0, we fixed it to 0 in the loop setup, but here we handle generically
        # Actually, let's just iterate all 3^11 combinations for i=1..11, fixing b(0)=0.
        
    # Iterative base-3 enumeration for b(1)..b(11)
    total_combinations = 3 ** (N - 1)
    # To avoid 177k full BFS if they are slow, we might need to be careful.
    # But 177k * ~0.5ms = ~90s. This is tight.
    # Let's restrict Phase 2 to a heuristic subset or stop if time is low.
    
    # Alternative Phase 2: Local search around the best found in Phase 1.
    if best_b and time.time() < deadline - 10.0:
        current_b = list(best_b)
        current_len = best_len
        # Try flipping one bit in the "0 vs i-1" choice, or adding "i"
        # We'll do a simple hill climbing: try all single-position changes from {0, i-1, i}
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(1, N):
                if time.time() > deadline - 5.0:
                    break
                for val in [0, i - 1, i]:
                    if val == current_b[i]:
                        continue
                    new_b = list(current_b)
                    new_b[i] = val
                    length = _compute_reset_length(a_cycle, new_b)
                    if length > current_len:
                        current_b = new_b
                        current_len = length
                        improved = True
                        if current_len > best_len:
                            best_a = list(a_cycle)
                            best_b = list(current_b)
                            best_len = current_len
                            improved = True
                            break # Restart loop with new best

    # Fallback if no synchronizing automaton found (should not happen)
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
        best_b[1] = 0 # Ensure it's not identity/permutation if possible

    return [best_a, best_b]
