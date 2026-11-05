"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses targeted search focusing on Černý-bound structures with fast BFS evaluation.
"""
import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _threshold_fast(a, b):
    """
    Compute the exact reset threshold using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    Uses list-based visited array for speed.
    """
    # Pre-compute transitions for all 2^N masks
    num_masks = 1 << N
    trans_a = [0] * num_masks
    trans_b = [0] * num_masks
    
    for mask in range(num_masks):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= 1 << a[i]
                mb |= 1 << b[i]
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    # List-based visited: -1 = unvisited, else depth
    visited = [-1] * num_masks
    visited[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    
    while queue:
        mask = queue.popleft()
        depth = visited[mask]
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
        
        # Apply 'a'
        next_a = trans_a[mask]
        if visited[next_a] == -1:
            visited[next_a] = depth + 1
            queue.append(next_a)
            
        # Apply 'b'
        next_b = trans_b[mask]
        if visited[next_b] == -1:
            visited[next_b] = depth + 1
            queue.append(next_b)
            
    return -1

def _generate_cherny_like_structures():
    """
    Generate automata with Černý-bound-like structure.
    One letter forms a cycle, the other has one 'defect' state.
    """
    structures = []
    
    # Try all permutations for the cycle letter
    # and all positions for the defect in the other letter
    for cycle_start in range(N):
        # Letter 'a' is the cycle: a[i] = (i + 1) % N
        a = [(i + 1) % N for i in range(N)]
        
        # Letter 'b' has one defect at position 'defect_pos'
        for defect_pos in range(N):
            b = [(i + 1) % N for i in range(N)]
            # The defect: b[defect_pos] goes to some state
            # Try different targets for the defect
            for defect_target in range(N):
                b_test = b[:]
                b_test[defect_pos] = defect_target
                structures.append((a[:], b_test))
    
    # Also try where 'b' is the cycle and 'a' has the defect
    for cycle_start in range(N):
        b = [(i + 1) % N for i in range(N)]
        
        for defect_pos in range(N):
            a = [(i + 1) % N for i in range(N)]
            for defect_target in range(N):
                a_test = a[:]
                a_test[defect_pos] = defect_target
                structures.append((a_test, b[:]))
    
    return structures

def _local_search(a, b, time_budget):
    """
    Steepest ascent local search: explore all single-transition changes.
    """
    start_time = time.time()
    current_score = _threshold_fast(a, b)
    best_a = a[:]
    best_b = b[:]
    best_score = current_score
    
    if current_score < 0:
        return best_a, best_b, best_score
    
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        # Try changing each transition in 'a'
        for pos in range(N):
            old_val = a[pos]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                a[pos] = new_val
                score = _threshold_fast(a, b)
                if score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
                    improved = True
                a[pos] = old_val
        
        # Try changing each transition in 'b'
        for pos in range(N):
            old_val = b[pos]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                b[pos] = new_val
                score = _threshold_fast(a, b)
                if score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
                    improved = True
                b[pos] = old_val
        
        # Update current to best found
        a = best_a[:]
        b = best_b[:]
        current_score = best_score
    
    return best_a, best_b, best_score

def _random_automaton():
    """Generate a random automaton."""
    a = [random.randint(0, N-1) for _ in range(N)]
    b = [random.randint(0, N-1) for _ in range(N)]
    return a, b

def construct():
    """
    Construct a 12-state binary automaton maximizing reset threshold.
    Uses targeted search on Černý-like structures with local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Phase 1: Evaluate all Černý-like structures
    cherny_structures = _generate_cherny_like_structures()
    random.shuffle(cherny_structures)
    
    for a, b in cherny_structures:
        if time.time() > deadline:
            break
        score = _threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Phase 2: Local search from best found so far
    if best_a is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            # Do local search with time budget
            local_time_budget = min(remaining_time - 2.0, 40.0)
            best_a, best_b, best_score = _local_search(best_a, best_b, local_time_budget)
    
    # Phase 3: If still time, try random restarts with local search
    if best_a is None:
        best_a, best_b = _random_automaton()
        best_score = _threshold_fast(best_a, best_b)
    
    while time.time() < deadline - 5.0:
        # Generate a random starting point
        a, b = _random_automaton()
        score = _threshold_fast(a, b)
        
        if score < 0:
            continue
        
        # Do a quick local search
        quick_budget = min(2.0, deadline - time.time() - 3.0)
        if quick_budget > 0.5:
            a, b, score = _local_search(a, b, quick_budget)
        
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Ensure we have a valid result
    if best_a is None or best_score < 0:
        # Fallback: simple known structure
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] + [(i + 1) % N for i in range(1, N)]
        best_score = _threshold_fast(best_a, best_b)
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
