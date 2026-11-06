import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Optimized with precomputed transition tables for all 2^N subsets.
    Returns -1 if not synchronizing.
    """
    # Precompute single-state transitions as bitmasks
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # Precompute next_subset for all possible subsets and both letters
    # next_subsets[s, 0] is the subset reached from s by letter 0
    # next_subsets[s, 1] is the subset reached from s by letter 1
    next_subsets = np.zeros((1 << N, 2), dtype=np.int32)
    
    for s in range(1 << N):
        res0 = 0
        res1 = 0
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res0 |= t0[idx]
            res1 |= t1[idx]
            temp_s ^= lsb
        next_subsets[s, 0] = res0
        next_subsets[s, 1] = res1

    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if current subset is a singleton (reset state)
        if current != 0 and (current & (current - 1)) == 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current, letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _pinwheel_base():
    """
    Generates the standard Pinwheel automaton:
    Letter 0: i -> (i+1) % N
    Letter 1: i -> (i-1) % N
    This is not synchronizing (it's a group), so we need to break it.
    """
    trans0 = [(i + 1) % N for i in range(N)]
    trans1 = [(i - 1) % N for i in range(N)]
    return trans0, trans1

def _create_defect(pin0, pin1, rng):
    """
    Creates a synchronizing automaton by introducing a 'defect' into the pinwheel.
    Common high-threshold structures involve making one letter a 'contraction' 
    (many-to-one) while the other is a 'shift' (permutation).
    """
    t0 = pin0[:]
    t1 = pin1[:]
    
    # Strategy: Make letter 1 a contraction.
    # Keep t0 as the shift (permutation).
    # Modify t1 so that it maps most states to i-1, but some to a 'sink' or similar.
    
    # A known good structure for high thresholds:
    # t0: i -> i+1
    # t1: i -> i-1 for i>0, t1[0] = N-1 (This is just the reverse cycle, not sync)
    # We need a contraction.
    # Let's try: t1[i] = i-1 for i in 1..N-1, and t1[0] = 0 (Sink at 0).
    # This is the standard Cerny automaton (threshold N-1).
    # To get higher, we need to delay the contraction.
    
    # "Delayed Contraction" strategy:
    # t0: i -> i+1
    # t1: i -> i-1 for i > k, t1[i] = i for i <= k (or similar)
    # Actually, let's use the "Pinwheel with a hole" or "Near-Permutation" approach.
    
    # Let's pick a random state 'defect_state' where t1 stops being a shift.
    # Instead of random, let's try a few specific structural patterns.
    
    pattern = rng.randint(0, 2)
    
    if pattern == 0:
        # Standard Cerny: t1 is a chain to 0
        for i in range(1, N):
            t1[i] = i - 1
        t1[0] = 0
    elif pattern == 1:
        # "Two-cycle" defect: t1 reverses direction but has a sink
        # t1[i] = i-1, t1[0] = N-1 (Cycle) -> Not sync.
        # t1[i] = i-1, t1[0] = 0 (Sink) -> Cerny.
        # Let's try a sink in the middle.
        sink = rng.randint(0, N-1)
        for i in range(N):
            t1[i] = (i - 1) % N
        t1[sink] = sink # Sink
        # This might not sync if the cycle part is closed. 
        # If t1 has a cycle not containing the sink, it won't sync.
        # So t1 must be a functional graph with a single sink and no other cycles.
        # A simple way: t1[i] = i-1 for i>0, t1[0]=0 is safe.
        # Let's try: t1[i] = i-1 for i > k, t1[i] = 0 for i <= k.
        k = rng.randint(0, N-1)
        for i in range(N):
            if i > k:
                t1[i] = i - 1
            else:
                t1[i] = 0
    else:
        # Random contraction preserving connectivity
        # Start with reverse shift
        t1 = [(i - 1) % N for i in range(N)]
        # Break the cycle by pointing one state to a neighbor
        break_idx = rng.randint(0, N-1)
        t1[break_idx] = rng.randint(0, N-1)
        # Ensure no other cycles? Hard to check quickly. 
        # If it doesn't sync, _compute_threshold will return -1.

    return t0, t1

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin for 90s limit
    
    rng = random.Random(42)
    
    # 1. Initialize with a good seed
    pin0, pin1 = _pinwheel_base()
    best0, best1 = _create_defect(pin0, pin1, rng)
    best_score = _compute_threshold(best0, best1)
    
    if best_score < 0:
        # Fallback to known Cerny
        best0 = [(i + 1) % N for i in range(N)]
        best1 = [0] * N
        for i in range(1, N):
            best1[i] = i - 1
        best_score = _compute_threshold(best0, best1)
        
    current0 = best0[:]
    current1 = best1[:]
    current_score = best_score
    
    # 2. Local Search (Hill Climbing)
    # We will perform "Single Edge Flips".
    # To speed up, we can batch evaluations or use a smarter heuristic.
    # However, _compute_threshold is O(2^N * N) ~ 40k ops. 
    # 85s / 40k ops is plenty for thousands of evaluations.
    # Let's do a steepest-ascent hill climb with random restarts if stuck.
    
    no_improvement_count = 0
    max_no_improvement = 500 # Restart if stuck
    
    while time.time() < deadline:
        improved = False
        # Shuffle the order of mutations to avoid local minima traps
        # We have 2 * N = 24 possible single-edge changes.
        # For each, try changing the destination to all N possibilities.
        # Total candidates per step: 24 * 11 (excluding current) = 264.
        
        # To be faster, we can just pick a random subset of candidates.
        # Let's pick 50 random candidates.
        
        candidates = []
        # Generate random mutations
        for _ in range(50):
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            
            if letter == 0:
                if current0[state] == new_dest:
                    continue
                mut0 = current0[:]
                mut0[state] = new_dest
                mut1 = current1[:]
            else:
                if current1[state] == new_dest:
                    continue
                mut0 = current0[:]
                mut1 = current1[:]
                mut1[state] = new_dest
            
            candidates.append((mut0, mut1))
        
        best_candidate_score = current_score
        best_candidate_0 = None
        best_candidate_1 = None
        
        for mut0, mut1 in candidates:
            score = _compute_threshold(mut0, mut1)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate_0 = mut0
                best_candidate_1 = mut1
                improved = True # Found a better one, keep looking for even better?
                # For steepest ascent, we should find the BEST candidate.
                # For first-ascent, we can stop early. 
                # Let's do a "best-of-batch" approach to be more robust.
        
        if best_candidate_0 is not None:
            current0 = best_candidate_0
            current1 = best_candidate_1
            current_score = best_candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best0 = current0[:]
                best1 = current1[:]
                no_improvement_count = 0
            else:
                # We improved locally but not globally? 
                # Actually, if we accepted a move, it was better than previous current.
                # So current_score increased. 
                # best_score tracks the global max.
                pass
        else:
            no_improvement_count += 1
            
        # If stuck for a while, do a random restart or larger jump
        if no_improvement_count > max_no_improvement:
            # Restart from best known, but with a larger perturbation
            current0 = best0[:]
            current1 = best1[:]
            current_score = best_score
            
            # Perturb significantly
            for _ in range(rng.randint(5, 10)):
                letter = rng.randint(0, 1)
                state = rng.randint(0, N - 1)
                if letter == 0:
                    current0[state] = rng.randint(0, N - 1)
                else:
                    current1[state] = rng.randint(0, N - 1)
            
            # Check if this restart is valid/syncing
            new_score = _compute_threshold(current0, current1)
            if new_score > 0:
                current_score = new_score
            else:
                # Revert to best
                current0 = best0[:]
                current1 = best1[:]
                current_score = best_score
            
            no_improvement_count = 0

    return [best0, best1]

# EVOLVE-BLOCK-END
