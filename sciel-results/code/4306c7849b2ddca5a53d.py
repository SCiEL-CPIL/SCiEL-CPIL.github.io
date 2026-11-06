import time
import random
import numpy as np
from collections import deque

N = 12

def _compute_threshold_fast(trans):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Uses NumPy for vectorized state transitions.
    Returns -1 if not synchronizing.
    """
    # trans is a list of two lists, or a 2xN numpy array
    # trans[letter][state] = next_state
    trans = np.array(trans, dtype=np.uint8)
    
    # Precompute bitmasks for each letter and state
    # mask[letter][state] is a bitmask with bit 'state' set? 
    # No, we need to map a subset (bitmask) to a new subset.
    # Let's precompute for each letter L and each possible state s, the destination d = trans[L][s].
    # Then for a subset represented as a bitmask, we need to map each set bit s to trans[L][s].
    # This is not a simple bit operation unless we precompute the transition of every possible subset.
    # 2^12 = 4096 subsets. 2 letters. 
    # Precomputing the transition table for all 4096 subsets is very fast and allows O(1) lookup in BFS.
    
    num_subsets = 1 << N
    # subset_trans[letter][subset_mask] = new_subset_mask
    subset_trans = np.zeros((2, num_subsets), dtype=np.int32)
    
    for letter in range(2):
        for s in range(N):
            dest = trans[letter][s]
            # We can't just OR bits easily if multiple states map to the same dest.
            # But we can iterate through set bits of the subset.
            # Since we are precomputing, we can do this efficiently.
            
            # Alternative: Iterate through all 4096 subsets for each letter
            pass
            
    # Let's do the precomputation properly
    for letter in range(2):
        for mask in range(num_subsets):
            new_mask = 0
            # Iterate through set bits in mask
            # Or simply iterate through all states s
            for s in range(N):
                if mask & (1 << s):
                    dest = trans[letter][s]
                    new_mask |= (1 << dest)
            subset_trans[letter][mask] = new_mask

    full_set_mask = (1 << N) - 1
    target_mask = None
    
    # BFS
    visited = np.full(num_subsets, -1, dtype=np.int32)
    queue = deque()
    
    visited[full_set_mask] = 0
    queue.append(full_set_mask)
    
    while queue:
        current_mask = queue.popleft()
        dist = visited[current_mask]
        
        # Check if singleton
        # A mask is a singleton if it has exactly one bit set.
        # We can precompute a lookup table for "is singleton" and "which singleton"
        # Or just check bit count.
        # For speed, precompute singleton status.
        # But 4096 is small. 
        # Let's check: if mask & (mask - 1) == 0 and mask != 0
        if current_mask != 0 and (current_mask & (current_mask - 1)) == 0:
            return dist
            
        # Expand
        for letter in range(2):
            next_mask = subset_trans[letter][current_mask]
            if visited[next_mask] == -1:
                visited[next_mask] = dist + 1
                queue.append(next_mask)
                
    return -1

def _generate_pavlov_like(rng):
    """
    Generates a 'Pavlov' style automaton or similar structured family.
    These are known to have higher thresholds than the trivial N-1.
    
    Structure:
    State 0 is a sink for letter 'b' (or close to it).
    Letter 'a' is a permutation.
    Letter 'b' maps most states to state 0, but some map to others to create "delay".
    """
    # Letter a: random permutation (preferably a single cycle)
    trans_a = list(range(N))
    rng.shuffle(trans_a)
    
    # Letter b: 
    # Standard Pavlov: 
    # b(0) = 0
    # b(i) = 0 for i in 1..N-1 ? No, that syncs in N-1.
    # To get longer, we need b to not collapse everything to 0 immediately.
    # Let's try:
    # b(0) = 0
    # b(i) = i-1 for i=1..k
    # b(i) = 0 for i=k+1..N-1
    # This is still N-1.
    
    # Better:
    # Let's use the known "Cherny" construction for small N.
    # Or random search within the "near-permutation" space for 'b'.
    
    # Let's generate 'b' as a permutation with 1 or 2 "defects" (collisions).
    trans_b = list(range(N))
    rng.shuffle(trans_b)
    
    # Introduce 1-3 collisions
    num_collisions = rng.randint(1, 3)
    for _ in range(num_collisions):
        idx = rng.randint(0, N - 1)
        # Map to a random state, creating a collision
        # To bias towards slow sync, map to a state that is "far" in the permutation cycle?
        # Or just random.
        new_dest = rng.randint(0, N - 1)
        trans_b[idx] = new_dest
        
    return [trans_a, trans_b]

def _generate_random_biased(rng):
    """
    Generates a random automaton biased towards synchronizability and long thresholds.
    Letter a: Cycle.
    Letter b: Maps to a small set of hubs, but with some structure.
    """
    # Letter a: single cycle
    trans_a = list(range(N))
    rng.shuffle(trans_a)
    
    # Letter b: 
    # Pick a random subset of "hubs" of size 2-4
    num_hubs = rng.randint(2, 4)
    hubs = rng.sample(range(N), num_hubs)
    
    trans_b = []
    for i in range(N):
        if rng.random() < 0.7: # 70% chance to go to a hub
            trans_b.append(rng.choice(hubs))
        else:
            # 30% chance to go to a random state (might be a hub or not)
            trans_b.append(rng.randint(0, N - 1))
            
    return [trans_a, trans_b]

def construct():
    """
    Search for a synchronizing automaton with N=12 states that maximizes
    the shortest reset word length.
    """
    start_time = time.time()
    deadline = 88.0  # Leave 2s margin for safety
    
    best_score = -1
    best_trans = None
    
    # 1. Check the known baseline (N-1 = 11)
    baseline_0 = [(i + 1) % N for i in range(N)]
    baseline_1 = [0] * N
    for i in range(1, N):
        baseline_1[i] = i - 1
    baseline_trans = [baseline_0, baseline_1]
    score = _compute_threshold_fast(baseline_trans)
    if score > best_score:
        best_score = score
        best_trans = baseline_trans
        
    # 2. Search
    rng = random.Random(42)
    attempts = 0
    
    # We will alternate between different generation strategies
    strategies = [_generate_pavlov_like, _generate_random_biased]
    
    while time.time() < deadline:
        attempts += 1
        strategy = strategies[attempts % len(strategies)]
        
        candidate = strategy(rng)
        
        # Evaluate
        score = _compute_threshold_fast(candidate)
        
        if score > best_score:
            best_score = score
            best_trans = candidate
            
            # If we find a significantly better one, we might want to bias future searches towards it
            # For now, we just continue.
            
    # Fallback
    if best_trans is None:
        return baseline_trans
        
    return best_trans

# EVOLVE-BLOCK-END
