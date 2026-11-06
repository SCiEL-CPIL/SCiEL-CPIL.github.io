import time
import random
from collections import deque

N = 12

def _apply_subset(subset, trans, letter):
    """Apply a letter to a subset of states."""
    new_set = set()
    for s in subset:
        new_set.add(trans[letter][s])
    return frozenset(new_set)

def _compute_threshold(trans):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    full_set = frozenset(range(N))
    target_set = None
    
    # Check if it synchronizes at all and find a target
    # We can just BFS from full_set. If we reach any singleton, we are done.
    # But to be efficient, we can check if it's synchronizing first.
    # For N=12, the subset graph has 2^12 = 4096 nodes. BFS is very fast.
    
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # If current is a singleton, we found the threshold
        if len(current) == 1:
            return dist
            
        # Expand
        for letter in range(2):
            next_set = _apply_subset(current, trans, letter)
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
                
    return -1

def _generate_candidate(seed):
    """
    Generate a random candidate automaton biased towards long reset words.
    Strategy:
    1. Letter 0 is a permutation (cycle) to ensure no immediate collapse.
    2. Letter 1 is a function that maps most states to a small set of "hubs" 
       but preserves some structure to allow slow reduction.
    """
    rng = random.Random(seed)
    
    # Letter 0: A single cycle of length N is often optimal for "winding"
    # states to specific positions before compression.
    trans_0 = list(range(N))
    rng.shuffle(trans_0)
    
    # Letter 1: 
    # To maximize reset length, we want letter 1 to reduce the set size slowly.
    # A common pattern for high thresholds is that letter 1 maps i -> i-1 (mod N) 
    # or similar, but with "gaps" or specific mappings that prevent full sync
    # until many steps have passed.
    
    # Let's try a "shift" like map for letter 1 as well, but with a fixed point or
    # a specific structure.
    # Actually, the "Pavlov" or similar constructions often use:
    # a: cycle
    # b: maps everything to a small subset, but one state maps to itself or 
    #    creates a bottleneck.
    
    # Let's try a random function for letter 1 but ensure it's not a permutation.
    # To encourage long resets, letter 1 should not collapse the set too fast.
    # A good heuristic: letter 1 is also a permutation or near-permutation.
    
    # Let's generate trans_1 as a random permutation first, then maybe break it?
    # Or, let's use the known good structure:
    # trans_0[i] = (i + 1) % N
    # trans_1[i] = (i - 1) % N  <-- This just rotates, never syncs.
    # trans_1[i] = 0 if i==0 else i-1 <-- This syncs in N-1 steps.
    
    # To get MORE than N-1, we need the "alignment" to take longer.
    # This implies that applying 'b' doesn't just shift, but maybe the 'a' steps
    # needed to align the states before 'b' can effectively reduce the set are longer.
    
    # Let's try:
    # trans_0 is a cycle.
    # trans_1 is a map where most states go to i-1, but one specific state 'x' 
    # maps to 'y' such that it doesn't help sync immediately.
    
    # Actually, a better strategy for search:
    # Start with the known N-1 construction and mutate it.
    
    base_0 = [(i + 1) % N for i in range(N)]
    base_1 = [0] * N
    for i in range(1, N):
        base_1[i] = i - 1
        
    # Mutate base_1
    # Randomly change the destination of 1 or 2 states in trans_1
    trans_1 = base_1[:]
    num_mutations = rng.randint(0, 2)
    for _ in range(num_mutations):
        idx = rng.randint(0, N - 1)
        new_dest = rng.randint(0, N - 1)
        trans_1[idx] = new_dest
        
    return [base_0, trans_1]

def construct():
    """
    Search for a synchronizing automaton with N=12 states that maximizes
    the shortest reset word length.
    """
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for overhead
    
    best_score = -1
    best_trans = None
    
    # 1. Check the known baseline (N-1 = 11)
    baseline_0 = [(i + 1) % N for i in range(N)]
    baseline_1 = [0] * N
    for i in range(1, N):
        baseline_1[i] = i - 1
    baseline_trans = [baseline_0, baseline_1]
    score = _compute_threshold(baseline_trans)
    if score > best_score:
        best_score = score
        best_trans = baseline_trans
    
    # 2. Heuristic Search
    # We will try random permutations for letter 0 and random "shift-like" maps for letter 1.
    # To speed up, we can pre-filter candidates that are not synchronizing?
    # No, checking synchronizability is part of the BFS.
    
    # Let's try a specific family of automata known to have high thresholds.
    # The "Černý" automata often involve two cycles or a cycle and a map.
    
    # Strategy: 
    # Let letter 0 be a cycle.
    # Let letter 1 be a map that is a cycle on a subset and maps others into it?
    
    # Let's do a randomized local search around the baseline.
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate a candidate
        # We want to explore the space of "slow" synchronizing automata.
        # A key feature of high-threshold automata is that the "b" letter 
        # often acts as a "sliding window" or "shift" that only reduces 
        # the diameter of the subset by a small amount.
        
        # Let's try:
        # trans_0: random permutation (preferably a single cycle)
        # trans_1: random function with high "mixing" but not full collapse.
        
        # To bias towards long resets, we can try to make trans_1 also a permutation 
        # or close to it, but with one "defect" that allows synchronization.
        
        # Generate trans_0 as a single cycle
        trans_0 = list(range(N))
        rng.shuffle(trans_0)
        
        # Generate trans_1
        # Start with a shift
        shift = rng.randint(1, N - 1)
        trans_1 = [(i + shift) % N for i in range(N)]
        
        # Introduce 1-3 "breaks" in the permutation to allow synchronization
        num_breaks = rng.randint(1, 3)
        for _ in range(num_breaks):
            idx = rng.randint(0, N - 1)
            # Map this state to a random other state, or to 0, or to itself
            # Mapping to itself creates a fixed point, which can help sync
            # Mapping to 0 creates a sink
            # Let's try mapping to a random state
            new_dest = rng.randint(0, N - 1)
            trans_1[idx] = new_dest
            
        candidate = [trans_0, trans_1]
        
        # Evaluate
        score = _compute_threshold(candidate)
        
        if score > best_score:
            best_score = score
            best_trans = candidate
            
            # If we find a very good one, we might want to stay in this neighborhood
            # For now, just continue searching.
            
            # Optional: If score is very high, we can try local search around it
            # but for now, random restarts are cheap.

    # Final check: ensure best_trans is valid
    if best_trans is None:
        # Fallback to baseline
        return [baseline_0, baseline_1]
        
    return best_trans

# EVOLVE-BLOCK-END
