"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    # Precompute masks for "vertex and its neighbors"
    # mask[v] is a 512-bit integer where bit v and bits of neighbors of v are 1
    masks = [0] * (1 << N)
    for v in range(1 << N):
        m = 1 << v
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        masks[v] = m
    
    def greedy_extend_fast(start, rng, max_time):
        """
        Fast greedy extension using bitmask for blocked vertices.
        blocked: 512-bit int. Bit v is 1 if v is in the 'exclusion zone' of the path so far.
        Exclusion zone for path p_0...p_k is union of {p_i} U N(p_i) for i < k.
        """
        path = [start]
        blocked = 0
        cur = start
        
        while time.time() < max_time:
            # Find valid next vertices: neighbors of cur not in blocked
            valid_candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if not (blocked >> v) & 1:
                    valid_candidates.append(v)
            
            if not valid_candidates:
                break
            
            # Randomly pick a candidate
            next_v = valid_candidates[rng.randint(0, len(valid_candidates) - 1)]
            
            # Update state
            path.append(next_v)
            # Add current vertex and its neighbors to blocked set
            blocked |= masks[cur]
            cur = next_v
        
        return path
    
    # Multiple restarts with different random seeds
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 1)
        
        # Start from a random vertex
        start = rng.randint(0, (1 << N) - 1)
        
        # Do a pure greedy extension
        path = greedy_extend_fast(start, rng, deadline - 1.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Early termination if we've found a very long snake
        if best_len >= 190:
            break
    
    if best_path is None:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
