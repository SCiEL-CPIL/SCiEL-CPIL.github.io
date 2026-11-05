"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
TIME_LIMIT = 110.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104

# Precompute neighbors for all vertices in Q_9
# neighbors[v] is a tuple of 9 integers
neighbors = []
for v in range(1 << N):
    nb = tuple(v ^ (1 << b) for b in range(N))
    neighbors.append(nb)

# EVOLVE-BLOCK-START
def _greedy_search(rng, time_budget, use_lookahead=True):
    """
    Run one greedy snake construction.
    If use_lookahead is True, picks the candidate with the most valid extensions.
    Otherwise, picks a random candidate.
    """
    path = [0]
    used = {0}
    # exclusion: set of vertices at Hamming distance 1 from any non-terminal vertex
    exclusion = set()
    
    # Local variables for speed
    nbrs = neighbors
    rng_choice = rng.choice
    time_check = time.time
    
    while True:
        cur = path[-1]
        # Find valid next vertices
        candidates = []
        for v in nbrs[cur]:
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            return path
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            if use_lookahead:
                # Score candidates by number of valid extensions
                # Note: We use a slightly more accurate score by considering
                # that 'cur' will become excluded, so its neighbors are effectively blocked.
                # However, the standard greedy heuristic is just count available neighbors of v.
                best_score = -1
                best_v = None
                tied_vs = []
                for v in candidates:
                    score = 0
                    for nv in nbrs[v]:
                        if nv not in used and nv not in exclusion:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                        tied_vs = [v]
                    elif score == best_score:
                        tied_vs.append(v)
                
                if len(tied_vs) > 1:
                    v = rng_choice(tied_vs)
                else:
                    v = best_v
            else:
                v = rng_choice(candidates)
        
        path.append(v)
        used.add(v)
        
        # Update exclusion: the old terminal (cur) becomes non-terminal
        # Add all neighbors of cur to exclusion
        for nb in nbrs[cur]:
            exclusion.add(nb)
        
        # Time check
        if time_check() > time_budget:
            return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First run: deterministic greedy (original) to get baseline
    # We can reuse the greedy function with a fixed seed or just run it
    # Let's run a standard greedy first to ensure we have a baseline
    path, used = [0], {0}
    exclusion = set()
    nbrs = neighbors
    while True:
        cur = path[-1]
        nxt = None
        for v in nbrs[cur]:
            if v not in used and v not in exclusion:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        for nb in nbrs[cur]:
            exclusion.add(nb)
    
    best_path = path
    best_len = len(path)
    
    # Now run randomized greedy searches
    search_count = 0
    # We will run many fast iterations
    while time.time() < deadline - 1.0:  # 1s safety margin
        # Alternate strategies to maintain diversity
        # 1. Lookahead greedy
        # 2. Random greedy
        # 3. Lookahead greedy with different random seed behavior (implied by rng)
        
        if search_count % 3 == 0:
            p = _greedy_search(rng, deadline - 0.5, use_lookahead=True)
        elif search_count % 3 == 1:
            p = _greedy_search(rng, deadline - 0.5, use_lookahead=False)
        else:
            # Hybrid: Lookahead but with random tie-breaking (already in _greedy_search)
            # Or maybe a "biased" random?
            p = _greedy_search(rng, deadline - 0.5, use_lookahead=True)
            
        search_count += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    return best_path
# EVOLVE-BLOCK-END
