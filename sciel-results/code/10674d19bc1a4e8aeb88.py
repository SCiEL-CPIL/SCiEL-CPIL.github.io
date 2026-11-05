"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TIME_LIMIT = 110.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104

# EVOLVE-BLOCK-START
def _greedy_search_with_lookahead(rng, time_budget, lookahead_depth=2):
    """
    Run a greedy snake construction with lookahead scoring.
    At each step, for each candidate, estimate the number of valid extensions
    after choosing it (up to lookahead_depth levels deep), and pick the candidate
    that maximizes this score. Tie-break randomly.
    """
    path = [0]
    used = {0}
    exclusion = set()
    
    while True:
        cur = path[-1]
        # Find valid next vertices
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            return path
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Score each candidate by looking ahead
            best_score = -1
            best_candidates = []
            for c in candidates:
                score = _lookahead_score(c, path, used, exclusion, lookahead_depth)
                if score > best_score:
                    best_score = score
                    best_candidates = [c]
                elif score == best_score:
                    best_candidates.append(c)
            v = rng.choice(best_candidates)
        
        path.append(v)
        used.add(v)
        
        # Update exclusion: the old terminal (cur) becomes non-terminal
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
        
        # Time check
        if time.time() > time_budget:
            return path

def _lookahead_score(candidate, path, used, exclusion, depth):
    """
    Estimate how many further steps are possible after choosing candidate.
    Returns a score based on the number of valid extensions at the next level(s).
    """
    if depth <= 0:
        return 1  # base case: at least 1 (the candidate itself)
    
    # Simulate adding candidate
    new_used = used | {candidate}
    new_exclusion = exclusion | set()
    # Add all neighbors of path[-1] (the old terminal) to new_exclusion
    old_term = path[-1]
    for b in range(N):
        nb = old_term ^ (1 << b)
        if nb not in exclusion:  # only add if not already there
            new_exclusion.add(nb)
    
    # Now count valid next candidates from candidate
    count = 0
    for b in range(N):
        v = candidate ^ (1 << b)
        if v not in new_used and v not in new_exclusion:
            count += 1
            if depth > 1:
                # Recursively estimate (shallow)
                count += _lookahead_score(v, path + [candidate], new_used, new_exclusion, depth - 1)
    
    return count

def _greedy_search(rng, time_budget):
    """Run one greedy snake construction with randomization (no lookahead)."""
    path = [0]
    used = {0}
    exclusion = set()
    
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            return path
        
        v = rng.choice(candidates)
        path.append(v)
        used.add(v)
        
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
        
        if time.time() > time_budget:
            return path

def _try_extend(path, used, exclusion, rng, time_budget):
    """
    Try to extend an existing path further using greedy lookahead.
    Returns the extended path.
    """
    # Reconstruct used and exclusion from path
    # used is already given
    # exclusion: all vertices at Hamming distance 1 from any non-terminal vertex
    # Non-terminal vertices are path[0:-1]
    for p in path[:-1]:
        for b in range(N):
            nb = p ^ (1 << b)
            exclusion.add(nb)
    
    # Now try to extend
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            return path
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Use lookahead to pick best
            best_score = -1
            best_cands = []
            for c in candidates:
                score = _lookahead_score(c, path, used, exclusion, 2)
                if score > best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        
        path.append(v)
        used.add(v)
        
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
        
        if time.time() > time_budget:
            return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First run: deterministic greedy to get baseline
    path, used = [0], {0}
    exclusion = set()
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or v in exclusion:
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
    
    best_path = path
    best_len = len(path)
    
    # Run multiple strategies:
    # 1. Randomized greedy without lookahead (fast, many restarts)
    # 2. Randomized greedy with lookahead depth 1 (slower, fewer restarts)
    # 3. Randomized greedy with lookahead depth 2 (slowest, fewest restarts)
    
    # Strategy 1: Fast randomized greedy
    while time.time() < deadline - 30.0:
        p = _greedy_search(rng, deadline - 29.0)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 2: Lookahead depth 1
    while time.time() < deadline - 15.0:
        p = _greedy_search_with_lookahead(rng, deadline - 14.0, lookahead_depth=1)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 3: Lookahead depth 2 (fewer but potentially better)
    while time.time() < deadline - 2.0:
        p = _greedy_search_with_lookahead(rng, deadline - 1.0, lookahead_depth=2)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Final: try to extend the best path found
    if best_path is not None and time.time() < deadline - 1.0:
        used = set(best_path)
        exclusion = set()
        extended = _try_extend(list(best_path), used, exclusion, rng, deadline - 0.5)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    return best_path
# EVOLVE-BLOCK-END
