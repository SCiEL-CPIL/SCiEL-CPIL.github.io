"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
TIME_LIMIT = 110.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104

# EVOLVE-BLOCK-START
def _count_valid_extensions(v, used, exclusion):
    """Count how many valid next moves would be available if we add v to the path."""
    count = 0
    for b in range(N):
        nv = v ^ (1 << b)
        if nv not in used and nv not in exclusion:
            count += 1
    return count

def _greedy_search_with_lookahead(rng, time_budget, lookahead_depth=1):
    """Run greedy snake construction with look-ahead heuristic."""
    path = [0]
    used = {0}
    # exclusion: set of vertices at Hamming distance 1 from any non-terminal vertex
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
            # Use look-ahead: score candidates by number of valid extensions
            if lookahead_depth >= 1 and len(candidates) <= 10:
                # For small candidate sets, do full look-ahead
                scored = []
                for v in candidates:
                    # Simulate adding v and count next moves
                    # The new exclusion would include neighbors of cur
                    # But we don't want to actually modify, so we estimate
                    # Count neighbors of v that are not used and not in (exclusion ∪ neighbors of cur)
                    # Actually, simpler: just count valid neighbors of v given current state
                    # Note: when we add v, cur becomes non-terminal, so neighbors of cur become excluded
                    # But for the look-ahead, we're asking: if I go to v, how many moves from v?
                    # The exclusion for v's neighbors would be: current exclusion ∪ {neighbors of cur}
                    # But we can approximate by just checking current exclusion + used
                    score = _count_valid_extensions(v, used, exclusion)
                    scored.append((score, v))
                # Pick the one with highest score, with some randomness for ties
                max_score = max(s for s, _ in scored)
                best_candidates = [v for s, v in scored if s == max_score]
                v = rng.choice(best_candidates)
            else:
                # Random choice for large candidate sets or no look-ahead
                v = rng.choice(candidates)
        
        path.append(v)
        used.add(v)
        
        # Update exclusion: the old terminal (cur) becomes non-terminal
        # Add all neighbors of cur to exclusion
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
        
        # Time check
        if time.time() > time_budget:
            return path

def _greedy_search_no_lookahead(rng, time_budget):
    """Run one greedy snake construction with randomization (no look-ahead)."""
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

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First run: deterministic greedy (original) to get baseline
    path, used = [0], {0}
    exclusion = set()
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in exclusion:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        # Update exclusion: cur becomes non-terminal
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
    best_path = path
    best_len = len(path)
    
    # Now run randomized greedy searches with look-ahead
    search_count = 0
    while time.time() < deadline - 2.0:  # 2s safety margin
        # Alternate between look-ahead and no look-ahead for diversity
        if search_count % 2 == 0:
            p = _greedy_search_with_lookahead(rng, deadline - 1.0, lookahead_depth=1)
        else:
            p = _greedy_search_no_lookahead(rng, deadline - 1.0)
        search_count += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    return best_path
# EVOLVE-BLOCK-END
