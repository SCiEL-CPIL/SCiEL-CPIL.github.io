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
    Uses a weighted score that considers both immediate options and future openness.
    """
    if depth <= 0:
        return 1  # base case: at least 1 (the candidate itself)
    
    # Simulate adding candidate
    new_used = used | {candidate}
    new_exclusion = set(exclusion)
    # Add all neighbors of path[-1] (the old terminal) to new_exclusion
    old_term = path[-1]
    for b in range(N):
        nb = old_term ^ (1 << b)
        new_exclusion.add(nb)
    
    # Count valid next candidates from candidate, with weighted scoring
    total_score = 0.0
    valid_count = 0
    for b in range(N):
        v = candidate ^ (1 << b)
        if v not in new_used and v not in new_exclusion:
            valid_count += 1
            if depth > 1:
                # Recursively estimate with exponential decay
                sub_score = _lookahead_score(v, path + [candidate], new_used, new_exclusion, depth - 1)
                total_score += sub_score * (0.7 ** (depth - 1))
            else:
                total_score += 1.0
    
    # Bonus for having more options (encourages branching)
    if valid_count > 0:
        total_score *= (1.0 + 0.1 * valid_count)
    
    return total_score

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
    Try to extend an existing path further using greedy lookahead with backtracking.
    Returns the extended path.
    """
    # Reconstruct used and exclusion from path
    for p in path[:-1]:
        for b in range(N):
            nb = p ^ (1 << b)
            exclusion.add(nb)
    
    # Now try to extend with limited backtracking
    backtrack_limit = 3
    best_extension = list(path)
    best_len = len(path)
    
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            # Try backtracking
            if backtrack_limit > 0 and len(path) > 1:
                # Remove last vertex and try alternative
                removed = path.pop()
                used.remove(removed)
                # Remove exclusions that were added for the new terminal
                old_term = path[-1] if path else None
                # Recompute exclusion from scratch is expensive; instead, 
                # we'll just try different choices by re-adding and trying alternatives
                path.append(removed)
                used.add(removed)
                backtrack_limit -= 1
                # For now, just return if we can't extend
                return best_extension if len(best_extension) > len(path) else path
            else:
                return best_extension if len(best_extension) > len(path) else path
        
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
        
        if len(path) > best_len:
            best_len = len(path)
            best_extension = list(path)
        
        if time.time() > time_budget:
            return best_extension if len(best_extension) > len(path) else path

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
    
    # Final: try to extend the best path found with multiple attempts
    if best_path is not None and time.time() < deadline - 1.0:
        used = set(best_path)
        exclusion = set()
        
        # Try multiple extension attempts with different random seeds
        for ext_attempt in range(3):
            if time.time() > deadline - 0.5:
                break
            ext_rng = random.Random(42 + ext_attempt * 1000)
            extended = _try_extend(list(best_path), set(best_path), set(), ext_rng, deadline - 0.5)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended
    
    return best_path
# EVOLVE-BLOCK-END
