"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
"""
import math
import random
import time
import sys

def _compute_score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0  # degenerate: |U-U|=1, |U+U|=1, log(1/1)=0, but log(1)=0 -> undefined; return 1
    # Compute U+U and U-U sizes using sets
    U_set = set(U)
    # Sumset
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Difference set
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    log_num = math.log(len(diffset)) - math.log(len(sumset))
    log_den = math.log(2 * max_u + 1)
    if log_den == 0:
        return 1.0
    return 1.0 + log_num / log_den

def _greedy_construct(n, seed=42):
    """Greedy construction: add elements to maximize score incrementally."""
    random.seed(seed)
    U = [0]
    U_set = {0}
    best_score = _compute_score(U)
    
    # Candidate pool: start with small integers, expand as needed
    # We'll try adding elements in a structured way
    # Strategy: try to add elements that make the difference set large relative to sumset
    
    # Use a greedy approach: at each step, try adding a candidate and keep if it improves score
    max_val = 20000
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    deadline = time.time() + 100  # 100 second deadline
    
    for i in range(len(candidates)):
        if time.time() > deadline:
            break
        if len(U) >= n:
            break
        
        cand = candidates[i]
        # Quick check: will this help?
        # Try adding
        new_U = U + [cand]
        # Only compute full score if it's promising
        # Heuristic: prefer larger gaps
        if len(U) == 1:
            # First element: pick something reasonable
            U = new_U
            U_set.add(cand)
            best_score = _compute_score(U)
            continue
        
        # Compute score for new_U (expensive for large U, so limit)
        if len(U) > 200:
            # Too expensive to compute full score, use heuristic
            # Skip this candidate if it's too close to existing elements
            min_dist = min(abs(cand - u) for u in U)
            if min_dist < len(U):
                continue
        
        new_score = _compute_score(new_U)
        if new_score > best_score + 1e-9:
            U = new_U
            U_set.add(cand)
            best_score = new_score
    
    return sorted(U), best_score

def _random_local_search(U, iterations=5000, deadline=None):
    """Local search: try swapping or adding/removing elements."""
    random.seed(123)
    U = sorted(U)
    U_set = set(U)
    best_score = _compute_score(U)
    best_U = U[:]
    
    if deadline is None:
        deadline = time.time() + 100
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        if random.random() < 0.5 and len(U) < 300:
            # Pick a random candidate
            max_u = max(U) if U else 0
            # Try elements in range
            cand = random.randint(1, max_u + max(100, len(U) * 10))
            if cand not in U_set:
                new_U = sorted(U + [cand])
                new_score = _compute_score(new_U)
                if new_score > best_score + 1e-9:
                    U = new_U
                    U_set.add(cand)
                    best_score = new_score
                    best_U = U[:]
        else:
            # Try removing an element (if size > 2)
            if len(U) > 2:
                idx = random.randint(0, len(U) - 1)
                if U[idx] != 0:  # Must keep 0
                    new_U = U[:idx] + U[idx+1:]
                    new_score = _compute_score(new_U)
                    if new_score > best_score + 1e-9:
                        U = new_U
                        U_set.discard(U[idx])
                        best_score = new_score
                        best_U = U[:]
    
    return best_U, best_score

def _construct_with_budget():
    """Main construction with time budget management."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with a reasonable seed set
    # Known good constructions use geometric-like progressions
    # Try: powers of 2, or mixed bases
    
    # Strategy 1: Greedy from small set
    # Strategy 2: Structured construction
    
    best_U = [0]
    best_score = _compute_score(best_U)
    
    # Phase 1: Quick greedy construction
    U_greedy, score_greedy = _greedy_construct(100, seed=42)
    if score_greedy > best_score:
        best_U = U_greedy
        best_score = score_greedy
    
    if time.time() > deadline - 5:
        return sorted(best_U)
    
    # Phase 2: Local search refinement
    U_ls, score_ls = _random_local_search(best_U, iterations=2000, deadline=deadline - 5)
    if score_ls > best_score:
        best_U = U_ls
        best_score = score_ls
    
    if time.time() > deadline - 2:
        return sorted(best_U)
    
    # Phase 3: Try different seeds for greedy
    for seed in [1, 7, 13, 99, 12345]:
        if time.time() > deadline - 2:
            break
        U_g, s_g = _greedy_construct(100, seed=seed)
        if s_g > best_score:
            best_U = U_g
            best_score = s_g
            # Quick local search
            U_ls2, s_ls2 = _random_local_search(best_U, iterations=1000, deadline=deadline - 2)
            if s_ls2 > best_score:
                best_U = U_ls2
                best_score = s_ls2
    
    return sorted(best_U)

def construct():
    """Construct a finite set U of nonnegative integers containing 0.
    
    Returns a list of nonnegative integers.
    """
    try:
        result = _construct_with_budget()
        # Ensure 0 is in the result and all elements are nonnegative integers
        result = [x for x in result if isinstance(x, int) and x >= 0]
        if 0 not in result:
            result.insert(0, 0)
        return sorted(result)
    except Exception:
        # Fallback: return a simple valid set
        return [0, 1, 3]

# EVOLVE-BLOCK-START
# (construct is defined above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
