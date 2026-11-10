import time
import random

N = 256
_DEADLINE = 28.0  # Safety margin for 30s limit

def _is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    """Return set of all pairwise sums a+b for a<=b in S."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1, N]."""
    if not S:
        return False
    S_set = set(S)
    sums = _get_sums(S)
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        blocked = False
        for a in S:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _shrink_maximal(S, N):
    """
    Try to remove elements from a maximal Sidon set S while keeping it maximal.
    Returns the shrunk set.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        # Try removing each element
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            # Check if candidate is still maximal
            # We don't need to check if it's Sidon, because subset of Sidon is Sidon.
            # We only need to check maximality.
            if _is_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def _greedy_max_blocking(N, deadline):
    """
    Greedy construction using Max Blocking heuristic.
    At each step, pick the candidate that blocks the most other candidates.
    """
    selected = []
    used_sums = set()
    candidates = list(range(1, N + 1))
    
    while candidates:
        if time.time() > deadline:
            break
        
        valid_candidates = []
        for x in candidates:
            # Check if x is valid with current 'selected'
            new_sums = [2 * x] + [x + s for s in selected]
            if any(s in used_sums for s in new_sums):
                continue
            valid_candidates.append(x)
            
        if not valid_candidates:
            break
            
        best_candidate = None
        max_blocks = -1
        
        # Evaluate each valid candidate to see how many others it blocks
        for x in valid_candidates:
            blocks = 0
            added_sums = set([2 * x] + [x + s for s in selected])
            
            for y in valid_candidates:
                if y == x:
                    continue
                # Check if y would be blocked by adding x
                # y is blocked if any of its new sums (2y, y+s for s in selected, y+x)
                # collides with used_sums OR added_sums.
                # We know y is currently valid, so its sums don't collide with used_sums.
                # We just need to check against added_sums.
                
                blocked = False
                if 2 * y in added_sums:
                    blocked = True
                else:
                    for s in selected:
                        if y + s in added_sums:
                            blocked = True
                            break
                    # Also check y+x against used_sums?
                    # No, y+x is a new sum. It must not be in used_sums.
                    # But y is valid, so y+s not in used_sums.
                    # Is y+x in used_sums?
                    # If y+x = a+b (a,b in selected), then adding x creates sum x+y.
                    # If x+y is in used_sums, then x+y = a+b.
                    # This doesn't immediately block y in the sense of "y cannot be added later".
                    # Wait, the definition of "blocking" in greedy is:
                    # If we add x, can we still add y later?
                    # To add y later, y must be valid w.r.t selected U {x}.
                    # Sums for y: 2y, y+s (s in selected), y+x.
                    # These must be disjoint from used_sums U added_sums.
                    # We checked 2y and y+s against added_sums.
                    # We must check y+x against used_sums.
                    # And y+x against added_sums (already done: y+x not in added_sums as y!=x and y not in selected).
                    
                    if not blocked:
                        if y + x in used_sums:
                            blocked = True
                    
                if blocked:
                    blocks += 1
                    
            if blocks > max_blocks:
                max_blocks = blocks
                best_candidate = x
                
        if best_candidate is None:
            break
            
        selected.append(best_candidate)
        used_sums.update([2 * best_candidate] + [best_candidate + s for s in selected[:-1]])
        
        # Remove blocked candidates from the list for next iteration
        new_candidates = []
        added_sums = set([2 * best_candidate] + [best_candidate + s for s in selected[:-1]])
        for y in candidates:
            if y == best_candidate:
                continue
            # Check if y is still valid w.r.t selected U {best_candidate}
            blocked = False
            if 2 * y in used_sums:
                blocked = True
            else:
                for s in selected:
                    if y + s in used_sums:
                        blocked = True
                        break
                if not blocked:
                    # Check against the specific new sums from best_candidate?
                    # Actually, updating used_sums covers this.
                    # But we need to check if y is valid w.r.t the NEW used_sums.
                    # The loop above checks against 'selected' which includes best_candidate.
                    # So it checks y+s for all s in new selected.
                    # And 2y in used_sums (which includes 2*best_candidate).
                    # So this is correct.
                    pass
            
            if not blocked:
                new_candidates.append(y)
                
        candidates = new_candidates
        
    return selected

def _greedy_random_order(N, seed, deadline):
    """
    Greedy construction using random order.
    """
    random.seed(seed)
    selected = []
    used_sums = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        new_sums = [2 * x] + [x + s for s in selected]
        if any(s in used_sums for s in new_sums):
            continue
        selected.append(x)
        used_sums.update(new_sums)
        
    return selected

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Phase 1: Generate diverse maximal sets and shrink them
    # Strategy 1: Max Blocking Greedy (deterministic)
    # Strategy 2: Random Order Greedy (multiple seeds)
    
    iterations = 0
    while time.time() - start_time < _DEADLINE - 5:
        iterations += 1
        if iterations == 1:
            # Deterministic max blocking
            S = _greedy_max_blocking(N, time.time() + 2.0)
        else:
            # Random order greedy
            S = _greedy_random_order(N, iterations, time.time() + 1.0)
            
        if not S:
            continue
            
        # Ensure it's maximal (greedy usually is, but just in case)
        # Actually, greedy stops when no more can be added, so it is maximal.
        
        # Shrink it
        S_shrunk = _shrink_maximal(S, N)
        
        if _is_maximal(S_shrunk, N) and len(S_shrunk) < best_size:
            best_size = len(S_shrunk)
            best_set = S_shrunk
            
        # If we find a very small set, we can try to polish it more or stop if time is tight
        if best_size <= 20:
             # Try a few more random restarts focused on this size?
             # Or just continue until time runs out.
             pass

    # Fallback if nothing found (should not happen)
    if best_set is None:
        S = _greedy_max_blocking(N, time.time() + 2.0)
        best_set = S

    return best_set
