import time
from math import lcm

_START = time.time()
_DEADLINE = 80.0

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    """
    Returns a nonempty subset of [1, 22] such that all reciprocal subset sums are distinct,
    maximizing cardinality.
    """
    N = 22
    candidates = list(range(1, N + 1))
    
    best_set = []
    best_size = 0
    
    def backtrack(current_sums, L, selected, remaining):
        nonlocal best_set, best_size
        
        if not _time_ok():
            return
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        if not remaining:
            return
            
        # To improve performance, we can sort remaining candidates.
        # Heuristic: Try candidates that divide L first (they are the ones that might fail).
        # If they fail, we prune. If they succeed, we continue.
        # Candidates that don't divide L are always safe.
        
        # Actually, if we always add safe candidates, we might end up with a large L,
        # making many subsequent candidates "divisors" and thus subject to collision checks.
        # Conversely, if we delay safe candidates, L stays small, more candidates are "divisors".
        # It's hard to say which is better. Let's stick to natural order or a simple heuristic.
        
        for i, x in enumerate(remaining):
            if not _time_ok():
                return
                
            new_L = lcm(L, x)
            scale = new_L // L
            w = new_L // x
            
            collision = False
            
            if scale == 1:
                # x divides L
                for s in current_sums:
                    if (s + w) in current_sums:
                        collision = True
                        break
            else:
                # x does not divide L, always safe
                collision = False
            
            if collision:
                continue
                
            if scale == 1:
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + w)
            else:
                scaled_sums = {s * scale for s in current_sums}
                new_sums = set(scaled_sums)
                for s in scaled_sums:
                    new_sums.add(s + w)
            
            remaining_new = remaining[:i] + remaining[i+1:]
            backtrack(new_sums, new_L, selected + [x], remaining_new)
            
            if best_size == N:
                return

    backtrack({0}, 1, [], candidates)
    
    if not best_set:
        best_set = [1]
        
    return sorted(best_set)
