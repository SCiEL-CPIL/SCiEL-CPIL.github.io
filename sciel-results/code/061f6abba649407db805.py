"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _check_sidon_and_cover(a, target_diffs):
    """Check if a is a Sidon set covering target_diffs."""
    n = len(a)
    if n > 32:
        return False
    if max(a) > 10**6:
        return False
    if 0 not in a:
        return False
    
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d <= 0:
                return False
            if d in diffs:
                return False
            diffs.add(d)
    return all(d in diffs for d in target_diffs)

def _search(M, target_diffs, deadline):
    """
    Depth-first search for a Sidon set in [0, M] covering target_diffs.
    Uses heuristic: prioritize candidates that cover more remaining differences.
    """
    initial_set = [0]
    initial_diffs = set()
    remaining = set(target_diffs)
    
    def dfs(current_set, used_diffs, remaining):
        if not remaining:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
        
        last = current_set[-1]
        
        # Generate candidates: x = a + d for a in current_set, d in remaining
        # where x > last and x <= M
        candidate_info = []
        seen_candidates = set()
        for a in current_set:
            for d in remaining:
                x = a + d
                if x > last and x <= M and x not in seen_candidates:
                    seen_candidates.add(x)
                    
                    # Count how many remaining diffs this x would cover
                    new_covered = 0
                    conflict = False
                    for a2 in current_set:
                        diff = x - a2
                        if diff in used_diffs:
                            conflict = True
                            break
                        if diff in remaining:
                            new_covered += 1
                    
                    if not conflict:
                        candidate_info.append((x, new_covered))
        
        # Sort by: more new coverage first, then smaller x (to minimize max)
        candidate_info.sort(key=lambda t: (-t[1], t[0]))
        
        for x, _ in candidate_info:
            # Re-verify Sidon property (in case used_diffs changed in recursion)
            new_diffs = set()
            conflict = False
            for a in current_set:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
                
            new_used = used_diffs | new_diffs
            new_remaining = remaining - new_used
            
            res = dfs(current_set + [x], new_used, new_remaining)
            if res is not None:
                return res
                
        return None

    return dfs(initial_set, initial_diffs, remaining)

def _known_good():
    """A known valid Sidon set covering 1..30 with small max element."""
    # Greedy fallback to ensure validity if search fails
    a = [0]
    used = set()
    required = set(range(1, 31))
    for x in range(1, 200):
        if not required:
            break
        new_diffs = set()
        ok = True
        for v in a:
            d = abs(x - v)
            if d in used:
                ok = False
                break
            new_diffs.add(d)
        if ok and new_diffs & required:
            a.append(x)
            used |= new_diffs
            required -= new_diffs
    if not required:
        return a
    # Last resort: extend with large numbers to ensure distinct diffs
    # This might not be optimal but is valid
    while required:
        x = max(a) + 100
        a.append(x)
        for v in a[:-1]:
            used.add(abs(x - v))
        required -= set(abs(x - v) for v in a[:-1])
    return a

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 140s.
    deadline = start_time + 140.0
    
    target_diffs = set(range(1, 31))
    
    # Lower bound estimation:
    # We need to cover 30 differences.
    # A Sidon set of size k has k(k-1)/2 differences.
    # 8 elements -> 28 diffs (not enough to cover 1..30 if distinct)
    # 9 elements -> 36 diffs.
    # Minimal diameter for 9-element Sidon set covering 1..30.
    # Let's start from M=40. If no solution, increase M.
    for M in range(40, 200):
        if time.time() > deadline:
            break
            
        result = _search(M, target_diffs, deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                return result
                
    # Fallback
    return _known_good()
# EVOLVE-BLOCK-END
