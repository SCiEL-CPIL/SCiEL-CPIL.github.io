import random
import time

N = 256

def _is_sidon(S):
    S = sorted(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal_fast(S, sums):
    """Check maximality: every missing element must be blocked."""
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for x in range(1, N+1):
        if x in S_set:
            continue
        if 2*x in sums:
            continue
        # Check if any s in S has x+s in sums
        blocked = False
        for s in S_list:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal():
    selected = []
    sums = set()
    for x in range(1, N+1):
        added = [x + a for a in selected] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _reduce(S, deadline=None):
    """Try to remove elements while maintaining maximality."""
    S = sorted(S)
    changed = True
    while changed:
        changed = False
        if deadline and time.time() > deadline:
            break
        for i in range(len(S)):
            if deadline and time.time() > deadline:
                break
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            if not _is_sidon(candidate):
                continue
            sums = _get_sums(candidate)
            if _is_maximal_fast(candidate, sums):
                S = candidate
                changed = True
                break
    return S

def _local_search_swap(S, sums, deadline):
    """Try swap moves: remove one element, add another, check if still maximal and smaller or same size."""
    S = sorted(S)
    S_set = set(S)
    improved = False
    for i in range(len(S)):
        if time.time() > deadline:
            break
        # Try removing S[i]
        S_without = S[:i] + S[i+1:]
        if not S_without:
            continue
        if not _is_sidon(S_without):
            continue
        sums_without = _get_sums(S_without)
        S_without_set = set(S_without)
        
        # Now try adding each missing element
        for x in range(1, N+1):
            if x in S_without_set:
                continue
            # Check if x can be added to S_without
            added = [x + a for a in S_without] + [2*x]
            if len(set(added)) != len(added):
                continue
            if sums_without.intersection(added):
                continue
            # S_without + [x] is a valid Sidon set of same size
            # Check if it's maximal
            new_S = S_without + [x]
            new_S.sort()
            new_sums = sums_without | set(added)
            if _is_maximal_fast(new_S, new_sums):
                return new_S
    return None

def _blocking_power(x, current_set, current_sums):
    """Count how many missing elements would be blocked if x were added."""
    S_set = set(current_set)
    new_sums = current_sums | {x + a for a in current_set} | {2*x}
    blocked_count = 0
    for y in range(1, N+1):
        if y in S_set or y == x:
            continue
        # Check if y would be blocked by the new sums
        if 2*y in new_sums:
            blocked_count += 1
            continue
        for s in current_set:
            if y + s in new_sums:
                blocked_count += 1
                break
        else:
            if y + x in new_sums:
                blocked_count += 1
    return blocked_count

def _dominating_greedy(seed_offset=0):
    """Build a Sidon set using dominating-set heuristic: prioritize high-blocking elements."""
    S = set()
    sums = set()
    candidates = list(range(1, N+1))
    random.seed(42 + seed_offset)
    
    # Iteratively pick the element with highest blocking power
    while candidates:
        if not candidates:
            break
        
        # Compute blocking power for each candidate
        best_x = None
        best_power = -1
        for x in candidates:
            # Check if x is compatible
            added = [x + a for a in S] + [2*x]
            if len(set(added)) != len(added):
                continue
            if sums.intersection(set(added)):
                continue
            power = _blocking_power(x, S, sums)
            if power > best_power:
                best_power = power
                best_x = x
        
        if best_x is None:
            break
        
        S.add(best_x)
        added = [best_x + a for a in S] + [2*best_x]
        sums.update(added)
        candidates.remove(best_x)
    
    return sorted(S), sums

def _search_maximal_sidon(k, deadline):
    """
    Backtracking search for a maximal Sidon set of exactly size k.
    Uses heuristic: prefer elements that block many other candidates.
    """
    if time.time() > deadline:
        return None
    
    def backtrack(current_set, current_sums, remaining, target_size):
        if time.time() > deadline:
            return None
        
        if len(current_set) == target_size:
            # Check if maximal
            if _is_maximal_fast(current_set, current_sums):
                return current_set[:]
            return None
        
        if not remaining:
            return None
            
        # Heuristic: Sort remaining by blocking power (descending)
        candidates = list(remaining)
        # Compute blocking power for each candidate and sort
        scored = []
        for x in candidates:
            added = [x + a for a in current_set] + [2*x]
            if len(set(added)) != len(added):
                continue
            if current_sums.intersection(set(added)):
                continue
            power = _blocking_power(x, current_set, current_sums)
            scored.append((power, x))
        
        scored.sort(key=lambda t: -t[0])
        
        for _, x in scored:
            new_sums_added = [x + a for a in current_set] + [2*x]
            new_set = current_set + [x]
            new_sums = current_sums.union(set(new_sums_added))
            
            res = backtrack(new_set, new_sums, [y for y in remaining if y > x], target_size)
            
            if res:
                return res
                
        return None

    max_restarts = 20
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        res = backtrack([], set(), list(range(1, N+1)), k)
        if res:
            return res
    return None

def construct():
    deadline = time.time() + 25  # safety margin
    
    best = _greedy_maximal()
    best_size = len(best)
    
    # Try to reduce the greedy result
    reduced = _reduce(best, deadline)
    if len(reduced) < best_size:
        best = reduced
        best_size = len(best)
    
    # Try dominating greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        S, sums = _dominating_greedy(seed)
        
        if len(S) < best_size:
            # Check maximality
            if _is_maximal_fast(S, sums):
                best = S
                best_size = len(best)
                # Try to reduce further
                reduced = _reduce(best, deadline)
                if len(reduced) < best_size:
                    best = reduced
                    best_size = len(best)
                # Try swap local search
                if time.time() < deadline:
                    swapped = _local_search_swap(best, _get_sums(best), deadline)
                    if swapped and len(swapped) < best_size:
                        best = swapped
                        best_size = len(best)
                        reduced = _reduce(best, deadline)
                        if len(reduced) < best_size:
                            best = reduced
                            best_size = len(best)
    
    # New Strategy: Backtracking for small sizes
    # Try k from 8 to best_size - 1
    for k in range(8, best_size):
        if time.time() > deadline:
            break
        found = _search_maximal_sidon(k, deadline)
        if found:
            best = found
            best_size = k
            break

    return best
