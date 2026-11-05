import time
import random
import sys

sys.setrecursionlimit(10000)

def _is_valid_fast(seq):
    """Check if seq has all distinct consecutive block sums."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_forward_optimized():
    """Simple forward greedy: try each x from 1 to 256, append if valid."""
    selected = []
    sums = set()
    for x in range(1, 257):
        new_sums = []
        current_sum = 0
        valid = True
        for a in reversed(selected):
            current_sum += a
            new_sum = current_sum + x
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        selected.append(x)
        sums.update(new_sums)
    return selected

def _greedy_reverse_optimized():
    """Simple reverse greedy: try each x from 256 down to 1, prepend if valid."""
    selected = []
    sums = set()
    for x in range(256, 0, -1):
        new_sums = []
        current_sum = 0
        valid = True
        for a in selected:
            current_sum += a
            new_sum = x + current_sum
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        selected.insert(0, x)
        sums.update(new_sums)
    return selected

def _get_valid_candidates(values, sums, last_val):
    """Return list of x > last_val such that appending x to values keeps all block sums distinct."""
    valid = []
    n = len(values)
    for x in range(last_val + 1, 257):
        total = x
        if total in sums:
            continue
        ok = True
        for i in range(n - 1, -1, -1):
            total += values[i]
            if total in sums:
                ok = False
                break
        if ok:
            valid.append(x)
    return valid

def _score_candidate(x, values, sums):
    """
    Score a candidate x for appending to values.
    Higher score = more promising.
    Heuristic: prefer candidates that create larger gaps and avoid near-collisions.
    """
    n = len(values)
    if n == 0:
        return -x  # Prefer smaller values when starting
    
    # Gap from last value
    gap = x - values[-1]
    
    # Count how many existing sums are "close" to the new sums we'd create
    # New sums: x, x+values[-1], x+values[-1]+values[-2], ...
    new_sums = []
    total = x
    new_sums.append(total)
    for i in range(n - 1, -1, -1):
        total += values[i]
        new_sums.append(total)
    
    # Penalty for sums that are close to existing sums (potential future collisions)
    collision_penalty = 0
    for ns in new_sums:
        for s in sums:
            diff = abs(ns - s)
            if diff <= 3:
                collision_penalty += (4 - diff)
    
    # Prefer larger gaps (more room for future insertions)
    gap_bonus = gap * 0.5
    
    return gap_bonus - collision_penalty

def _branch_and_bound_dfs(start_val, deadline, max_depth, best_ref):
    """
    DFS with branch and bound. State: (last_val, frozenset_of_sums).
    We explore appending values > last_val. Use memo to prune states
    that have been visited with >= remaining depth potential.
    Returns the best sequence found in this DFS tree.
    """
    best_seq = []
    best_len = 0
    memo = {}
    
    def dfs(values, sums, last_val, remaining_depth):
        nonlocal best_seq, best_len, memo
        
        if time.time() > deadline:
            return
        
        current_len = len(values)
        if current_len > best_len:
            best_len = current_len
            best_seq = list(values)
        
        if remaining_depth <= 0:
            return
        
        candidates = _get_valid_candidates(values, sums, last_val)
        if not candidates:
            return
        
        # Score and sort candidates: prefer those with higher scores
        scored = [(x, _score_candidate(x, values, sums)) for x in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        candidates_sorted = [x for x, _ in scored]
        
        # Limit branching factor - be more aggressive
        max_branch = min(len(candidates_sorted), 12)
        
        for x in candidates_sorted[:max_branch]:
            if time.time() > deadline:
                return
            
            state_key = (last_val, frozenset(sums))
            if state_key in memo and memo[state_key] >= remaining_depth:
                continue
            memo[state_key] = remaining_depth
            
            new_sums = set(sums)
            total = x
            new_sums.add(total)
            for a in reversed(values):
                total += a
                new_sums.add(total)
            
            values.append(x)
            dfs(values, new_sums, x, remaining_depth - 1)
            values.pop()
    
    dfs([], set(), start_val, max_depth)
    return best_seq

def _dfs_with_randomization(seed, deadline, max_depth):
    """Run DFS with randomized candidate ordering."""
    rand = random.Random(seed)
    best_seq = []
    best_len = 0
    memo = {}
    
    def dfs(values, sums, last_val, remaining_depth):
        nonlocal best_seq, best_len, memo
        
        if time.time() > deadline:
            return
        
        current_len = len(values)
        if current_len > best_len:
            best_len = current_len
            best_seq = list(values)
        
        if remaining_depth <= 0:
            return
        
        candidates = _get_valid_candidates(values, sums, last_val)
        if not candidates:
            return
        
        # Score candidates, then randomly select from top ones
        scored = [(x, _score_candidate(x, values, sums)) for x in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        
        # Take top 15, then shuffle for diversity
        top_k = min(len(scored), 15)
        top_candidates = [x for x, _ in scored[:top_k]]
        rand.shuffle(top_candidates)
        
        max_branch = min(len(top_candidates), 8)
        
        for x in top_candidates[:max_branch]:
            if time.time() > deadline:
                return
            
            state_key = (last_val, frozenset(sums))
            if state_key in memo and memo[state_key] >= remaining_depth:
                continue
            memo[state_key] = remaining_depth
            
            new_sums = set(sums)
            total = x
            new_sums.add(total)
            for a in reversed(values):
                total += a
                new_sums.add(total)
            
            values.append(x)
            dfs(values, new_sums, x, remaining_depth - 1)
            values.pop()
    
    dfs([], set(), 0, max_depth)
    return best_seq

def _local_improve(seq, time_limit):
    """Try to improve sequence by inserting new elements."""
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    n = len(best)
    
    def compute_sums(s):
        sums = set()
        nn = len(s)
        for i in range(nn):
            t = 0
            for j in range(i, nn):
                t += s[j]
                sums.add(t)
        return sums
    
    # Try inserting a new element not in best
    existing = set(best)
    available = [x for x in range(1, 257) if x not in existing]
    
    # Sort available by heuristic: prefer values that create larger gaps
    def insertion_score(val, pos):
        left = best[pos-1] if pos > 0 else 0
        right = best[pos] if pos < len(best) else 257
        gap = (right - left)
        # Prefer positions with larger gaps
        return gap
    
    # Try all insertions, sorted by gap size
    insertions = []
    for val in available:
        for pos in range(len(best) + 1):
            if pos > 0 and best[pos-1] >= val:
                continue
            if pos < len(best) and val >= best[pos]:
                continue
            score = insertion_score(val, pos)
            insertions.append((score, val, pos))
    
    insertions.sort(key=lambda t: t[0], reverse=True)
    
    for score, val, pos in insertions:
        if time.time() > time_limit:
            break
        
        candidate = best[:pos] + [val] + best[pos:]
        
        if _is_valid_fast(candidate):
            best = candidate
            best_len = len(best)
            n = len(best)
            # Recompute available and break to restart with new state
            existing = set(best)
            available = [x for x in range(1, 257) if x not in existing]
            # Continue trying more insertions
            insertions = []
            for v in available:
                for p in range(len(best) + 1):
                    if p > 0 and best[p-1] >= v:
                        continue
                    if p < len(best) and v >= best[p]:
                        continue
                    s = insertion_score(v, p)
                    insertions.append((s, v, p))
            insertions.sort(key=lambda t: t[0], reverse=True)
    
    # Try removing one element and reinserting elsewhere (to enable different insertions)
    for _ in range(150):
        if time.time() > time_limit:
            break
        
        i = random.randint(0, n - 1)
        val = best[i]
        new_seq = best[:i] + best[i+1:]
        
        for pos in range(len(new_seq) + 1):
            if pos == i:
                continue
            candidate = new_seq[:pos] + [val] + new_seq[pos:]
            
            if any(candidate[k] >= candidate[k+1] for k in range(len(candidate)-1)):
                continue
            
            if _is_valid_fast(candidate):
                # Check if this enables a better insertion
                existing2 = set(candidate)
                for v2 in range(1, 257):
                    if v2 in existing2:
                        continue
                    for pos2 in range(len(candidate) + 1):
                        if pos2 > 0 and candidate[pos2-1] >= v2:
                            continue
                        if pos2 < len(candidate) and v2 >= candidate[pos2]:
                            continue
                        cand2 = candidate[:pos2] + [v2] + candidate[pos2:]
                        if _is_valid_fast(cand2) and len(cand2) > best_len:
                            best = cand2
                            best_len = len(best)
                            n = len(best)
                            break
                    else:
                        continue
                    break
                break
    
    return best

def _multi_start_greedy(deadline):
    """Try multiple greedy strategies with different starting points and orderings."""
    best = []
    
    # Standard forward greedy
    seq = _greedy_forward_optimized()
    if len(seq) > len(best):
        best = seq
    
    # Reverse greedy
    seq = _greedy_reverse_optimized()
    if len(seq) > len(best):
        best = seq
    
    # Greedy with random perturbations
    for trial in range(10):
        if time.time() > deadline:
            break
        
        rand = random.Random(trial * 100 + 42)
        selected = []
        sums = set()
        last_val = 0
        
        for _ in range(256):
            if time.time() > deadline:
                break
            
            # Get valid candidates
            candidates = _get_valid_candidates(selected, sums, last_val)
            if not candidates:
                break
            
            # Score and pick from top with some randomness
            scored = [(x, _score_candidate(x, selected, sums) + rand.gauss(0, 1)) for x in candidates]
            scored.sort(key=lambda t: t[1], reverse=True)
            
            # Pick from top 3 randomly
            top_k = min(3, len(scored))
            x = scored[rand.randint(0, top_k-1)][0]
            
            # Add x
            new_sums = set(sums)
            total = x
            new_sums.add(total)
            for a in reversed(selected):
                total += a
                new_sums.add(total)
            
            selected.append(x)
            sums = new_sums
            last_val = x
        
        if len(selected) > len(best):
            best = selected
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s wall-clock
    
    best_seq = []
    
    # 1. Fast deterministic greedies and multi-start (very fast, ~5ms)
    seq_multi = _multi_start_greedy(start_time + 5.0)
    if len(seq_multi) > len(best_seq):
        best_seq = seq_multi
    
    # 2. DFS with structured branching (explore non-greedy paths)
    dfs_deadline = start_time + 35.0
    dfs_result = _branch_and_bound_dfs(0, dfs_deadline, 14, None)
    if len(dfs_result) > len(best_seq):
        best_seq = dfs_result
    
    # 3. Multiple randomized DFS runs with varied depths
    dfs_deadline2 = start_time + 60.0
    for seed in range(25):
        if time.time() > dfs_deadline2:
            break
        # Vary max_depth: some shallow (fast), some deep (thorough)
        if seed % 3 == 0:
            md = 8
        elif seed % 3 == 1:
            md = 12
        else:
            md = 16
        result = _dfs_with_randomization(seed * 17 + 3, dfs_deadline2, md)
        if len(result) > len(best_seq):
            best_seq = result
    
    # 4. Local improvement with remaining time
    local_deadline = start_time + 78.0
    improved = _local_improve(best_seq, local_deadline)
    if len(improved) > len(best_seq):
        best_seq = improved
    
    # 5. One more local pass if time allows
    if time.time() < local_deadline + 1:
        improved2 = _local_improve(best_seq, local_deadline + 1)
        if len(improved2) > len(best_seq):
            best_seq = improved2
    
    return best_seq
