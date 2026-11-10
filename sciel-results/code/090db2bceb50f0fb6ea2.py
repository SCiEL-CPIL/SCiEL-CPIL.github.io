"""Improved program: DFS with backtracking + local search polish."""
import time
import random
from collections import defaultdict

DEADLINE = 85.0  # safety margin before 90s kill

def _block_sums(seq):
    """Return set of all nonempty consecutive block sums for seq."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _is_valid(seq):
    """Check if seq is strictly increasing in [1,256] and all block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    # Check all consecutive block sums are distinct
    prefix = [0]
    for x in seq:
        prefix.append(prefix[-1] + x)
    seen = set()
    for i in range(n):
        for j in range(i+1, n+1):
            s = prefix[j] - prefix[i]
            if s in seen:
                return False
            seen.add(s)
    return True

def _count_conflicts(seq):
    """Return number of duplicate block sums (0 if all distinct)."""
    n = len(seq)
    prefix = [0]
    for x in seq:
        prefix.append(prefix[-1] + x)
    seen = set()
    conflicts = 0
    for i in range(n):
        for j in range(i+1, n+1):
            s = prefix[j] - prefix[i]
            if s in seen:
                conflicts += 1
            else:
                seen.add(s)
    return conflicts

def _try_extend(seq, used, deadline):
    """Try to extend seq by one element. Return (new_seq, new_used) or None."""
    if time.time() > deadline:
        return None
    n = len(seq)
    if n == 0:
        start = 1
    else:
        start = seq[-1] + 1
    # Compute existing block sums
    prefix = [0]
    for x in seq:
        prefix.append(prefix[-1] + x)
    existing_sums = set()
    for i in range(n):
        for j in range(i+1, n+1):
            existing_sums.add(prefix[j] - prefix[i])
    
    for x in range(start, 257):
        if x in used:
            continue
        # Check new block sums ending at x
        new_sums = []
        total = 0
        for i in range(n, -1, -1):
            total += seq[i] if i < n else 0
            # Actually let me recompute properly
            pass
        # Recompute: new blocks are seq[i..n-1] + x for i in 0..n-1, and just x
        tail = x
        ok = True
        for i in range(n-1, -1, -1):
            tail = seq[i] + tail if i < n-1 else x
            # Wait, this is getting confused. Let me redo.
            break
        # New consecutive blocks that include x:
        # seq[n-1]+x, seq[n-2]+seq[n-1]+x, ..., seq[0]+...+seq[n-1]+x, and x itself
        new_tail_sums = []
        s = x
        new_tail_sums.append(s)
        for i in range(n-1, -1, -1):
            s += seq[i]
            new_tail_sums.append(s)
        
        for s in new_tail_sums:
            if s in existing_sums:
                ok = False
                break
        if not ok:
            continue
        # Also check new_tail_sums don't conflict with each other (they can't since strictly increasing seq means strictly increasing sums)
        # Actually they are strictly increasing so no conflict among themselves
        new_used = used | {x}
        return seq + [x], new_used
    return None

def _dfs_extend(seq, used, deadline):
    """Recursively try to extend seq. Returns longest valid extension found."""
    if time.time() > deadline:
        return seq
    result = _try_extend(seq, used, deadline)
    if result is None:
        return seq
    new_seq, new_used = result
    # Try to extend further
    deeper = _dfs_extend(new_seq, new_used, deadline)
    return deeper

def _local_search_polish(seq, deadline):
    """Try to improve seq by local search: swap, insert, delete elements."""
    if time.time() > deadline:
        return seq
    best = list(seq)
    best_conflicts = _count_conflicts(best)
    
    if best_conflicts == 0:
        # Try to extend
        extended = _dfs_extend(best, set(best), deadline)
        if len(extended) > len(best):
            best = extended
        return best
    
    # Try local moves
    for _ in range(200):
        if time.time() > deadline:
            break
        n = len(best)
        if n == 0:
            break
        move_type = random.choice(['swap', 'insert', 'delete', 'adjust'])
        trial = list(best)
        
        if move_type == 'swap' and n >= 2:
            i, j = random.sample(range(n), 2)
            trial[i], trial[j] = trial[j], trial[i]
            # Must remain strictly increasing
            valid = all(trial[k] < trial[k+1] for k in range(len(trial)-1))
            if not valid:
                continue
        elif move_type == 'insert' and n < 256:
            # Insert a new value
            pos = random.randint(0, n)
            if pos == 0:
                lo, hi = 1, trial[0]-1 if n > 0 else 256
            elif pos == n:
                lo, hi = trial[-1]+1, 256
            else:
                lo, hi = trial[pos-1]+1, trial[pos]-1
            if lo > hi:
                continue
            val = random.randint(lo, hi)
            trial.insert(pos, val)
        elif move_type == 'delete' and n > 1:
            pos = random.randint(0, n-1)
            trial.pop(pos)
        elif move_type == 'adjust' and n >= 1:
            idx = random.randint(0, n-1)
            old = trial[idx]
            if idx == 0:
                lo, hi = 1, (trial[1]-1 if n > 1 else 256)
            elif idx == n-1:
                lo, hi = (trial[idx-1]+1), 256
            else:
                lo, hi = trial[idx-1]+1, trial[idx+1]-1
            if lo > hi:
                continue
            new_val = random.randint(lo, hi)
            if new_val == old:
                continue
            trial[idx] = new_val
        
        c = _count_conflicts(trial)
        if c < best_conflicts or (c == best_conflicts and len(trial) > len(best)):
            best = trial
            best_conflicts = c
            if best_conflicts == 0:
                break
    
    # Final polish: try to extend if valid
    if _is_valid(best):
        extended = _dfs_extend(best, set(best), deadline)
        if len(extended) > len(best):
            best = extended
    
    return best

def _greedy_construct(deadline):
    """Original greedy approach as baseline."""
    values = []
    sums = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _dfs_from_start(deadline):
    """DFS starting from empty sequence, trying to build longest valid sequence."""
    # Use the greedy result as a starting point, then try to improve
    greedy = _greedy_construct(deadline)
    
    # Try different starting points with DFS
    best = greedy
    best_len = len(greedy)
    
    # Try to extend greedy
    extended = _dfs_extend(greedy, set(greedy), deadline)
    if len(extended) > best_len:
        best = extended
        best_len = len(extended)
    
    # Try local search polish
    polished = _local_search_polish(best, deadline)
    if len(polished) > best_len:
        best = polished
        best_len = len(polished)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # First, get a good baseline with greedy
    baseline = _greedy_construct(deadline)
    
    # Try DFS-based extension
    result = _dfs_from_start(deadline)
    
    # If time remains, try multiple random restarts with local search
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            # Create a random perturbation of the best found
            trial = list(result)
            if len(trial) > 0:
                # Randomly remove some elements and try to rebuild
                n = len(trial)
                keep = random.randint(max(1, n-3), n)
                indices = random.sample(range(n), keep)
                trial = [trial[i] for i in sorted(indices)]
                # Ensure strictly increasing
                valid_trial = []
                for x in trial:
                    if not valid_trial or x > valid_trial[-1]:
                        valid_trial.append(x)
                trial = valid_trial
                if not trial:
                    continue
                # Extend
                extended = _dfs_extend(trial, set(trial), deadline)
                if len(extended) > len(result):
                    result = extended
                # Polish
                polished = _local_search_polish(extended, deadline)
                if len(polished) > len(result):
                    result = polished
    
    # Final validation
    if not _is_valid(result):
        # Fallback to greedy
        result = baseline
    
    return result
