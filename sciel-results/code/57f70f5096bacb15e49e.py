"""
Structural change: model as longest path in a DAG of "valid prefix states".
Each node is a sequence prefix; an edge exists if appending a value preserves
strict increase, range [1,256], and distinct consecutive-block sums.
We run a BFS/DFS over this implicit graph with aggressive pruning and memoization.
Additionally, we use a different search strategy: iterative deepening with
branch-and-bound on the maximum possible remaining length.
"""
import time
import random
import sys
from collections import defaultdict

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _get_all_sums(seq):
    """Compute all consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _search_longest(deadline):
    """
    Depth-first search with branch-and-bound.
    State: (current_seq, sums_set, last_val)
    We try to extend by any value v > last_val, v <= 256, such that
    all new block sums (v, seq[-1]+v, seq[-2]+seq[-1]+v, ...) are not in sums_set.
    
    Pruning:
    1. If len(current_seq) + (256 - last_val) <= best_len, prune (optimistic upper bound)
    2. If time exceeds deadline, return
    
    We use a randomized ordering of candidate values to escape local minima.
    """
    best_seq = []
    best_len = 0
    
    def _try_append(seq, sums, last_val, v):
        """Check if v can be appended; return new_sums list or None."""
        new_sums = [v]
        total = v
        for i in range(len(seq) - 1, -1, -1):
            total += seq[i]
            if total in sums:
                return None
            new_sums.append(total)
        return new_sums
    
    def dfs(seq, sums, last_val, depth):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return False
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        # Upper bound: even if we pick every remaining value, can we beat best?
        # Max possible additional elements: 256 - last_val (if we could pick all)
        # But we need strictly increasing, so max add = 256 - last_val
        if n + (256 - last_val) <= best_len:
            return True  # Pruned
        
        # Generate candidate values: v in (last_val, 256]
        # Try smaller values first (more room for future extensions)
        candidates = []
        for v in range(last_val + 1, 257):
            new_sums = _try_append(seq, sums, last_val, v)
            if new_sums is not None:
                candidates.append((v, new_sums))
                # Early exit: if we have enough candidates, stop searching
                # to save time for deeper exploration
                if len(candidates) >= 15:
                    break
        
        if not candidates:
            return True
        
        # Limit branching factor: try at most 12 candidates per node
        # Prioritize smaller values for more remaining room
        if len(candidates) > 12:
            candidates = candidates[:12]
        
        for v, new_sums in candidates:
            new_seq = seq + [v]
            new_sums_set = sums.union(new_sums)
            dfs(new_seq, new_sums_set, v, depth + 1)
            if time.time() > deadline:
                return True
            # If we found a sequence longer than best, continue exploring
            # but limit further branching if we're already well above best
            if best_len >= n + 5:
                break
        
        return True
    
    # Start with empty sequence
    dfs([], set(), 0, 0)
    
    return best_seq

def _iterative_deepening(deadline):
    """
    Iterative deepening: try to find sequences of length L, L+1, ... 
    This helps focus the search on achieving a specific target length.
    """
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    # Try to improve one element at a time
    target = best_len + 1
    while time.time() < deadline - 2:
        found = _search_longest_with_target(target, time.time() + 5)
        if found and len(found) >= target:
            best_seq = found
            best_len = len(found)
            target = best_len + 1
        else:
            # Try a shorter target? Or break
            break
    
    return best_seq

def _search_longest_with_target(target_len, deadline):
    """
    DFS that stops as soon as we find a sequence of length >= target_len.
    """
    def _try_append(seq, sums, last_val, v):
        """Check if v can be appended; return new_sums list or None."""
        new_sums = [v]
        total = v
        for i in range(len(seq) - 1, -1, -1):
            total += seq[i]
            if total in sums:
                return None
            new_sums.append(total)
        return new_sums
    
    def dfs(seq, sums, last_val):
        nonlocal best_found
        if time.time() > deadline:
            return None
        
        if len(seq) >= target_len:
            return list(seq)
        
        n = len(seq)
        if n + (256 - last_val) < target_len:
            return None
        
        # Try candidates - smaller values first for more room
        for v in range(last_val + 1, 257):
            new_sums = _try_append(seq, sums, last_val, v)
            if new_sums is None:
                continue
            
            result = dfs(seq + [v], sums.union(new_sums), v)
            if result is not None:
                return result
            if time.time() > deadline:
                return None
        
        return None
    
    best_found = None
    result = dfs([], set(), 0)
    return result

def _beam_search_structural(deadline):
    """
    Beam search where each beam entry is a (seq, sums) pair.
    We keep the top-K partial sequences by potential (length + remaining_room).
    """
    # Start with greedy
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    beam = [([], set())]
    K = 1000  # beam width
    
    while time.time() < deadline - 1:
        new_beam = []
        seen_states = set()
        
        for seq, sums in beam:
            n = len(seq)
            if n > best_len:
                best_len = n
                best_seq = list(seq)
            
            last_val = seq[-1] if seq else 0
            
            # Generate extensions
            for v in range(last_val + 1, 257):
                new_sums_list = [v]
                total = v
                for i in range(n - 1, -1, -1):
                    total += seq[i]
                    new_sums_list.append(total)
                
                valid = True
                for s in new_sums_list:
                    if s in sums:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_seq = seq + [v]
                new_sums = sums.union(new_sums_list)
                
                # Deduplicate by (tuple(seq), frozenset(sums))
                key = (tuple(new_seq), frozenset(new_sums))
                if key in seen_states:
                    continue
                seen_states.add(key)
                
                new_beam.append((new_seq, new_sums))
            
            # Prune beam: keep top K by score
            if len(new_beam) > K:
                # Score: length * 10 + (256 - last_val)
                new_beam.sort(key=lambda x: (-len(x[0]), -(256 - (x[0][-1] if x[0] else 0))))
                new_beam = new_beam[:K]
        
        if not new_beam:
            break
        
        beam = new_beam
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Baseline
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    if time.time() > deadline - 5:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Try direct DFS search (most promising for this problem structure)
    try:
        dfs_result = _search_longest(deadline)
        if len(dfs_result) > best_len:
            best = dfs_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Try iterative deepening (focuses on achieving specific lengths)
    try:
        id_result = _iterative_deepening(deadline)
        if len(id_result) > best_len:
            best = id_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Try structural beam search with remaining time
    try:
        beam_result = _beam_search_structural(deadline)
        if len(beam_result) > best_len:
            best = beam_result
            best_len = len(best)
    except Exception:
        pass
    
    # Final verification
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
