import time
import math
import sys

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _beam_search_beam(max_val=256, beam_width=100, deadline=None):
    """
    Beam search using the prefix-sum representation.
    
    State: (d_list, P_list, diffs_set)
    - d_list: the sequence values (strictly increasing)
    - P_list: prefix sums, P[0]=0, P[k] = sum of first k elements
    - diffs_set: all pairwise differences P[j]-P[i] for i<j
    
    A new value d must satisfy:
    1. d > d_list[-1] (strictly increasing)
    2. d <= max_val
    3. For all existing P[i], (P_last + d) - P[i] not in diffs_set
    
    Score: prefer sequences with smaller last element (more room)
    and smaller P_last (more room in difference space).
    """
    if deadline is None:
        deadline = time.time() + 45
    
    # Each beam entry: (score, d_list, P_list, diffs_set)
    # Score: we want to minimize last_d to keep options open
    # But also want to minimize P_last for more difference space
    # Use composite: (last_d, P_last) as sort key
    
    beam = [([], [0], set())]
    best_seq = []
    best_len = 0
    
    max_depth = max_val  # absolute upper bound on length
    
    for depth in range(max_depth):
        if time.time() > deadline - 1:
            break
        
        new_beam = []
        for d_list, P_list, diffs_set in beam:
            if time.time() > deadline - 1:
                break
            
            last_d = d_list[-1] if d_list else 0
            P_last = P_list[-1]
            
            for new_d in range(last_d + 1, max_val + 1):
                new_P = P_last + new_d
                
                # Check all new differences
                valid = True
                new_diffs = []
                for i in range(len(P_list)):
                    diff = new_P - P_list[i]
                    if diff in diffs_set:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                new_d_list = d_list + [new_d]
                new_P_list = P_list + [new_P]
                new_diffs_set = diffs_set | set(new_diffs)
                
                new_beam.append((new_d_list, new_P_list, new_diffs_set))
                
                # Update best
                if len(new_d_list) > best_len:
                    best_len = len(new_d_list)
                    best_seq = new_d_list
        
        if not new_beam:
            break
        
        # Sort by score: prefer smaller last_d, then smaller P_last
        # This keeps sequences that have more room to grow
        new_beam.sort(key=lambda x: (x[0][-1], x[1][-1]))
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # If all sequences in beam have same length, we might be stuck
        # but let's continue
    
    return best_seq

def _iterative_beam(max_val=256, deadline=None):
    """
    Iterative beam search with increasing beam width.
    Start with small beam to get a quick answer, then widen.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    best_seq = _greedy_forward([])
    best_len = len(best_seq)
    
    for width in [5, 20, 50, 100, 200, 500, 1000]:
        if time.time() > deadline - 3:
            break
        
        result = _beam_search_beam(max_val, beam_width=width, deadline=deadline)
        if len(result) > best_len:
            best_seq = result
            best_len = len(result)
    
    return best_seq

def _dfs_with_pruning(max_val=256, deadline=None):
    """
    DFS with strong pruning using the prefix-sum distinct-differences constraint.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    best_seq = []
    best_len = 0
    
    def _upper_bound(d_list, P_list):
        """Upper bound on achievable length from current state."""
        n_cur = len(d_list)
        if not d_list:
            last_d = 0
        else:
            last_d = d_list[-1]
        
        if last_d >= max_val:
            return n_cur
        
        max_rem = max_val - last_d
        # Sum of all possible remaining d values
        sum_rem = (max_val * (max_val + 1) // 2) - (last_d * (last_d + 1) // 2)
        P_total_max = P_list[-1] + sum_rem
        
        # Need m*(m-1)/2 <= P_total_max for m total P points
        # m = n_cur + 1 + rem, and rem <= max_rem
        # Also m*(m-1)/2 <= P_total_max
        m_max_possible = n_cur + 1 + max_rem
        m_bound = (1 + math.isqrt(1 + 8 * P_total_max)) // 2
        if m_bound < m_max_possible:
            m_max_possible = m_bound
        return m_max_possible - 1
    
    def dfs(d_list, P_list, diffs_set):
        nonlocal best_seq, best_len
        
        if time.time() > deadline - 1:
            return
        
        n_cur = len(d_list)
        
        if n_cur > best_len:
            best_len = n_cur
            best_seq = list(d_list)
        
        ub = _upper_bound(d_list, P_list)
        if ub <= best_len:
            return
        
        last_d = d_list[-1] if d_list else 0
        P_last = P_list[-1]
        
        # Try candidates in increasing order (smallest first for more room)
        for new_d in range(last_d + 1, max_val + 1):
            if time.time() > deadline - 1:
                return
            
            new_P = P_last + new_d
            
            valid = True
            new_diffs = []
            for i in range(len(P_list)):
                diff = new_P - P_list[i]
                if diff in diffs_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            dfs(d_list + [new_d], P_list + [new_P], diffs_set | set(new_diffs))
            
            if best_len >= ub:
                break
    
    dfs([], [0], set())
    return best_seq

def _hybrid_search(max_val=256, deadline=None):
    """
    Combine beam search with local improvement.
    After finding a good sequence via beam search, try to extend it
    or modify it to find longer sequences.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    # Phase 1: Beam search
    result = _iterative_beam(max_val, deadline=deadline)
    best = result
    best_len = len(best)
    
    # Phase 2: Try DFS from the best sequence found
    if time.time() < deadline - 5 and best_len < 100:
        # Try to extend the best sequence
        extended = _greedy_forward(best, max_val)
        if len(extended) > best_len:
            best = extended
            best_len = len(best)
    
    # Phase 3: Full DFS if time remains
    if time.time() < deadline - 5:
        dfs_result = _dfs_with_pruning(max_val, deadline=deadline)
        if len(dfs_result) > best_len:
            best = dfs_result
            best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Quick greedy for baseline
    greedy_seq = _greedy_forward([])
    best = greedy_seq
    best_len = len(best)
    
    # 2. Iterative beam search (main strategy)
    if time.time() < deadline - 10:
        beam_result = _iterative_beam(max_val=256, deadline=deadline)
        if len(beam_result) > best_len:
            best = beam_result
            best_len = len(best)
    
    # 3. Hybrid search with DFS extension
    if time.time() < deadline - 5:
        hybrid_result = _hybrid_search(max_val=256, deadline=deadline)
        if len(hybrid_result) > best_len:
            best = hybrid_result
            best_len = len(best)
    
    # Final validation
    if best and not _is_valid_full(best):
        best = _greedy_forward([])
    
    if not best:
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return best
