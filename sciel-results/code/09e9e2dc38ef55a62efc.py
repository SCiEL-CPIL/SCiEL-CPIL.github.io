"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_reset_length_fast(a_trans, b_trans):
    """Fast BFS on subset automaton using precomputed transition tables."""
    N = len(a_trans)
    full_mask = (1 << N) - 1
    
    # Check if either letter is a constant (immediate reset to 1 state)
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    
    # Precompute single-state transition masks
    a_single = [1 << a_trans[i] for i in range(N)]
    b_single = [1 << b_trans[i] for i in range(N)]
    
    # BFS with precomputed subset transitions
    # We'll compute transitions on-the-fly but cache them
    subset_trans_cache = {}
    
    def apply_letter(mask, single_masks):
        """Apply a letter to a subset (bitmask)."""
        result = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            result |= single_masks[idx]
            m ^= lsb
        return result
    
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        # Try 'a'
        next_a = apply_letter(mask, a_single)
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        
        # Try 'b'
        next_b = apply_letter(mask, b_single)
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    
    return -1  # Not synchronizing

def _bfs_with_precomputed(a_trans, b_trans):
    """BFS with fully precomputed subset transition table for repeated use."""
    N = len(a_trans)
    full_mask = (1 << N) - 1
    
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    
    a_single = [1 << a_trans[i] for i in range(N)]
    b_single = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transitions for all reachable subsets
    # Use BFS to discover reachable subsets and compute their transitions
    trans_a = {}
    trans_b = {}
    
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        # Compute and store transitions
        next_a = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            next_a |= a_single[idx]
            m ^= lsb
        trans_a[mask] = next_a
        
        next_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            next_b |= b_single[idx]
            m ^= lsb
        trans_b[mask] = next_b
        
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin before 60s limit (90s evaluator limit)
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Cache for BFS results to avoid recomputation
    bfs_cache = {}
    
    def try_automaton(a_trans, b_trans):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return
        key = (tuple(a_trans), tuple(b_trans))
        if key in bfs_cache:
            score = bfs_cache[key]
        else:
            score = _bfs_reset_length_fast(a_trans, b_trans)
            bfs_cache[key] = score
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]
    
    # Strategy 1: Systematic pinwheel enumeration
    # Classic pinwheel: a = cycle by step, b = reverse cycle with one exception
    for step in range(1, N):
        a_trans = [(i + step) % N for i in range(N)]
        for target in range(N):
            b_trans = [(i - 1) % N for i in range(N)]
            b_trans[0] = target
            try_automaton(a_trans, b_trans)
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # Strategy 2: Rank-2 constructions - a is a cycle, b collapses rank-2 subsets slowly
    # For n states, rank-2 constructions can achieve ~n^2/4 reset length
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
        # b_trans: map most states to create slow collapse
        for special_state in range(N):
            for special_target in range(N):
                b_trans = [(i + 1) % N for i in range(N)]
                b_trans[special_state] = special_target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # Strategy 3: Two-modification b_trans on base cycle
    a_trans_base = [(i + 1) % N for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b_trans = [(k + 1) % N for k in range(N)]
            b_trans[i] = j
            try_automaton(a_trans_base, b_trans)
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # Strategy 4: Diverse a_trans structures with systematic b_trans
    # 4a: a_trans as permutation with one 2-cycle
    for s1 in range(N):
        for s2 in range(N):
            if s2 <= s1:
                continue
            a_trans = list(range(N))
            a_trans[s1] = s2
            a_trans[s2] = s1
            # Complete a_trans as a permutation (fill remaining as cycle)
            remaining = [i for i in range(N) if i not in (s1, s2)]
            if remaining:
                for idx, state in enumerate(remaining):
                    if idx < len(remaining) - 1:
                        a_trans[state] = remaining[idx + 1]
                    else:
                        a_trans[state] = remaining[0]
            
            for b_target in range(N):
                b_trans = [(i - 1) % N for i in range(N)]
                b_trans[0] = b_target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # Strategy 5: Local search from best solution and diverse starting points
    if time.time() < deadline - 8.0:
        random.seed(12345)
        
        # Starting configurations
        start_configs = []
        if best_a is not None:
            start_configs.append((best_a[:], best_b[:]))
        
        # Structured starting points
        for step in [1, 2, 3, 5, 7, 11]:
            a_start = [(i + step) % N for i in range(N)]
            b_start = [(i - 1) % N for i in range(N)]
            start_configs.append((a_start, b_start))
        
        for start_a, start_b in start_configs:
            if time.time() > deadline - 3.0:
                break
            
            current_a = start_a[:]
            current_b = start_b[:]
            current_score = _bfs_reset_length_fast(current_a, current_b)
            
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
            
            # Iterative local search
            no_improve = 0
            max_no_improve = 80
            
            while time.time() < deadline - 2.0 and no_improve < max_no_improve:
                # Random mutation
                if random.random() < 0.5:
                    # Mutate a
                    i = random.randint(0, N - 1)
                    old = current_a[i]
                    current_a[i] = random.randint(0, N - 1)
                    new_score = _bfs_reset_length_fast(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                    else:
                        current_a[i] = old
                        no_improve += 1
                else:
                    # Mutate b
                    i = random.randint(0, N - 1)
                    old = current_b[i]
                    current_b[i] = random.randint(0, N - 1)
                    new_score = _bfs_reset_length_fast(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                    else:
                        current_b[i] = old
                        no_improve += 1
    
    # Final polish: targeted perturbations around best
    if best_a is not None and time.time() < deadline - 1.0:
        for _ in range(300):
            if time.time() > deadline - 0.5:
                break
            trial_a = best_a[:]
            trial_b = best_b[:]
            # Change 1-2 random positions
            n_changes = random.randint(1, 2)
            for _ in range(n_changes):
                if random.random() < 0.5:
                    trial_a[random.randint(0, N - 1)] = random.randint(0, N - 1)
                else:
                    trial_b[random.randint(0, N - 1)] = random.randint(0, N - 1)
            score = _bfs_reset_length_fast(trial_a, trial_b)
            if score > best_score:
                best_score = score
                best_a = trial_a
                best_b = trial_b
    
    # Fallback
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
