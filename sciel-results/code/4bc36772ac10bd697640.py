"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_reset_length(a_trans, b_trans):
    """Computes the exact shortest reset word length using BFS on the subset automaton with bitmask optimization."""
    N = len(a_trans)
    full_mask = (1 << N) - 1
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    
    # Precompute transition masks for each letter
    a_masks = [0] * N
    b_masks = [0] * N
    for i in range(N):
        a_masks[i] = 1 << a_trans[i]
        b_masks[i] = 1 << b_trans[i]
    
    # Single state check
    single_states = set()
    for i in range(N):
        single_states.add(a_trans[i])
        single_states.add(b_trans[i])
    
    visited = {full_mask}
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        # Check if mask is a single state
        if mask & (mask - 1) == 0:
            return dist
        
        # Apply 'a'
        next_a = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            next_a |= a_masks[idx]
            m ^= lsb
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        
        # Apply 'b'
        next_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            next_b |= b_masks[idx]
            m ^= lsb
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a_trans, b_trans):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return
        score = _bfs_reset_length(a_trans, b_trans)
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]
    
    # Strategy 1: Pinwheel constructions with different parameters
    # Classic pinwheel: a = cycle, b = shift with one exception
    for step in range(1, N):
        a_trans = [(i + step) % N for i in range(N)]
        for target in range(N):
            b_trans = [0 if i == 0 else i - 1 for i in range(N)]
            b_trans[0] = target
            try_automaton(a_trans, b_trans)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 2: Systematic enumeration of structured b_trans
    # Try all possible b_trans where b is a function with limited structure
    # Focus on b_trans that create "almost cycles" with one or two exceptions
    
    # 2a: b_trans as permutation with one fixed point moved
    a_trans_base = [(i + 1) % N for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b_trans = [(k + 1) % N for k in range(N)]
            b_trans[i] = j
            try_automaton(a_trans_base, b_trans)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # 2b: b_trans with two modified positions
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                for l in range(N):
                    if l == i or l == j:
                        continue
                    b_trans = [(m + 1) % N for m in range(N)]
                    b_trans[i] = j
                    b_trans[k] = l
                    try_automaton(a_trans_base, b_trans)
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 3: Known high-threshold constructions from literature
    # The "Rank-2" construction for n states has reset word length ~n^2/4
    # For n=12, this gives ~36, but we want to push higher
    
    # Try a_trans as a single N-cycle, b_trans designed to slowly collapse
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
        
        # b_trans that maps most states forward but one state to create slow collapse
        for special_state in range(N):
            for special_target in range(N):
                b_trans = [(i + 1) % N for i in range(N)]
                b_trans[special_state] = special_target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 4: More diverse a_trans structures
    # Try a_trans that are not simple cycles
    
    # 4a: a_trans with a 2-cycle and rest as a long path
    for start_2cycle in range(N):
        for second_2cycle in range(N):
            if second_2cycle == start_2cycle:
                continue
            a_trans = list(range(N))
            a_trans[start_2cycle] = second_2cycle
            a_trans[second_2cycle] = start_2cycle
            # Make rest a cycle
            remaining = [i for i in range(N) if i not in (start_2cycle, second_2cycle)]
            for idx, state in enumerate(remaining):
                if idx < len(remaining) - 1:
                    a_trans[state] = remaining[idx + 1]
                else:
                    a_trans[state] = remaining[0]
            
            for b_target in range(N):
                b_trans = [0 if i == 0 else i - 1 for i in range(N)]
                b_trans[0] = b_target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 5: Enhanced random search with multiple restarts and better mutations
    if time.time() < deadline:
        random.seed(42)  # Deterministic for reproducibility
        
        # Multiple restarts from different starting points
        start_configs = []
        if best_a is not None:
            start_configs.append((best_a[:], best_b[:]))
        
        # Add some structured starting points
        start_configs.append(([(i + 1) % N for i in range(N)], [0 if i == 0 else i - 1 for i in range(N)]))
        start_configs.append(([(i + 2) % N for i in range(N)], [0 if i == 0 else i - 1 for i in range(N)]))
        start_configs.append(([(i + 5) % N for i in range(N)], [0 if i == 0 else i - 1 for i in range(N)]))
        
        for start_a, start_b in start_configs:
            if time.time() > deadline - 5.0:
                break
            
            current_a = start_a[:]
            current_b = start_b[:]
            current_score = _bfs_reset_length(current_a, current_b)
            
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
            
            # Local search with temperature-like acceptance
            no_improve_count = 0
            max_no_improve = 50
            
            while time.time() < deadline - 3.0 and no_improve_count < max_no_improve:
                # Choose mutation type: single change, swap, or cycle modification
                mut_type = random.random()
                
                if mut_type < 0.4:
                    # Single destination change in a
                    i = random.randint(0, N - 1)
                    old_dest = current_a[i]
                    new_dest = random.randint(0, N - 1)
                    if new_dest == old_dest:
                        continue
                    current_a[i] = new_dest
                    new_score = _bfs_reset_length(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                    else:
                        current_a[i] = old_dest
                        no_improve_count += 1
                        
                elif mut_type < 0.8:
                    # Single destination change in b
                    i = random.randint(0, N - 1)
                    old_dest = current_b[i]
                    new_dest = random.randint(0, N - 1)
                    if new_dest == old_dest:
                        continue
                    current_b[i] = new_dest
                    new_score = _bfs_reset_length(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                    else:
                        current_b[i] = old_dest
                        no_improve_count += 1
                        
                else:
                    # Swap two destinations in a or b
                    if random.random() < 0.5:
                        i, j = random.sample(range(N), 2)
                        current_a[i], current_a[j] = current_a[j], current_a[i]
                        new_score = _bfs_reset_length(current_a, current_b)
                        if new_score > current_score:
                            current_score = new_score
                            no_improve_count = 0
                            if current_score > best_score:
                                best_score = current_score
                                best_a = current_a[:]
                                best_b = current_b[:]
                        else:
                            current_a[i], current_a[j] = current_a[j], current_a[i]
                            no_improve_count += 1
                    else:
                        i, j = random.sample(range(N), 2)
                        current_b[i], current_b[j] = current_b[j], current_b[i]
                        new_score = _bfs_reset_length(current_a, current_b)
                        if new_score > current_score:
                            current_score = new_score
                            no_improve_count = 0
                            if current_score > best_score:
                                best_score = current_score
                                best_a = current_a[:]
                                best_b = current_b[:]
                        else:
                            current_b[i], current_b[j] = current_b[j], current_b[i]
                            no_improve_count += 1
    
    # Final polish: try small perturbations around the best solution
    if best_a is not None and time.time() < deadline - 1.0:
        for _ in range(200):
            if time.time() > deadline - 0.5:
                break
            trial_a = best_a[:]
            trial_b = best_b[:]
            # Change 1-2 random destinations
            num_changes = random.randint(1, 2)
            for _ in range(num_changes):
                if random.random() < 0.5:
                    i = random.randint(0, N - 1)
                    trial_a[i] = random.randint(0, N - 1)
                else:
                    i = random.randint(0, N - 1)
                    trial_b[i] = random.randint(0, N - 1)
            score = _bfs_reset_length(trial_a, trial_b)
            if score > best_score:
                best_score = score
                best_a = trial_a
                best_b = trial_b
    
    # Fallback
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
