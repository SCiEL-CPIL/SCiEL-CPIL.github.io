"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_reset_length_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses integer bitmasks for subsets to improve performance.
    """
    N = len(a_trans)
    start_mask = (1 << N) - 1  # All states present
    
    # Quick check: if a or b is constant, it's not a cycle, but we need to check if it resets
    # Actually, if a is constant, applying 'a' once makes the set size 1.
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
        
    visited = {start_mask}
    # Queue stores (mask, distance)
    queue = deque([(start_mask, 0)])
    
    # Precompute destination masks for each state for a and b?
    # dest_mask_a[i] = 1 << a_trans[i]
    dest_a = [1 << a_trans[i] for i in range(N)]
    dest_b = [1 << b_trans[i] for i in range(N)]
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Generate next states
        # Apply 'a'
        next_mask_a = 0
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & (-m)
            idx = (lsb.bit_length() - 1)
            next_mask_a |= dest_a[idx]
            m ^= lsb
            
        if next_mask_a not in visited:
            visited.add(next_mask_a)
            queue.append((next_mask_a, dist + 1))
            
        # Apply 'b'
        next_mask_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = (lsb.bit_length() - 1)
            next_mask_b |= dest_b[idx]
            m ^= lsb
            
        if next_mask_b not in visited:
            visited.add(next_mask_b)
            queue.append((next_mask_b, dist + 1))
            
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
        score = _bfs_reset_length_fast(a_trans, b_trans)
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]
            return True
        return False

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
            
    # Strategy 3: Known high-threshold constructions
    # Try a_trans as a single N-cycle, b_trans designed to slowly collapse
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
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
    # 4a: a_trans with a 2-cycle and rest as a long path
    for start_2cycle in range(N):
        for second_2cycle in range(N):
            if second_2cycle == start_2cycle:
                continue
            a_trans = list(range(N))
            a_trans[start_2cycle] = second_2cycle
            a_trans[second_2cycle] = start_2cycle
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

    # Strategy 5: Enhanced Local Search
    # Uses swaps and random flips, with restarts if stuck.
    if time.time() < deadline:
        random.seed(42)
        
        if best_a is None:
            current_a = [(i + 1) % N for i in range(N)]
            current_b = [0 if i == 0 else i - 1 for i in range(N)]
        else:
            current_a = best_a[:]
            current_b = best_b[:]
        
        current_score = best_score
        
        # Local search loop
        no_improve_count = 0
        max_no_improve = 500  # Restart if stuck for this many iterations
        
        while time.time() < deadline - 2.0:
            if no_improve_count > max_no_improve:
                # Perturb current best and restart local search
                # Randomly change 2-3 transitions in a or b
                for _ in range(random.randint(2, 3)):
                    if random.random() < 0.5:
                        i = random.randint(0, N - 1)
                        current_a[i] = random.randint(0, N - 1)
                    else:
                        i = random.randint(0, N - 1)
                        current_b[i] = random.randint(0, N - 1)
                
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    # Revert to best known
                    current_a = best_a[:]
                    current_b = best_b[:]
                    current_score = best_score
                
                no_improve_count = 0
                max_no_improve = random.randint(500, 1000) # Randomize restart threshold
                
            # Mutation choice: 50% Flip, 50% Swap
            if random.random() < 0.5:
                # Flip: change destination of one state
                if random.random() < 0.5:
                    i = random.randint(0, N - 1)
                    old_dest = current_a[i]
                    new_dest = random.randint(0, N - 1)
                    if new_dest == old_dest:
                        continue
                    current_a[i] = new_dest
                    new_score = _bfs_reset_length_fast(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                        no_improve_count = 0
                    else:
                        current_a[i] = old_dest
                        no_improve_count += 1
                else:
                    i = random.randint(0, N - 1)
                    old_dest = current_b[i]
                    new_dest = random.randint(0, N - 1)
                    if new_dest == old_dest:
                        continue
                    current_b[i] = new_dest
                    new_score = _bfs_reset_length_fast(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                        no_improve_count = 0
                    else:
                        current_b[i] = old_dest
                        no_improve_count += 1
            else:
                # Swap: swap destinations of two states
                if random.random() < 0.5:
                    i, j = random.sample(range(N), 2)
                    current_a[i], current_a[j] = current_a[j], current_a[i]
                    new_score = _bfs_reset_length_fast(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                        no_improve_count = 0
                    else:
                        current_a[i], current_a[j] = current_a[j], current_a[i]
                        no_improve_count += 1
                else:
                    i, j = random.sample(range(N), 2)
                    current_b[i], current_b[j] = current_b[j], current_b[i]
                    new_score = _bfs_reset_length_fast(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_a = current_a[:]
                            best_b = current_b[:]
                        no_improve_count = 0
                    else:
                        current_b[i], current_b[j] = current_b[j], current_b[i]
                        no_improve_count += 1

    # Fallback
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
