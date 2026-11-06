"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_reset_length(a_trans, b_trans):
    """Computes the exact shortest reset word length using BFS on the subset automaton."""
    N = len(a_trans)
    start = tuple(range(N))
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        state, dist = queue.popleft()
        if len(state) == 1:
            return dist
        # Apply 'a'
        next_a = tuple(sorted(set(a_trans[i] for i in state)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        # Apply 'b'
        next_b = tuple(sorted(set(b_trans[i] for i in state)))
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
    
    # Strategy 5: Random search with structured mutations
    if time.time() < deadline:
        random.seed(42)  # Deterministic for reproducibility
        
        # Start from best found so far
        if best_a is None:
            current_a = [(i + 1) % N for i in range(N)]
            current_b = [0 if i == 0 else i - 1 for i in range(N)]
        else:
            current_a = best_a[:]
            current_b = best_b[:]
        
        current_score = best_score
        
        while time.time() < deadline - 2.0:
            # Mutate: pick random state and change destination in a or b
            if random.random() < 0.5:
                i = random.randint(0, N - 1)
                old_dest = current_a[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                current_a[i] = new_dest
                new_score = _bfs_reset_length(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    current_a[i] = old_dest
            else:
                i = random.randint(0, N - 1)
                old_dest = current_b[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                current_b[i] = new_dest
                new_score = _bfs_reset_length(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    current_b[i] = old_dest
    
    # Fallback
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
