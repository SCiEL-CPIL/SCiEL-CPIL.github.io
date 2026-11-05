import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_diffs(s):
    """Compute the set of all d > 0 such that a, a+d, a+2d are all in s."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            if a + 2 * d in s_set:
                diffs.add(d)
    return diffs

def _score(s):
    """Score = number of distinct d > 0 with a 3-term AP."""
    return len(_compute_diffs(s))

def _delta_score(s, remove_elem, add_elem):
    """
    Compute the change in score when removing `remove_elem` and adding `add_elem` from set s.
    Returns (delta, new_diffs_set).
    """
    s_set = set(s)
    old_diffs = _compute_diffs(s_set)
    
    # Remove element
    s_set.discard(remove_elem)
    # Add element
    s_set.add(add_elem)
    
    new_diffs = _compute_diffs(s_set)
    delta = len(new_diffs) - len(old_diffs)
    return delta, new_diffs

def _try_move(s, remove_elem, add_elem, current_score):
    """
    Try moving remove_elem out and add_elem in. Returns (new_score, new_set) or (None, None).
    """
    if add_elem in s:
        return None, None
    
    new_set = set(s)
    new_set.discard(remove_elem)
    new_set.add(add_elem)
    
    if len(new_set) != 32:
        return None, None
    
    new_diffs = _compute_diffs(new_set)
    new_score = len(new_diffs)
    return new_score, new_set

def _generate_initial_set():
    """Generate a good initial set using multiple strategies."""
    strategies = []
    
    # Strategy 1: Pure arithmetic progression
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100]:
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            s = set(a + i * d for i in range(32))
            if all(1 <= x <= 512 for x in s):
                strategies.append(s)
    
    # Strategy 2: Two interleaved APs
    for _ in range(10):
        d1 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        d2 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
        s = set()
        a1 = random.randint(1, max(1, 512 - 15 * d1))
        for i in range(16):
            s.add(a1 + i * d1)
        a2 = random.randint(1, max(1, 512 - 15 * d2))
        for i in range(16):
            val = a2 + i * d2
            if val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 3: Random
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # Strategy 4: Dense block
    base = random.randint(1, 481)
    s = set(range(base, base + 32))
    if max(s) <= 512:
        strategies.append(s)
    
    # Strategy 5: Multi-scale structure
    for _ in range(5):
        s = set()
        # Pick a base structure
        num_blocks = random.randint(2, 4)
        block_size = 32 // num_blocks
        for b in range(num_blocks):
            block_base = random.randint(1, 512 - block_size * 10)
            for k in range(block_size):
                val = block_base + k * random.choice([1, 2, 3])
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 6: Quadratic-like structure
    for _ in range(3):
        s = set()
        a = random.randint(1, 3)
        b = random.randint(1, 10)
        c = random.randint(1, 50)
        for i in range(32):
            val = a * i * i + b * i + c
            if 1 <= val <= 512 and val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Pick the best
    best = None
    best_sc = -1
    for s in strategies:
        if len(s) != 32:
            continue
        sc = _score(s)
        if sc > best_sc:
            best_sc = sc
            best = s
    
    return best if best else set(random.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = None
    best_score = 0
    
    # Phase 1: Multiple restarts with diverse initializations
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial set
        current_set = _generate_initial_set()
        current_score = _score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Simulated Annealing
        T = 30.0
        cooling = 0.9985
        min_T = 0.3
        stuck_count = 0
        max_stuck = 2000
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            # Generate a neighbor
            elem_out = random.choice(list(current_set))
            
            # Try several candidate positions for elem_in
            best_move = None
            best_move_score = -1
            
            for _ in range(5):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                new_set = set(current_set)
                new_set.discard(elem_out)
                new_set.add(elem_in)
                
                if len(new_set) != 32:
                    continue
                
                new_score = _score(new_set)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (elem_out, elem_in, new_set, new_score)
            
            if best_move is None:
                stuck_count += 1
                continue
            
            elem_out, elem_in, new_set, new_score = best_move
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                stuck_count = 0
                no_improve = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                # Accept with probability exp(delta/T)
                if delta < -5:
                    stuck_count += 1
                    no_improve += 1
                elif random.random() < np.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
                    stuck_count = 0
                    no_improve += 1
                else:
                    stuck_count += 1
                    no_improve += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > max_stuck:
                # Perturbation
                for _ in range(3):
                    e_out = random.choice(list(current_set))
                    e_in = random.randint(1, 512)
                    if e_in not in current_set:
                        current_set = set(current_set)
                        current_set.discard(e_out)
                        current_set.add(e_in)
                        current_score = _score(current_set)
                stuck_count = 0
                T = max(T, 20.0)
            
            if no_improve > max_no_improve:
                break
    
    # Phase 2: Focused local search on best solution
    if best_set is not None and time.time() < deadline - 2:
        current_set = set(best_set)
        current_score = _score(current_set)
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.1
        stuck_count = 0
        
        while time.time() < deadline - 1:
            elem_out = random.choice(list(current_set))
            
            # Try multiple candidates
            best_move = None
            best_move_score = -1
            
            for _ in range(8):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                new_set = set(current_set)
                new_set.discard(elem_out)
                new_set.add(elem_in)
                
                if len(new_set) != 32:
                    continue
                
                new_score = _score(new_set)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (elem_in, new_set, new_score)
            
            if best_move is None:
                stuck_count += 1
                continue
            
            elem_in, new_set, new_score = best_move
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                stuck_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
                    stuck_count = 0
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > 3000:
                break
    
    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END
