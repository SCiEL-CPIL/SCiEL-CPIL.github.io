"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def compute_threshold(trans_a, trans_b):
    """
    Compute the length of the shortest reset word for the automaton defined by
    trans_a and trans_b (each a list of N destination states).
    Returns -1 if the automaton is not synchronizing.
    Uses bitmask BFS on the subset automaton for speed.
    """
    FULL_MASK = (1 << N) - 1
    
    # Precompute subset transitions for both letters
    # next_a[mask] = resulting mask after applying letter 'a' to the subset represented by mask
    size = 1 << N
    next_a = [0] * size
    next_b = [0] * size
    
    for mask in range(1, size):
        lb = mask & -mask
        i = lb.bit_length() - 1
        prev_mask = mask ^ lb
        next_a[mask] = next_a[prev_mask] | (1 << trans_a[i])
        next_b[mask] = next_b[prev_mask] | (1 << trans_b[i])
    
    # BFS from the full set
    start = FULL_MASK
    visited = bytearray(size)
    visited[start] = 1
    queue = deque([(start, 0)])
    
    while queue:
        subset, dist = queue.popleft()
        if subset & (subset - 1) == 0:
            # It's a singleton
            return dist
        na = next_a[subset]
        if not visited[na]:
            visited[na] = 1
            queue.append((na, dist + 1))
        nb = next_b[subset]
        if not visited[nb]:
            visited[nb] = 1
            queue.append((nb, dist + 1))
    
    return -1

def _generate_cycle_with_random_map():
    """Generate an automaton where one letter is a cycle and the other is random."""
    a = [(i + 1) % N for i in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def _generate_random_automaton():
    """Generate a fully random automaton."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def _generate_star_automaton():
    """Star automaton: one letter maps everything to 0, other is a cycle."""
    a = [0] * N
    b = [(i + 1) % N for i in range(N)]
    return a, b

def _generate_known_good_automaton():
    """Return a known synchronizing automaton with decent reset length."""
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]  # cycle
    b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return a, b

def _simulated_annealing(start_a, start_b, max_iterations=50000, timeout=25):
    """Simulated annealing to improve reset length."""
    best_a = start_a[:]
    best_b = start_b[:]
    best_len = compute_threshold(best_a, best_b)
    
    if best_len < 0:
        return None
    
    start_time = time.time()
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    T_start = 5.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iterations))
    T = T_start
    
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Randomly modify one state of one letter
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        old_val = current_a[state] if letter == 0 else current_b[state]
        new_val = random.randint(0, N - 1)
        
        if new_val == old_val:
            T *= cooling
            continue
        
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
        
        new_len = compute_threshold(current_a, current_b)
        
        if new_len < 0:
            # Not synchronizing: revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
            T *= cooling
            continue
        
        delta = new_len - current_len
        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
            current_len = new_len
            if current_len > best_len:
                best_a = current_a[:]
                best_b = current_b[:]
                best_len = current_len
        else:
            # Revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
        
        T *= cooling
    
    return best_a, best_b, best_len

def construct():
    """Construct a synchronizing automaton maximizing reset word length."""
    deadline = time.time() + 80  # Leave 10s safety margin
    
    best_result = None
    best_score = -1
    
    # Strategy 1: Try known good automaton
    a, b = _generate_known_good_automaton()
    length = compute_threshold(a, b)
    if length > best_score:
        best_score = length
        best_result = (a[:], b[:])
    
    # Strategy 2: Try various structured generators
    generators = [
        _generate_cycle_with_random_map,
        _generate_star_automaton,
    ]
    
    for gen in generators:
        if time.time() > deadline:
            break
        a, b = gen()
        length = compute_threshold(a, b)
        if length > best_score:
            best_score = length
            best_result = (a[:], b[:])
    
    # Strategy 3: Simulated annealing from multiple starting points
    if time.time() < deadline - 30:
        # Start from the best known
        if best_result is not None:
            result = _simulated_annealing(best_result[0], best_result[1], max_iterations=30000, timeout=15)
            if result is not None:
                sa, sb, slen = result
                if slen > best_score:
                    best_score = slen
                    best_result = (sa, sb)
        
        # Try a few random starting points
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            a = [random.randint(0, N - 1) for _ in range(N)]
            b = [random.randint(0, N - 1) for _ in range(N)]
            
            length = compute_threshold(a, b)
            if length < 0:
                continue
            
            if length > best_score:
                best_score = length
                best_result = (a[:], b[:])
            
            # Simulated annealing from this point
            result = _simulated_annealing(a, b, max_iterations=15000, timeout=8)
            if result is not None:
                ha, hb, hlen = result
                if hlen > best_score:
                    best_score = hlen
                    best_result = (ha, hb)
    
    # Strategy 4: Quick random search if time remains
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            a = [random.randint(0, N - 1) for _ in range(N)]
            b = [random.randint(0, N - 1) for _ in range(N)]
            
            length = compute_threshold(a, b)
            if length > best_score:
                best_score = length
                best_result = (a[:], b[:])
    
    if best_result is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_result
# EVOLVE-BLOCK-END
