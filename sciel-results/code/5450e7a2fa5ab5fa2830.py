"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def _reset_length_fast(a, b):
    """
    Compute shortest reset word length via BFS on subsets using bitsets.
    Returns -1 if not synchronizing.
    """
    # Start with all states set
    start_mask = (1 << N) - 1
    
    # Precompute transition masks for each state for a and b
    # trans_a[state] = bit mask of destination? No, we need subset transformation.
    # For a subset S (mask), S.a = union of {1 << a[i] for i in S}
    
    # To speed up, we can precompute the image of every possible subset?
    # 2^12 = 4096 subsets. This is small enough to precompute or cache.
    
    # Let's use a cache for the next state of a subset under a and b.
    # cache_a[mask] = next_mask
    # cache_b[mask] = next_mask
    
    # We will compute these on the fly and store them in local dicts to avoid global state issues.
    
    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    # Optimization: If any letter is constant, length is 1.
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1
        
    # Precompute destination bits for each state
    dest_a = [1 << x for x in a]
    dest_b = [1 << x for x in b]
    
    # Cache for transitions
    trans_a_cache = {}
    trans_b_cache = {}
    
    def get_next(mask, trans_cache, dests):
        if mask in trans_cache:
            return trans_cache[mask]
        
        # Calculate union of destinations for all states in mask
        next_mask = 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            next_mask |= dests[idx]
            m ^= lsb
            
        trans_cache[mask] = next_mask
        return next_mask

    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton (only one bit set)
        if mask & (mask - 1) == 0:
            return dist
            
        # Apply a
        next_a = get_next(mask, trans_a_cache, dest_a)
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply b
        next_b = get_next(mask, trans_b_cache, dest_b)
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def _generate_initial():
    """
    Generate a good initial candidate.
    Uses a known construction for long reset words:
    A 'cycle' automaton where a is a cycle and b has a specific structure.
    """
    a = list(range(N))
    b = [0] * N
    
    # A known good pattern for N=12 is related to the Černý bound N(N-1)/2 = 66.
    # We start with a structure close to the "Pinsker" or "Černý" examples.
    # Let's try the standard "cycle + one reset" but modified to be harder.
    # Actually, a good starting point is a random permutation for a, and b mapping most to 0, some to 1.
    
    random.shuffle(a)
    for i in range(N):
        if random.random() < 0.5:
            b[i] = 0
        else:
            b[i] = 1
            
    # Ensure it's synchronizing. If not, fallback.
    if _reset_length_fast(a, b) == -1:
        # Fallback: Cycle + Constant
        a = list(range(N))
        b = [0] * N
        
    return a, b

def construct():
    """Construct a synchronizing automaton with 12 states maximizing reset threshold."""
    # Safety margin: Evaluator kills at 90s, but task says 60s.
    # We will target 50s to be safe.
    deadline = time.time() + 50.0
    
    best_a, best_b = _generate_initial()
    best_len = _reset_length_fast(best_a, best_b)
    
    if best_len == -1:
        # Absolute fallback
        best_a = list(range(N))
        best_b = [0] * N
        best_len = 1
        
    current_a, current_b = best_a[:], best_b[:]
    current_len = best_len
    
    # Local Search with Random Restarts
    # We perform hill climbing. If stuck, we restart from a mutated version of best.
    
    while time.time() < deadline:
        # 1. Local Search (Hill Climbing)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomly shuffle order of states to avoid bias
            states = list(range(N))
            random.shuffle(states)
            
            for i in states:
                if time.time() > deadline:
                    break
                
                # Try changing a[i]
                orig_a = current_a[i]
                for val in range(N):
                    if val == orig_a:
                        continue
                    current_a[i] = val
                    new_len = _reset_length_fast(current_a, current_b)
                    if new_len > current_len:
                        current_len = new_len
                        improved = True
                        if new_len > best_len:
                            best_len = new_len
                            best_a = current_a[:]
                            best_b = current_b[:]
                    else:
                        current_a[i] = orig_a
                        
                # Try changing b[i]
                orig_b = current_b[i]
                for val in range(N):
                    if val == orig_b:
                        continue
                    current_b[i] = val
                    new_len = _reset_length_fast(current_a, current_b)
                    if new_len > current_len:
                        current_len = new_len
                        improved = True
                        if new_len > best_len:
                            best_len = new_len
                            best_a = current_a[:]
                            best_b = current_b[:]
                    else:
                        current_b[i] = orig_b

        # 2. If no improvement, try a random restart from the best solution
        # Mutate the best solution slightly to escape local optima
        if time.time() > deadline:
            break
            
        # Random Restart: Take best_a/b, apply a few random changes
        restart_a = best_a[:]
        restart_b = best_b[:]
        
        # Mutate 1-3 transitions
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                restart_a[idx] = random.randint(0, N - 1)
            else:
                idx = random.randint(0, N - 1)
                restart_b[idx] = random.randint(0, N - 1)
                
        restart_len = _reset_length_fast(restart_a, restart_b)
        
        if restart_len != -1:
            current_a, current_b = restart_a, restart_b
            current_len = restart_len
        else:
            # If not synchronizing, try a more drastic reset
            current_a, current_b = _generate_initial()
            current_len = _reset_length_fast(current_a, current_b)
            if current_len == -1:
                current_a = best_a[:]
                current_b = best_b[:]
                current_len = best_len

    return best_a, best_b
# EVOLVE-BLOCK-END
