"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 12

# EVOLVE-BLOCK-START

def _compute_threshold_bitmask(a_map, b_map, n):
    """
    Compute the exact shortest reset word length using bitmask-based BFS.
    Returns threshold length, or None if not synchronizing.
    """
    size = 1 << n
    full_mask = size - 1
    
    # Precompute transitions for all masks
    # img_a[mask] is the bitmask of states reachable from 'mask' via 'a'
    img_a = [0] * size
    img_b = [0] * size
    
    for mask in range(1, size):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        img_a[mask] = img_a[prev_mask] | (1 << a_map[idx])
        img_b[mask] = img_b[prev_mask] | (1 << b_map[idx])
        
    # BFS
    from collections import deque
    visited = [False] * size
    queue = deque()
    queue.append(full_mask)
    visited[full_mask] = True
    depth = 0
    
    while queue:
        size_q = len(queue)
        for _ in range(size_q):
            mask = queue.popleft()
            if mask & (mask - 1) == 0: # Power of 2, i.e., singleton
                return depth
            
            next_a = img_a[mask]
            next_b = img_b[mask]
            
            if not visited[next_a]:
                visited[next_a] = True
                queue.append(next_a)
            
            if not visited[next_b]:
                visited[next_b] = True
                queue.append(next_b)
                
        depth += 1
        
    return None

def _random_automaton(rng):
    return [
        [rng.randint(0, N - 1) for _ in range(N)],
        [rng.randint(0, N - 1) for _ in range(N)]
    ]

def _pawlik_automaton():
    # Standard Černý/Pawlik automaton
    # a: i -> i+1 mod N
    # b: 0 -> 0, i -> i-1 for i>0
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    return [a, b]

def construct():
    deadline = time.time() + 50  # Safety margin
    rng = random.Random(42)
    
    # Initialize with Pawlik, though it's weak (threshold N-1)
    # We might want a better starting point. 
    # Let's try a few random starts and pick the best one.
    
    best_trans = None
    best_thresh = -1
    
    # Try a few random initializations
    for _ in range(10):
        trans = _random_automaton(rng)
        thresh = _compute_threshold_bitmask(trans[0], trans[1], N)
        if thresh is not None and thresh > best_thresh:
            best_thresh = thresh
            best_trans = trans
            
    if best_trans is None:
        # Fallback to Pawlik if no random one synchronized
        best_trans = _pawlik_automaton()
        best_thresh = _compute_threshold_bitmask(best_trans[0], best_trans[1], N)
        if best_thresh is None:
            best_thresh = 0 # Should not happen
            
    # Simulated Annealing
    current_trans = [row[:] for row in best_trans]
    current_thresh = best_thresh
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Propose mutation
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        
        old_dest = current_trans[letter][state]
        new_dest = rng.randint(0, N - 1)
        
        if new_dest == old_dest:
            continue
            
        current_trans[letter][state] = new_dest
        
        new_thresh = _compute_threshold_bitmask(current_trans[0], current_trans[1], N)
        
        if new_thresh is None:
            # Not synchronizing, revert
            current_trans[letter][state] = old_dest
            continue
            
        # Acceptance criterion
        if new_thresh > current_thresh:
            # Always accept improvement
            current_thresh = new_thresh
            if new_thresh > best_thresh:
                best_thresh = new_thresh
                best_trans = [row[:] for row in current_trans]
        else:
            # Accept with probability exp((new - old) / T)
            # Note: new_thresh < current_thresh
            delta = new_thresh - current_thresh
            if delta < 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_thresh = new_thresh
                else:
                    # Revert
                    current_trans[letter][state] = old_dest
            else:
                # Should not happen if we handled improvement above, but just in case
                current_thresh = new_thresh
        
        T *= cooling
        
    return best_trans

# EVOLVE-BLOCK-END
