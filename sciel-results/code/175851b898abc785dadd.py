"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random

def _bfs_reset_length(a_trans, b_trans):
    """Computes the exact shortest reset word length using BFS on the subset automaton."""
    N = len(a_trans)
    start = tuple(range(N))
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    visited = {start}
    queue = [(start, 0)]
    while queue:
        state, dist = queue.pop(0)
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
    return -1 # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Initialize with a known synchronizing structure (Pinwheel-like)
    # a: cycle, b: shift down
    a_trans = [(i + 1) % N for i in range(N)]
    b_trans = [0 if i == 0 else i - 1 for i in range(N)]
    
    best_score = _bfs_reset_length(a_trans, b_trans)
    best_a = a_trans[:]
    best_b = b_trans[:]
    
    if best_score < 0:
        # Fallback to simple synchronizing
        return [[(i + 1) % N for i in range(N)], [0] * N]

    # Local search: mutate b_trans to increase reset length
    current_a = a_trans[:]
    current_b = b_trans[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Mutate b_trans: pick a random state and change its destination
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
            # Revert if it broke synchronization or didn't improve
            current_b[i] = old_dest
            
    return [best_a, best_b]
# EVOLVE-BLOCK-END
