import time
from collections import deque

N = 12

def _bfs_reset_length(a, b):
    """
    Computes the exact shortest reset word length for a 12-state automaton.
    Uses bitmask representation for subsets.
    """
    # Precompute transitions for all 2^N subsets? 
    # 2^12 = 4096 subsets.
    # For each subset, we need to know where 'a' and 'b' take it.
    # We can compute on the fly or precompute.
    # On the fly is fast enough for a few thousand evaluations.
    
    full_set = (1 << N) - 1
    
    # Check if full_set is already a singleton (not possible for N=12)
    if full_set & (full_set - 1) == 0:
        return 0
        
    # BFS
    # dist[subset] = distance from full_set
    # We store visited subsets
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        curr = queue.popleft()
        dist = visited[curr]
        
        # Check if curr is a singleton
        if curr & (curr - 1) == 0:
            return dist
            
        # Apply 'a'
        next_a = 0
        # Iterate through bits in curr
        temp = curr
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            next_a |= (1 << a[idx])
            temp ^= lsb
            
        if next_a not in visited:
            if next_a & (next_a - 1) == 0:
                return dist + 1
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        # Apply 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            next_b |= (1 << b[idx])
            temp ^= lsb
            
        if next_b not in visited:
            if next_b & (next_b - 1) == 0:
                return dist + 1
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    # If not synchronizing (should not happen for our constructions)
    return -1

def construct():
    # Start with a known high-threshold construction: Vorel (Halving)
    # a: cycle
    a = [(i + 1) % N for i in range(N)]
    # b: halving
    b = [i // 2 for i in range(N)]
    
    current_best = _bfs_reset_length(a, b)
    best_b = b[:]
    
    # Hill Climbing
    # Time budget: 80 seconds
    start_time = time.time()
    timeout = 80.0
    
    # Local search: try changing one transition of b at a time
    # We can also try swapping transitions or other moves, but single-step is good.
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(N):
            if time.time() - start_time > timeout:
                break
            old_val = b[i]
            for j in range(N):
                if j == old_val:
                    continue
                b[i] = j
                # Ensure 'a' and 'b' are valid (they always are)
                # Check if it's a valid automaton (always is)
                # Check if it synchronizes
                val = _bfs_reset_length(a, b)
                if val > current_best:
                    current_best = val
                    best_b = b[:]
                    improved = True
                    # Do not break, keep looking for even better in this iteration?
                    # Or break to restart from new best?
                    # Restarting is better to explore new neighborhood.
                    # But we are in a loop. Let's just update best_b and break inner loop?
                    # No, let's just update and continue to see if another move is even better from this state.
                    # Actually, standard hill climbing: take the best move.
                    # Here we take the first improvement.
                    # Let's stick to first improvement to be fast.
                    break
            # If we broke due to improvement, b[i] is the new val.
            # If not, b[i] is restored? No, if j==old_val we skipped.
            # If we didn't find improvement, b[i] is the last j tried.
            # We need to restore b[i] if we didn't improve?
            # Actually, if we didn't improve, we should revert b[i] to old_val.
            if not improved:
                b[i] = old_val
            if improved:
                # Restart the outer loop to search from the new best
                break
                
        if not improved:
            # No single-step improvement found.
            # Try a 2-step move? Or accept and stop?
            # For N=12, single step might get stuck.
            # But Vorel is already 66. We want 121.
            # Let's try a few random restarts if stuck?
            # Or just break.
            # Given the time limit, if we can't improve single-step, we might be at a local max.
            # But 66 is far from 121.
            # Let's try a different initial seed if stuck?
            # For now, break to avoid infinite loop.
            break
            
    return [a, best_b]
