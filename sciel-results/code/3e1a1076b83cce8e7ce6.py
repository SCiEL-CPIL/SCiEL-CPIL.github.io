"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    # Precompute masks for "vertex and its neighbors"
    # mask[v] is a 512-bit integer where bit v and bits of neighbors of v are 1
    masks = [0] * (1 << N)
    for v in range(1 << N):
        m = 1 << v
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        masks[v] = m
    
    def greedy_extend_heuristic(start, rng, max_time):
        """
        Greedy extension with 2-step lookahead heuristic.
        """
        path = [start]
        blocked = 0
        cur = start
        
        while time.time() < max_time:
            # Find valid next vertices: neighbors of cur not in blocked
            valid_candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if not (blocked >> v) & 1:
                    valid_candidates.append(v)
            
            if not valid_candidates:
                break
            
            if len(valid_candidates) == 1:
                next_v = valid_candidates[0]
            else:
                # Heuristic: Pick the candidate that allows the most moves in the next step
                best_v = None
                best_score = -1
                # Shuffle to break ties randomly
                rng.shuffle(valid_candidates)
                for v in valid_candidates:
                    # Count valid moves from v
                    # Next blocked will be blocked | masks[cur]
                    new_blocked = blocked | masks[cur]
                    moves = 0
                    for b in range(N):
                        nv = v ^ (1 << b)
                        if not (new_blocked >> nv) & 1:
                            moves += 1
                    if moves > best_score:
                        best_score = moves
                        best_v = v
                
                next_v = best_v
            
            # Update state
            path.append(next_v)
            # Add current vertex and its neighbors to blocked set
            blocked |= masks[cur]
            cur = next_v
        
        return path
    
    # Multiple restarts with different random seeds
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 1)
        
        # Start from a random vertex
        start = rng.randint(0, (1 << N) - 1)
        
        # Do a heuristic greedy extension
        # Limit time per restart to allow more restarts
        # If we use heuristic, paths might be longer, so fewer restarts needed?
        # Let's give each restart a small time slice, e.g., 10ms, or run until dead end.
        # Running until dead end is better if dead ends are fast.
        # But heuristic is slower per step.
        # Let's just run until dead end, but cap the total restarts.
        path = greedy_extend_heuristic(start, rng, deadline - 1.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Early termination if we've found a very long snake
        if best_len >= 190:
            break
    
    if best_path is None:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
