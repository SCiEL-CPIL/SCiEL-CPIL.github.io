"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a snake in Q_9 using Beam Search.
    This approach maintains a set of the most promising partial paths
    at each depth, pruning less promising ones based on a heuristic score.
    """
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    # Precompute neighbors
    neighbors = [[] for _ in range(TOTAL)]
    for v in range(TOTAL):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    # Beam search parameters
    beam_width = 500
    rng = random.Random(42)
    
    # Initial state: path=[0], blocked=[0]*TOTAL
    # blocked[u] is the number of vertices in path[:-1] adjacent to u.
    # Initially path is [0], so path[:-1] is empty. blocked is all 0.
    
    # State: (path, blocked_bytearray)
    # Using bytearray for efficiency
    
    current_beam = [([0], bytearray(TOTAL))]
    best_path = [0]
    best_len = 1
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        next_beam_candidates = []
        
        for path, blocked in current_beam:
            last = path[-1]
            
            # Find valid candidates
            for u in neighbors[last]:
                if blocked[u] == 0:
                    # Candidate u is valid
                    
                    # Update blocked counts
                    # The vertex 'last' becomes part of the "previous" set for the next step.
                    # So we add its neighbors to the blocked count.
                    new_blocked = blocked[:]
                    for w in neighbors[last]:
                        new_blocked[w] += 1
                        
                    # Score: number of valid moves from u
                    score = 0
                    for v in neighbors[u]:
                        if new_blocked[v] == 0:
                            score += 1
                    
                    # Add small random noise to break ties and encourage exploration
                    score += rng.uniform(0, 0.5)
                    
                    next_beam_candidates.append((path + [u], new_blocked, score))
        
        if not next_beam_candidates:
            break
            
        # Sort by score (descending) and keep top beam_width
        next_beam_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Keep top K
        new_beam = []
        for i in range(min(beam_width, len(next_beam_candidates))):
            new_beam.append((next_beam_candidates[i][0], next_beam_candidates[i][1]))
            
        current_beam = new_beam
        
        # Update best
        if len(current_beam[0][0]) > best_len:
            best_len = len(current_beam[0][0])
            best_path = current_beam[0][0]
            
        # If beam is empty, stop
        if not current_beam:
            break
        
    return best_path

# EVOLVE-BLOCK-END
