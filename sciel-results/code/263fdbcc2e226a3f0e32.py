import time
import random
import numpy as np

N = 9
NUM_VERTICES = 1 << N
TIME_LIMIT = 118.0  # Safety margin before 150s kill

# Precompute neighbors for all vertices
NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    best_path = []
    
    def time_exceeded():
        return time.time() - start_time > TIME_LIMIT
    
    def update_best(path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]
    
    def greedy_extend(path, used, blocked):
        """Greedy extension with random tie-breaking."""
        while True:
            if time_exceeded():
                return
            cur = path[-1]
            valid_moves = []
            for nb in NEIGHBORS[cur]:
                if not used[nb] and not blocked[nb]:
                    valid_moves.append(nb)
            if not valid_moves:
                return
            # Pick a random valid move
            nb = random.choice(valid_moves)
            used[nb] = 1
            # Update blocked
            for nb_cur in NEIGHBORS[cur]:
                if not blocked[nb_cur]:
                    blocked[nb_cur] = 1
            path.append(nb)
            update_best(path)
    
    def beam_search(beam_width=200):
        """Beam search with diverse paths."""
        if time_exceeded():
            return
        
        # Initialize beam with several diverse starting paths
        beam = []
        
        # Try different initial prefixes
        # Symmetry: fix start at 0, first move bit 0
        # So path starts [0, 1]
        # Second move can be any bit except 0
        for b2 in range(1, N):
            v2 = 1 ^ (1 << b2)
            path = [0, 1, v2]
            used = bytearray(NUM_VERTICES)
            used[0] = 1
            used[1] = 1
            used[v2] = 1
            blocked = bytearray(NUM_VERTICES)
            for nb in NEIGHBORS[0]:
                blocked[nb] = 1
            for nb in NEIGHBORS[1]:
                blocked[nb] = 1
            beam.append((path, used, blocked))
        
        # Extend each path in the beam
        while beam and not time_exceeded():
            new_beam = []
            for path, used, blocked in beam:
                if time_exceeded():
                    break
                cur = path[-1]
                valid_moves = []
                for nb in NEIGHBORS[cur]:
                    if not used[nb] and not blocked[nb]:
                        valid_moves.append(nb)
                if not valid_moves:
                    continue
                # Add all valid moves to new beam (or sample if too many)
                for nb in valid_moves:
                    new_path = path + [nb]
                    new_used = bytearray(used)
                    new_used[nb] = 1
                    new_blocked = bytearray(blocked)
                    for nb_cur in NEIGHBORS[cur]:
                        if not new_blocked[nb_cur]:
                            new_blocked[nb_cur] = 1
                    new_beam.append((new_path, new_used, new_blocked))
                    update_best(new_path)
                
                # Limit beam size
                if len(new_beam) > beam_width * 2:
                    # Keep only the longest paths
                    new_beam.sort(key=lambda x: len(x[0]), reverse=True)
                    new_beam = new_beam[:beam_width]
                    break
            beam = new_beam
    
    def beam_search_efficient(beam_width=500):
        """More efficient beam search using in-place modification and undo."""
        if time_exceeded():
            return
        
        # We'll use a list of (path_length, path, used, blocked) and manage beam size
        # To avoid copying, we'll use a different approach: 
        # Keep the beam as a list of states, and for each state, try all moves,
        # then select the top beam_width states by path length (or some heuristic).
        
        # Initialize with diverse starts
        beam = []
        for b2 in range(1, N):
            v2 = 1 ^ (1 << b2)
            path = [0, 1, v2]
            used = bytearray(NUM_VERTICES)
            used[0] = 1
            used[1] = 1
            used[v2] = 1
            blocked = bytearray(NUM_VERTICES)
            for nb in NEIGHBORS[0]:
                blocked[nb] = 1
            for nb in NEIGHBORS[1]:
                blocked[nb] = 1
            beam.append((path, used, blocked))
        
        for iteration in range(500):  # Max iterations
            if time_exceeded():
                break
            
            new_beam = []
            for path, used, blocked in beam:
                if time_exceeded():
                    break
                cur = path[-1]
                valid_moves = []
                for nb in NEIGHBORS[cur]:
                    if not used[nb] and not blocked[nb]:
                        valid_moves.append(nb)
                if not valid_moves:
                    continue
                for nb in valid_moves:
                    new_path = path + [nb]
                    new_used = bytearray(used)
                    new_used[nb] = 1
                    new_blocked = bytearray(blocked)
                    for nb_cur in NEIGHBORS[cur]:
                        if not new_blocked[nb_cur]:
                            new_blocked[nb_cur] = 1
                    new_beam.append((new_path, new_used, new_blocked))
                    update_best(new_path)
            
            if not new_beam:
                break
            
            # Select top beam_width by length
            new_beam.sort(key=lambda x: len(x[0]), reverse=True)
            beam = new_beam[:beam_width]
    
    # Run multiple strategies
    # 1. Greedy with random restarts
    for restart in range(200):
        if time_exceeded():
            break
        # Random start
        start = random.randint(0, NUM_VERTICES - 1)
        path = [start]
        used = bytearray(NUM_VERTICES)
        used[start] = 1
        blocked = bytearray(NUM_VERTICES)
        greedy_extend(path, used, blocked)
    
    # 2. Beam search with different widths
    for width in [100, 200, 500]:
        if time_exceeded():
            break
        beam_search_efficient(width)
    
    return best_path
