import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Beam Search Parameters
    BEAM_WIDTH = 30
    MAX_STEPS = 200 # Safety, though time limit will stop it
    
    # Initial Beam: K random starts
    # To increase diversity, we can pick starts with different properties, but random is fine.
    beam = []
    for _ in range(BEAM_WIDTH):
        v0 = random.randint(0, MAX_VERTS - 1)
        path = [v0]
        used = np.zeros(MAX_VERTS, dtype=bool)
        used[v0] = True
        deg = np.zeros(MAX_VERTS, dtype=np.int8)
        for w in neighbors[v0]:
            deg[w] += 1
        beam.append((path, used, deg))
    
    if not beam:
        return [0]
        
    # Update best
    best_len = max(len(p[0]) for p in beam)
    best_path = max(beam, key=lambda x: len(x[0]))[0]
    
    step = 0
    while time.time() - start_time < DEADLINE and step < MAX_STEPS:
        candidates = []
        
        for path, used, deg in beam:
            last_v = path[-1]
            # Find valid moves
            valid_moves = []
            for w in neighbors[last_v]:
                if not used[w] and deg[w] == 1:
                    # Check onward moves for scoring
                    onward = 0
                    for x in neighbors[w]:
                        if not used[x] and deg[x] == 0:
                            onward += 1
                    valid_moves.append((w, onward))
            
            if not valid_moves:
                # Dead end for this path. 
                # We can try to "pivot" or just discard. 
                # In beam search, we just discard if we have better candidates.
                continue
                
            # Sort by onward moves (Warnsdorff)
            valid_moves.sort(key=lambda x: x[1])
            
            # Take top few moves from this path to add to candidates
            # We can take all, but to keep candidate pool manageable, maybe top 3?
            # With Beam=30 and 9 neighbors, max candidates = 30*9=270. This is small.
            for w, onward in valid_moves:
                # Create new state
                new_path = path + [w]
                new_used = used.copy()
                new_used[w] = True
                new_deg = deg.copy()
                # Update deg for neighbors of w
                for x in neighbors[w]:
                    new_deg[x] += 1
                
                # Score:
                # 1. Length (implicit in step)
                # 2. Warnsdorff (onward)
                # 3. Randomness
                score = -onward + random.random() * 0.5
                
                candidates.append((score, new_path, new_used, new_deg))
        
        if not candidates:
            break
            
        # Sort candidates by score (ascending, since we used -onward)
        candidates.sort(key=lambda x: x[0])
        
        # Select top BEAM_WIDTH
        new_beam = []
        for _, p, u, d in candidates[:BEAM_WIDTH]:
            new_beam.append((p, u, d))
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
                
        beam = new_beam
        step += 1
        
        # Early stopping if all paths in beam are same length and no improvement?
        # Not necessary, time limit handles it.
        
    return best_path
