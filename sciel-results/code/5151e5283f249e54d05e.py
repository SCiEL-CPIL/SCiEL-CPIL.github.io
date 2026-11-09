import time
import random

def _sq_nonzero_count(d):
    """Calculate nonzero terms in square of dict-represented poly."""
    if not d:
        return 0
    sq = {}
    for i, ci in d.items():
        for j, cj in d.items():
            k = i + j
            sq[k] = sq.get(k, 0) + ci * cj
    return sum(1 for v in sq.values() if v != 0)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Initialize with Thue-Morse
    tm = _thue_morse(12)
    best_d = {i: tm[i] for i in range(12)}
    best_score = _sq_nonzero_count(best_d)
    
    # Local Search: Sign Flipping
    current_d = best_d.copy()
    
    # We will perform several rounds of sign flipping
    # If a flip improves, accept. If not, revert.
    # To escape local minima, we might accept worse moves with low prob (Simulated Annealing)
    # but given time limit, simple hill climbing + random restarts is safer.
    
    for _ in range(100): # Max 100 restarts/rounds
        if time.time() > deadline:
            break
            
        current_d = best_d.copy()
        current_score = best_score
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in list(current_d.keys()):
                if time.time() > deadline:
                    break
                orig = current_d[i]
                current_d[i] = -orig
                new_score = _sq_nonzero_count(current_d)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_d = current_d.copy()
                else:
                    current_d[i] = orig
        
        # If we improved, continue from there.
        # If not, we are stuck. Try a random perturbation?
        # For now, just break if no improvement in this round?
        # No, the loop above runs until no single flip improves.
        # So we are at a local minimum.
        # Let's do a random restart with a different initial pattern?
        # Or just break?
        # Given the structure, Thue-Morse is likely near optimal.
        # Let's add a few random sign flips to current_d to escape, then hill climb again.
        
        # Escape: Flip 2 random signs
        if len(current_d) >= 2:
            keys = list(current_d.keys())
            i1, i2 = random.sample(keys, 2)
            current_d[i1] = -current_d[i1]
            current_d[i2] = -current_d[i2]
            
            # Hill Climb again
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in list(current_d.keys()):
                    if time.time() > deadline:
                        break
                    orig = current_d[i]
                    current_d[i] = -orig
                    new_score = _sq_nonzero_count(current_d)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_d = current_d.copy()
                    else:
                        current_d[i] = orig

    # Convert to list
    max_deg = max(best_d.keys()) if best_d else 0
    result = [0] * (max_deg + 1)
    for i, c in best_d.items():
        result[i] = c
        
    # Check constraints
    if sum(1 for c in result if c != 0) != 12:
        return [1] * 12
    if len(result) > 41:
        return [1] * 12
    if any(abs(c) > 64 for c in result):
        return [1] * 12
    if result[-1] == 0:
        return [1] * 12
        
    return result
