import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 90s limit, but 60s is mentioned in contract? 
    # Contract says "Each evaluation must finish within 60 s". 
    # Directive says "KILLS ... after 90s". 
    # I will use 50s to be safe.
    deadline = start_time + 50.0
    
    best_score = float('inf')
    best_poly = None
    
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return np.count_nonzero(sq)

    # Helper to generate a random valid polynomial
    def random_poly():
        p = [0] * 41
        indices = random.sample(range(41), 12)
        for idx in indices:
            p[idx] = random.choice([-1, 1])
        # Ensure last coeff non-zero
        if p[40] == 0:
            nz_indices = [i for i in range(41) if p[i] != 0]
            k = random.choice(nz_indices)
            p[40] = p[k]
            p[k] = 0
        return p

    # Initialize with a few structured candidates
    candidates = []
    
    # 1. Arithmetic progression with signs (often good for cancellation)
    for _ in range(20):
        p = [0] * 41
        signs = [random.choice([-1, 1]) for _ in range(12)]
        for i in range(12):
            p[i] = signs[i]
        if p[40] == 0: # AP is 0..11, so 40 is 0. Fix.
             # Move one term to 40? Or just accept and fix in loop?
             # Better: generate AP ending at 40?
             pass
        # Let's generate APs that include 40
        start = random.randint(-29, 0)
        p = [0] * 41
        for i in range(12):
            idx = start + i
            if 0 <= idx < 41:
                p[idx] = random.choice([-1, 1])
        if p[40] == 0:
            # If 40 is not covered, move last term to 40
            nz_indices = [i for i in range(41) if p[i] != 0]
            if nz_indices:
                k = nz_indices[-1]
                p[40] = p[k]
                p[k] = 0
        candidates.append(p)

    # 2. Random sparse
    for _ in range(20):
        candidates.append(random_poly())

    # 3. Known good structure (if any)
    # The current best is 21. Let's try to find something better.
    
    # Evaluate initial candidates
    for p in candidates:
        nz_count = sum(1 for x in p if x != 0)
        if nz_count != 12: continue
        if p[40] == 0: continue
        if any(abs(x) > 64 for x in p): continue
        
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_poly = p[:]

    # Simulated Annealing
    current_poly = best_poly[:]
    current_score = best_score
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > 1000:
            # Restart
            current_poly = random_poly()
            current_score = count_nz_square(current_poly)
            if current_score < best_score:
                best_score = current_score
                best_poly = current_poly[:]
            no_improve = 0
            T = 1.0
            continue

        # Generate a neighbor
        neighbor = current_poly[:]
        move_type = random.random()
        
        if move_type < 0.6:
            # Change sign or value of a random non-zero term
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            idx = random.choice(nz_indices)
            old_val = neighbor[idx]
            # Try small changes
            new_val = old_val + random.choice([-2, -1, 1, 2])
            if new_val == 0:
                new_val = old_val
            elif abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
            else:
                # Sometimes try a completely new small value
                if random.random() < 0.1:
                    new_val = random.randint(-10, 10)
                    if new_val == 0: new_val = 1
            neighbor[idx] = new_val
        else:
            # Move a term
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            z_indices = [i for i in range(41) if neighbor[i] == 0]
            if not z_indices: continue
            
            src = random.choice(nz_indices)
            # If src is 40, we cannot move it away (would make p[40]=0)
            # Unless we move another term to 40? 
            # Simplest: don't move 40.
            if src == 40:
                continue
                
            dst = random.choice(z_indices)
            val = neighbor[src]
            neighbor[src] = 0
            neighbor[dst] = val
            
            # Check validity: if dst == 40, valid. If src != 40, p[40] unchanged (non-zero).
            # If src == 40, we skipped.
            # So p[40] is always non-zero.
            
        new_score = count_nz_square(neighbor)
        
        # Acceptance criterion
        if new_score < current_score:
            current_poly = neighbor
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_poly = current_poly[:]
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_poly = neighbor
                    current_score = new_score
            no_improve += 1
            T *= cooling

    return best_poly
