import numpy as np
import random
import time
import math

def construct():
    base = 51975
    # Get all divisors > 1 of base
    divisors = sorted([d for d in range(2, base + 1) if base % d == 0])
    
    # Precompute indices for each divisor and residue? 
    # That would be sum(m) for all m, which is huge.
    # Instead, we maintain a cover_count array and update it incrementally.
    
    # State:
    # active: dict {modulus: residue}
    # cover_count: np array of length base
    
    # Initial state: Start with a few small moduli to get a baseline
    # A known good covering often uses small moduli.
    # Let's start with a random subset of the first 20 divisors.
    initial_divs = divisors[:20]
    active = {m: random.randint(0, m - 1) for m in initial_divs}
    
    cover_count = np.zeros(base, dtype=np.int32)
    for m, r in active.items():
        cover_count[r::m] += 1
        
    current_uncovered = np.sum(cover_count == 0)
    best_uncovered = current_uncovered
    best_active = {k: v for k, v in active.items()}
    
    start_time = time.time()
    time_limit = 105.0 # 15s buffer
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    # Precompute list of all divisors for random selection
    all_divs = divisors
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T # Or break if we want to stop, but let's keep going with low T for fine tuning
            
        # Choose a random move
        # 1. Change residue of an existing active modulus
        # 2. Add a new modulus
        # 3. Remove an existing modulus
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Change residue
            if not active:
                continue
            m = random.choice(list(active.keys()))
            old_r = active[m]
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta
            # Remove old, add new
            # We need to check overlaps.
            # old_slice = slice(old_r, base, m)
            # new_slice = slice(new_r, base, m)
            
            # Count how many positions become uncovered (1->0) and covered (0->1)
            # Note: old_slice and new_slice are disjoint if old_r != new_r.
            
            # Positions in old_slice: cover_count decreases by 1
            # Positions in new_slice: cover_count increases by 1
            
            # We can't easily calculate delta without touching the array if we want to be fast,
            # but we can estimate or just apply and check.
            # Given the time limit, applying and checking is O(N/m).
            # Let's try to calculate delta efficiently.
            
            # old_vals = cover_count[old_r::m]
            # new_vals = cover_count[new_r::m]
            
            # lost = sum(1 for v in old_vals if v == 1)
            # gained = sum(1 for v in new_vals if v == 0)
            
            # This is still O(N/m). With N=51975 and small m, this is fast.
            
            # Let's just apply and measure? No, we need to reject bad moves often.
            # Let's compute delta.
            
            old_vals = cover_count[old_r::m]
            new_vals = cover_count[new_r::m]
            
            # Careful: if we modify cover_count, we must revert if rejected.
            # But old_r and new_r are disjoint, so we can just look at current values.
            
            lost = np.sum(old_vals == 1)
            gained = np.sum(new_vals == 0)
            
            delta_uncovered = lost - gained
            
            if delta_uncovered < 0:
                # Improvement
                cover_count[old_r::m] -= 1
                cover_count[new_r::m] += 1
                active[m] = new_r
                current_uncovered += delta_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_active = {k: v for k, v in active.items()}
            else:
                # Accept with probability exp(-delta/T)
                if delta_uncovered == 0:
                    # Neutral move, always accept to explore?
                    cover_count[old_r::m] -= 1
                    cover_count[new_r::m] += 1
                    active[m] = new_r
                else:
                    prob = math.exp(-delta_uncovered / T)
                    if random.random() < prob:
                        cover_count[old_r::m] -= 1
                        cover_count[new_r::m] += 1
                        active[m] = new_r
                        current_uncovered += delta_uncovered
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_active = {k: v for k, v in active.items()}
                        
        elif move_type < 0.85:
            # Add a new modulus
            # Pick a random divisor not in active
            # To speed up, pick from a random subset of candidates
            # Candidates: divisors not in active
            # This list can be large.
            # Let's just pick a random divisor from all_divs. If it's in active, skip.
            m = random.choice(all_divs)
            if m in active:
                continue
            r = random.randint(0, m - 1)
            
            # Calculate delta
            # Adding m, r increases cover_count[r::m] by 1
            # gained = sum(1 for v in cover_count[r::m] if v == 0)
            # lost = 0 (since we are only adding, no 1->0 transitions)
            
            new_vals = cover_count[r::m]
            gained = np.sum(new_vals == 0)
            delta_uncovered = -gained
            
            if delta_uncovered < 0:
                cover_count[r::m] += 1
                active[m] = r
                current_uncovered += delta_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_active = {k: v for k, v in active.items()}
            else:
                # delta_uncovered is 0 or positive? 
                # gained >= 0, so delta <= 0.
                # If gained == 0, delta == 0.
                # If gained > 0, delta < 0.
                # So it's either improvement or neutral.
                if delta_uncovered == 0:
                    # Neutral
                    cover_count[r::m] += 1
                    active[m] = r
                # Note: It is impossible for adding a class to increase uncovered count.
                # So we always accept adds if neutral or improving.
                
        else:
            # Remove a modulus
            if not active:
                continue
            m = random.choice(list(active.keys()))
            r = active[m]
            
            # Calculate delta
            # Removing m, r decreases cover_count[r::m] by 1
            # lost = sum(1 for v in cover_count[r::m] if v == 1)
            # gained = 0
            
            old_vals = cover_count[r::m]
            lost = np.sum(old_vals == 1)
            delta_uncovered = lost
            
            if delta_uncovered < 0:
                # Impossible, lost >= 0
                pass
            else:
                # If lost == 0, it's a neutral improvement (removes redundancy)
                # If lost > 0, it's a worsening.
                
                if delta_uncovered == 0:
                    cover_count[r::m] -= 1
                    del active[m]
                    # current_uncovered unchanged
                else:
                    prob = math.exp(-delta_uncovered / T)
                    if random.random() < prob:
                        cover_count[r::m] -= 1
                        del active[m]
                        current_uncovered += delta_uncovered
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_active = {k: v for k, v in active.items()}
                            
        T *= cooling_rate
        
        if best_uncovered == 0:
            break
            
    # Convert best_active to list of lists
    return [[m, r] for m, r in best_active.items()]
