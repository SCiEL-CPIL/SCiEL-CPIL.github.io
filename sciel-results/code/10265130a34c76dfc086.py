import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    num_moduli = len(moduli)
    
    best_uncovered = float('inf')
    best_residues = None
    
    # Try multiple random seeds for diversity
    for seed in range(30):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Greedy initialization: process larger moduli first
        count_cover = np.zeros(base, dtype=np.int32)
        residues = {}
        
        # Sort moduli by size (larger first) with some randomness
        sorted_moduli = sorted(moduli, key=lambda m: (-m, random.random()))
        
        for m in sorted_moduli:
            # Find r that maximizes coverage of currently uncovered (count_cover==0)
            best_r = 0
            best_score = -1
            
            if m > 10000 and time.time() > deadline - 5:
                sample_size = min(m, 500)
                samples = random.sample(range(m), sample_size)
                for r in samples:
                    s = np.sum(count_cover[r::m] == 0)
                    if s > best_score:
                        best_score = s
                        best_r = r
            else:
                for r in range(m):
                    s = np.sum(count_cover[r::m] == 0)
                    if s > best_score:
                        best_score = s
                        best_r = r
            
            residues[m] = best_r
            count_cover[best_r::m] += 1
        
        # Local search with simulated annealing
        uncovered = np.sum(count_cover == 0)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = residues.copy()
        
        # Simulated annealing local search with faster delta computation
        current_uncovered = uncovered
        temperature = 100.0
        cooling = 0.998
        min_temperature = 0.1
        
        while temperature > min_temperature and time.time() < deadline - 5:
            # Pick a random modulus
            m = random.choice(moduli)
            r_old = residues[m]
            
            # Pick a new random residue
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            # Calculate delta in uncovered count efficiently
            # Positions affected: those where count changes
            old_positions = np.arange(r_old, base, m)
            new_positions = np.arange(r_new, base, m)
            
            # Decrement old
            old_vals = count_cover[old_positions]
            count_cover[old_positions] -= 1
            # Count how many went from 1 to 0 (newly uncovered)
            newly_uncovered_old = np.sum((old_vals == 1))
            
            # Increment new
            new_vals = count_cover[new_positions]
            count_cover[new_positions] += 1
            # Count how many went from 0 to 1 (newly covered)
            newly_covered_new = np.sum((new_vals == 0))
            
            delta = newly_uncovered_old - newly_covered_new
            new_uncovered = current_uncovered + delta
            
            # Accept if improvement, or with probability exp(-delta/temperature)
            if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                residues[m] = r_new
                current_uncovered = new_uncovered
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues.copy()
            else:
                # Revert
                count_cover[new_positions] -= 1
                count_cover[old_positions] += 1
            
            temperature *= cooling
        
        # Final local search pass (deterministic)
        for m in moduli:
            if time.time() > deadline - 2:
                break
                
            r_old = residues[m]
            count_others = count_cover.copy()
            count_others[r_old::m] -= 1
            
            best_r = r_old
            best_score = np.sum(count_others[r_old::m] == 0)
            
            # Check all residues for small moduli, sample for large
            if m <= 500:
                candidates = list(range(m))
            else:
                candidates = [r_old] + random.sample(range(m), min(200, m - 1))
            
            for r in candidates:
                if r == r_old:
                    continue
                s = np.sum(count_others[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            
            if best_r != r_old:
                count_cover[r_old::m] -= 1
                count_cover[best_r::m] += 1
                residues[m] = best_r
        
        # Update best
        final_uncovered = np.sum(count_cover == 0)
        if final_uncovered < best_uncovered:
            best_uncovered = final_uncovered
            best_residues = residues.copy()
    
    # Construct return value from best solution
    if best_residues is None:
        best_residues = residues
    
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
