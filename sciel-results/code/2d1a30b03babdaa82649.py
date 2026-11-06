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
    
    # Sort moduli ascending for greedy and local search
    # Small moduli cover more, so fixing them first is usually better
    sorted_moduli = sorted(moduli)
    
    best_uncovered = float('inf')
    best_residues = None
    
    # Helper to compute uncovered count and update arrays
    def get_uncovered(count_cover):
        return np.sum(count_cover == 0)
    
    # We will run multiple restarts with different random kicks
    # Each restart starts from a greedy solution or a perturbed greedy solution
    
    for seed in range(50):  # Max 50 restarts, but break early if time is low
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # 1. Greedy Initialization
        count_cover = np.zeros(base, dtype=np.int8)
        residues = {}
        
        # Ascending order greedy
        for m in sorted_moduli:
            best_r = 0
            best_score = -1
            # For greedy init, we just pick the r that covers most currently uncovered
            # Optimization: if m is large, we might sample, but 51975 is small enough
            # to just loop. 48 moduli * avg 1000 residues = 48k ops. Fast.
            for r in range(m):
                s = np.sum(count_cover[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            
            residues[m] = best_r
            count_cover[best_r::m] += 1
        
        current_uncovered = get_uncovered(count_cover)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues.copy()
        
        # 2. Iterated Local Search (ILS)
        # We optimize each modulus to its best possible residue given the others.
        # We repeat this until convergence or a kick is needed.
        
        for iteration in range(200):  # Max iterations per restart
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # Pass 1: Optimize each modulus in ascending order
            # We do a full sweep.
            for m in sorted_moduli:
                if time.time() > deadline - 2:
                    break
                
                r_old = residues[m]
                
                # Temporarily remove this modulus
                count_cover[r_old::m] -= 1
                
                # Find best new residue
                best_r = r_old
                best_score = -1
                
                # We want to maximize coverage of currently uncovered (count_cover == 0)
                # Note: count_cover currently does NOT include m's contribution
                for r in range(m):
                    s = np.sum(count_cover[r::m] == 0)
                    if s > best_score:
                        best_score = s
                        best_r = r
                
                # If we found a better residue, update
                if best_r != r_old:
                    residues[m] = best_r
                    count_cover[best_r::m] += 1
                    improved = True
                else:
                    # Re-add the old residue since we didn't change it
                    count_cover[r_old::m] += 1
            
            # Check if we improved globally
            current_uncovered = get_uncovered(count_cover)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues.copy()
            
            # If no improvement in this full sweep, we are in a local minimum.
            # Apply a kick to escape.
            if not improved:
                # Kick: randomly change residues of a few moduli
                # Changing small moduli has a bigger impact, so maybe bias towards them?
                # Or just random.
                kick_size = min(num_moduli, 3 + random.randint(0, 3))
                kick_moduli = random.sample(sorted_moduli, kick_size)
                for m in kick_moduli:
                    old_r = residues[m]
                    new_r = random.randint(0, m - 1)
                    if new_r != old_r:
                        count_cover[old_r::m] -= 1
                        count_cover[new_r::m] += 1
                        residues[m] = new_r
                
                # After kick, we must re-optimize in the next iteration
                # The loop continues, so 'improved' will be checked again.
                # But we should force the next iteration to run the optimization pass.
                # The loop structure does this automatically.
                
                # If we've kicked many times without improvement, maybe break this restart?
                # For now, let it run.
                
    # Construct return value from best solution
    if best_residues is None:
        # Fallback to last residues if something went wrong
        best_residues = residues
    
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
