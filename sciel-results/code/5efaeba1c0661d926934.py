import numpy as np
import time
import random

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = N
    best_residues = None
    
    # Sort divisors by size (smaller first) for greedy initialization
    divs_sorted = sorted(divisors)
    
    # Greedy initialization
    residues = {}
    count = np.zeros(N, dtype=np.int8)
    for d in divs_sorted:
        # Find residue class that covers most currently uncovered
        uncovered = np.nonzero(count == 0)[0]
        if len(uncovered) == 0:
            residues[d] = 0
            count[0::d] += 1
            continue
        
        # Count how many uncovered fall into each residue class mod d
        mods = uncovered % d
        gains = np.bincount(mods, minlength=d)
        best_r = int(np.argmax(gains))
        residues[d] = best_r
        count[best_r::d] += 1
    
    current_score = int(np.sum(count == 0))
    if current_score < best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Local search refinement
    while time.time() - start_time < timeout:
        improved = False
        shuffled_divs = divisors[:]
        random.shuffle(shuffled_divs)
        
        for d in shuffled_divs:
            if time.time() - start_time > timeout:
                break
                
            r_old = residues[d]
            
            # Calculate loss: number of x in class (d, r_old) that are covered exactly once
            x_old = np.arange(r_old, N, d)
            loss = int(np.sum(count[x_old] == 1))
            
            # Get indices of currently uncovered residues
            uncovered_idx = np.nonzero(count == 0)[0]
            
            if len(uncovered_idx) == 0:
                return [[d, int(residues[d])] for d in divisors]
                
            # For each uncovered x, x % d gives the residue class it falls into.
            # Count how many uncovered x fall into each residue class mod d.
            mods = uncovered_idx % d
            gains = np.bincount(mods, minlength=d)
            
            # Find the best residue to switch to
            best_r = int(np.argmax(gains))
            max_gain = int(gains[best_r])
            
            # Check if switching improves the score
            # delta = loss - max_gain
            # We want delta < 0, i.e., max_gain > loss
            
            if best_r != r_old and max_gain > loss:
                # Update count array
                count[r_old::d] -= 1
                count[best_r::d] += 1
                residues[d] = best_r
                improved = True
        
        if not improved:
            # Try a random perturbation
            for _ in range(10):
                if time.time() - start_time > timeout:
                    break
                d = random.choice(divisors)
                r_old = residues[d]
                r_new = random.randint(0, d - 1)
                if r_new == r_old:
                    continue
                
                # Evaluate this specific move
                x_old = np.arange(r_old, N, d)
                loss = int(np.sum(count[x_old] == 1))
                
                x_new = np.arange(r_new, N, d)
                gain = int(np.sum(count[x_new] == 0))
                
                # Accept if it's neutral or better
                if gain >= loss:
                    count[r_old::d] -= 1
                    count[r_new::d] += 1
                    residues[d] = r_new
                    improved = True
                    break
                    
            if not improved:
                # Try a stronger perturbation: change multiple residues
                k = random.randint(2, 5)
                for _ in range(k):
                    if time.time() - start_time > timeout:
                        break
                    d = random.choice(divisors)
                    r_old = residues[d]
                    r_new = random.randint(0, d - 1)
                    if r_new == r_old:
                        continue
                    
                    x_old = np.arange(r_old, N, d)
                    loss = int(np.sum(count[x_old] == 1))
                    
                    x_new = np.arange(r_new, N, d)
                    gain = int(np.sum(count[x_new] == 0))
                    
                    # Accept with some probability even if worse (simulated annealing style)
                    if gain >= loss or random.random() < 0.3:
                        count[r_old::d] -= 1
                        count[r_new::d] += 1
                        residues[d] = r_new
                        improved = True
                
                if not improved:
                    break
        
        # Update global best
        current_score = int(np.sum(count == 0))
        if current_score < best_score:
            best_score = current_score
            best_residues = residues.copy()
            if best_score == 0:
                break
            
    return [[d, int(best_residues[d])] for d in divisors]
