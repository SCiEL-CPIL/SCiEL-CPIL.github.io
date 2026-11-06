import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    moduli_set = set(moduli)
    
    # Identify "anchor" moduli: the largest prime-power factors
    # 27 = 3^3, 25 = 5^2, 7, 11
    anchors = [27, 25, 7, 11]
    non_anchors = [m for m in moduli if m not in anchors]
    
    best_uncovered = float('inf')
    best_residues = None
    
    def compute_uncovered(count_cover):
        return np.sum(count_cover == 0)
    
    # Try multiple random seeds for diversity
    for seed in range(25):
        if time.time() > deadline - 15:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        count_cover = np.zeros(base, dtype=np.int8)
        residues = {}
        
        # Phase 1: Fix anchors with random residues
        for m in anchors:
            r = random.randint(0, m - 1)
            residues[m] = r
            count_cover[r::m] += 1
        
        # Phase 2: Greedy assignment for non-anchor moduli
        # Sort by size (smaller first covers more)
        sorted_non_anchors = sorted(non_anchors, key=lambda m: m)
        
        for m in sorted_non_anchors:
            # Find r that maximizes coverage of currently uncovered residues
            best_r = 0
            best_score = -1
            
            # For large moduli, sample if time is tight
            if m > 5000 and time.time() > deadline - 10:
                sample_size = min(m, 200)
                candidates = random.sample(range(m), sample_size)
            else:
                candidates = range(m)
            
            for r in candidates:
                s = np.sum(count_cover[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            
            residues[m] = best_r
            count_cover[best_r::m] += 1
        
        # Compute current uncovered
        current_uncovered = compute_uncovered(count_cover)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues.copy()
        
        # Phase 3: Local search with simulated annealing + pairwise swaps
        temperature = 30.0
        cooling = 0.997
        min_temperature = 0.05
        
        while temperature > min_temperature and time.time() < deadline - 10:
            # Choose move type: single (70%) or pairwise swap (30%)
            if random.random() < 0.7:
                # Single residue change
                m = random.choice(moduli)
                r_old = residues[m]
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                
                count_cover[r_old::m] -= 1
                old_uncovered = compute_uncovered(count_cover)
                
                count_cover[r_new::m] += 1
                new_uncovered = compute_uncovered(count_cover)
                
                delta = new_uncovered - old_uncovered
                
                if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                    residues[m] = r_new
                    current_uncovered = new_uncovered
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_residues = residues.copy()
                else:
                    count_cover[r_new::m] -= 1
                    count_cover[r_old::m] += 1
            else:
                # Pairwise swap: pick two moduli and swap their residues
                m1, m2 = random.sample(moduli, 2)
                r1_old = residues[m1]
                r2_old = residues[m2]
                
                # Try swapping
                count_cover[r1_old::m1] -= 1
                count_cover[r2_old::m2] -= 1
                
                # Apply swap
                count_cover[r2_old::m1] += 1
                count_cover[r1_old::m2] += 1
                
                new_uncovered = compute_uncovered(count_cover)
                delta = new_uncovered - current_uncovered
                
                if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                    residues[m1] = r2_old
                    residues[m2] = r1_old
                    current_uncovered = new_uncovered
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_residues = residues.copy()
                else:
                    # Revert
                    count_cover[r2_old::m1] -= 1
                    count_cover[r1_old::m2] -= 1
                    count_cover[r1_old::m1] += 1
                    count_cover[r2_old::m2] += 1
            
            temperature *= cooling
        
        # Phase 4: Deterministic local search pass
        for m in moduli:
            if time.time() > deadline - 5:
                break
            
            r_old = residues[m]
            count_others = count_cover.copy()
            count_others[r_old::m] -= 1
            
            best_r = r_old
            best_score = np.sum(count_others[r_old::m] == 0)
            
            # Check all residues for small m, sample for large m
            if m <= 1000:
                candidates = range(m)
            else:
                candidates = [r_old] + random.sample(range(m), min(100, m - 1))
            
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
        
        final_uncovered = compute_uncovered(count_cover)
        if final_uncovered < best_uncovered:
            best_uncovered = final_uncovered
            best_residues = residues.copy()
    
    # Construct return value from best solution
    if best_residues is None:
        # Fallback: greedy
        best_residues = {}
        count_cover = np.zeros(base, dtype=np.int8)
        for m in sorted(moduli):
            best_r = 0
            best_score = -1
            for r in range(m):
                s = np.sum(count_cover[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            best_residues[m] = best_r
            count_cover[best_r::m] += 1
    
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
