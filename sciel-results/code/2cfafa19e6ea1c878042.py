import numpy as np
import time
import random

def construct():
    base = 51975
    # Find all odd divisors > 1
    moduli = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            moduli.append(d)
    
    # Sort moduli, maybe smaller first is better for greedy?
    # But we will randomize selection.
    
    best_uncovered = base
    best_classes = []
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Number of restarts
    # We don't know how long one run takes. 
    # Let's estimate one run.
    # Greedy phase: ~47 steps. Each step computes gains for ~115k residues.
    # Numpy sum is fast. 115k ops is ~0.001s?
    # So 47 steps ~ 0.05s.
    # Local search: 47 moduli * avg m residues. ~115k ops per pass.
    # A few passes ~ 0.05s.
    # So one run ~ 0.1s.
    # We can do ~1000 runs in 100s.
    
    max_restarts = 2000  # Cap, but time guard will stop earlier
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        counts = np.zeros(base, dtype=np.int8)
        used_moduli = set()
        current_classes = []
        available_moduli = list(moduli)
        
        # Randomize order of moduli for tie-breaking or initial bias?
        # The greedy step picks based on gain, so order of iteration doesn't matter much 
        # unless we pick top-K randomly.
        
        # Greedy Construction
        while available_moduli:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidates = []
            
            # Calculate gains for all available (m, r)
            # To speed up, we can precompute indices for each m?
            # No, r changes.
            
            # Optimization: 
            # For a given m, the residues are r, r+m...
            # We can compute gain for all r at once?
            # gain(r) = sum(counts[r::m] == 0)
            # This is a strided sum. Numpy doesn't have direct strided sum over all shifts easily without slicing.
            # Looping r is O(m). Sum over m is O(sum(m)) = 115k.
            
            for m in available_moduli:
                # For each residue r in 0..m-1
                # We can compute gains for all r for this m in one go?
                # No, they overlap.
                # Just loop r.
                # If m is large, loop is large.
                # But sum of m is 115k.
                
                # To optimize inner loop:
                # If m is small, loop is fast.
                # If m is large, loop is slow but fewer m's are large.
                
                for r in range(m):
                    # Compute gain
                    # indices = np.arange(r, base, m)
                    # gain = np.sum(counts[indices] == 0)
                    
                    # Faster: 
                    # slice = counts[r::m]
                    # gain = np.sum(slice == 0)
                    
                    # Even faster?
                    # If we maintain a list of uncovered residues?
                    # No, counts array is best.
                    
                    # Let's try to vectorize the r-loop for a fixed m?
                    # Hard.
                    
                    # Just do it.
                    # Optimization: if current best_gain is already high, 
                    # and potential max gain for this m is low, skip?
                    # Max gain for m is len(range(r, base, m)).
                    # If max_possible_gain <= best_gain, skip this m?
                    # No, because we need to find the *best* candidate overall.
                    # But if we just want *a* good candidate, we can stop early?
                    # No, we need top-K.
                    
                    # Let's just compute all.
                    slice_ = counts[r::m]
                    gain = np.sum(slice_ == 0)
                    
                    if gain > 0:
                        best_candidates.append((gain, m, r))
            
            if not best_candidates:
                break
            
            # Sort candidates by gain
            best_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top K
            K = 5
            top_k = best_candidates[:K]
            _, m_sel, r_sel = random.choice(top_k)
            
            # Apply selection
            indices = np.arange(r_sel, base, m_sel)
            counts[indices] += 1
            used_moduli.add(m_sel)
            current_classes.append([m_sel, r_sel])
            available_moduli.remove(m_sel)
            
        # Local Search
        # Try to swap residues for used moduli
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order to avoid bias?
            local_moduli = list(used_moduli)
            random.shuffle(local_moduli)
            
            for m in local_moduli:
                # Find current r for m
                curr_r = None
                for c in current_classes:
                    if c[0] == m:
                        curr_r = c[1]
                        break
                if curr_r is None: continue
                
                # Try all other r
                # We want to find r' that maximizes gain.
                # Gain of swap = (New Covered by m) - (Old Covered by m)
                # New Covered by m = count of residues in S(m, r') that are NOT covered by others.
                # Others = all classes except (m, curr_r).
                
                # To compute this efficiently:
                # 1. Temporarily remove (m, curr_r) from counts.
                indices_old = np.arange(curr_r, base, m)
                counts[indices_old] -= 1
                
                best_r_prime = curr_r
                best_gain_prime = 0 # Gain relative to state without m
                
                # Find best r'
                for r_prime in range(m):
                    if r_prime == curr_r:
                        continue
                    indices_new = np.arange(r_prime, base, m)
                    gain = np.sum(counts[indices_new] == 0)
                    if gain > best_gain_prime:
                        best_gain_prime = gain
                        best_r_prime = r_prime
                
                # If best_r_prime is not curr_r, we need to compare with keeping curr_r?
                # Actually, we removed curr_r. 
                # If we keep curr_r, gain is np.sum(counts[indices_old] == 0) (after re-adding? No, counts is currently without m).
                # So gain_keep = np.sum(counts[indices_old] == 0)
                gain_keep = np.sum(counts[indices_old] == 0)
                
                if best_gain_prime > gain_keep:
                    # Apply swap
                    indices_new = np.arange(best_r_prime, base, m)
                    counts[indices_new] += 1
                    # Update current_classes
                    for c in current_classes:
                        if c[0] == m:
                            c[1] = best_r_prime
                            break
                    improved = True
                else:
                    # Revert (keep curr_r)
                    counts[indices_old] += 1
        
        # Try adding unused moduli?
        # Check if any unused modulus has a residue with gain > 0
        # This might be expensive. 
        # Let's just do one pass of adding the best unused if gain > 0.
        if time.time() < deadline:
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                best_gain_add = 0
                best_m_add = None
                best_r_add = None
                
                for m in available_moduli:
                    # Find best r for m
                    best_r_for_m = 0
                    best_gain_for_m = 0
                    for r in range(m):
                        indices = np.arange(r, base, m)
                        gain = np.sum(counts[indices] == 0)
                        if gain > best_gain_for_m:
                            best_gain_for_m = gain
                            best_r_for_m = r
                    if best_gain_for_m > best_gain_add:
                        best_gain_add = best_gain_for_m
                        best_m_add = m
                        best_r_add = best_r_for_m
                
                if best_gain_add > 0:
                    indices = np.arange(best_r_add, base, best_m_add)
                    counts[indices] += 1
                    used_moduli.add(best_m_add)
                    current_classes.append([best_m_add, best_r_add])
                    available_moduli.remove(best_m_add)
                    added_any = True
                    # Restart local search?
                    # Maybe just continue adding?
                    # Local search might be needed to adjust residues after adding.
                    # For simplicity, let's just do the greedy add.
                    # The local search above only optimized existing ones.
                    # We should probably re-run local search after adding.
                    # But time is tight.
                    # Let's just do the add.

        # Calculate final score
        uncovered = np.sum(counts == 0)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c[:] for c in current_classes]
            
    return best_classes
