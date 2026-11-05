"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by increasing m for greedy (small moduli cover more, so decide them first)
    divisors.sort()
    
    num_moduli = len(divisors)
    
    # Greedy initialization with count array
    counts = np.zeros(base, dtype=np.int16)
    residues = [0] * num_moduli
    
    for i, m in enumerate(divisors):
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count how many residues with count==0 are in position r, r+m, r+2m, ...
            idx = np.arange(r, base, m)
            gain = np.count_nonzero(counts[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        counts[best_r::m] += 1
    
    current_score = int(np.count_nonzero(counts == 0))
    
    best_score = current_score
    best_residues = residues[:]
    
    # Fast local search: try changing one residue at a time
    # For modulus m_i with old residue r_old, new residue r_new:
    #   Remove coverage: counts[r_old::m_i] -= 1
    #   Add coverage: counts[r_new::m_i] += 1
    #   Score change = (# newly uncovered) - (# newly covered)
    #   where newly uncovered = positions where count went from 1 to 0
    #   and newly covered = positions where count went from 0 to 1
    
    def compute_score_delta(m, r_old, r_new, counts):
        """Compute delta in uncovered count when changing residue of modulus m from r_old to r_new."""
        # Positions affected by old residue
        old_idx = np.arange(r_old, base, m)
        # Positions affected by new residue
        new_idx = np.arange(r_new, base, m)
        
        # Remove old coverage
        old_counts = counts[old_idx].copy()
        counts[old_idx] -= 1
        # Count how many went from 1 to 0 (newly uncovered)
        newly_uncovered = np.count_nonzero((old_counts == 1))
        
        # Add new coverage
        new_counts = counts[new_idx].copy()
        counts[new_idx] += 1
        # Count how many went from 0 to 1 (newly covered)
        newly_covered = np.count_nonzero((new_counts == 0))
        
        # Restore (we'll do the actual update separately)
        counts[old_idx] += 1
        counts[new_idx] -= 1
        
        return newly_uncovered - newly_covered
    
    def apply_change(m, r_old, r_new, counts):
        """Actually apply the residue change."""
        counts[r_old::m] -= 1
        counts[r_new::m] += 1
    
    # Hill climbing with multiple passes
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(num_moduli):
            if time.time() > deadline - 20:
                break
            m = divisors[i]
            r_old = residues[i]
            best_r = r_old
            best_delta = 0  # 0 means no change
            
            for r_new in range(m):
                if r_new == r_old:
                    continue
                delta = compute_score_delta(m, r_old, r_new, counts)
                if delta < best_delta:
                    best_delta = delta
                    best_r = r_new
            
            if best_r != r_old:
                apply_change(m, r_old, best_r, counts)
                residues[i] = best_r
                current_score += best_delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues[:]
    
    # Simulated annealing phase
    if time.time() < deadline - 10:
        rng = random.Random(42)
        temperature = 5.0
        cooling = 0.998
        min_temp = 0.1
        
        while time.time() < deadline - 10 and temperature > min_temp:
            # Pick a random modulus
            i = rng.randint(0, num_moduli - 1)
            m = divisors[i]
            r_old = residues[i]
            r_new = rng.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            delta = compute_score_delta(m, r_old, r_new, counts)
            
            if delta < 0:
                # Improvement, always accept
                apply_change(m, r_old, r_new, counts)
                residues[i] = r_new
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues[:]
            else:
                # Worse, accept with probability exp(-delta / temperature)
                import math
                if delta < 50:  # Avoid overflow
                    prob = math.exp(-delta / temperature)
                    if rng.random() < prob:
                        apply_change(m, r_old, r_new, counts)
                        residues[i] = r_new
                        current_score += delta
            
            temperature *= cooling
    
    # Try removing classes that don't help
    if time.time() < deadline - 5:
        # Rebuild counts from current residues
        counts2 = np.zeros(base, dtype=np.int16)
        for i, m in enumerate(divisors):
            counts2[residues[i]::m] += 1
        
        # Try removing each class
        to_remove = []
        for i in range(num_moduli):
            if time.time() > deadline - 5:
                break
            m = divisors[i]
            r = residues[i]
            # Check if removing this class increases uncovered count
            # A class is useless if every residue it covers is also covered by another class
            # i.e., for all positions p in r::m, counts2[p] > 1
            idx = np.arange(r, base, m)
            if np.all(counts2[idx] > 1):
                to_remove.append(i)
        
        # Remove useless classes
        if to_remove:
            # Build final list without removed classes
            keep = [i for i in range(num_moduli) if i not in to_remove]
            final_classes = [[divisors[i], residues[i]] for i in keep]
            # Compute final score
            counts3 = np.zeros(base, dtype=np.int16)
            for m, r in final_classes:
                counts3[r::m] += 1
            final_score = int(np.count_nonzero(counts3 == 0))
            if final_score < best_score:
                best_score = final_score
                best_residues = [residues[i] for i in keep]
                # Rebuild divisors list to match
                divisors_final = [divisors[i] for i in keep]
                return [[divisors_final[i], best_residues[i]] for i in range(len(divisors_final))]
    
    # Build final answer from best_residues
    # Need to handle the case where we might have removed some
    # Use best_residues which corresponds to full divisors list
    return [[divisors[i], best_residues[i]] for i in range(num_moduli)]
# EVOLVE-BLOCK-END
