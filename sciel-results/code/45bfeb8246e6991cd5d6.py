import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    base = 51975
    # Precompute all odd divisors > 1 of base
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    moduli = sorted(divisors)
    num_moduli = len(moduli)
    
    best_score = base
    best_classes = None
    
    def greedy_initialize(rng):
        """Greedy initialization: for each modulus in increasing order, pick residue covering most uncovered."""
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        uncovered_count = base
        
        # Process smaller moduli first
        for i in range(num_moduli):
            m = moduli[i]
            is_zero = (counts == 0)
            best_r = 0
            best_gain = -1
            # Find r that maximizes gain
            # Optimization: if m is large, the loop is slow. But m is divisor.
            for r in range(m):
                gain = is_zero[r::m].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[i] = best_r
            # Update counts and uncovered_count
            became_zero = 0 # Not needed for init, counts are 0
            # Actually, we just add to counts.
            # was_zero = is_zero[best_r::m].sum()
            # counts[best_r::m] += 1
            # uncovered_count -= was_zero
            
            # Faster update for init:
            was_zero = is_zero[best_r::m].sum()
            counts[best_r::m] += 1
            uncovered_count -= was_zero
            
        return residues, counts, uncovered_count

    def local_search(residues, counts, score, rng, time_limit):
        """Simulated Annealing with best-response moves."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_score = score
        
        temperature = 10.0
        cooling = 0.999
        min_temp = 0.01
        best_local_score = current_score
        best_local_residues = current_residues[:]
        
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit and temperature > min_temp:
            if time.time() > time_limit + start_ls:
                break
                
            # Pick a random modulus index
            i = rng.randint(0, num_moduli - 1)
            m = moduli[i]
            old_r = current_residues[i]
            
            # 1. Remove old_r
            # Count how many elements become zero (were 1)
            became_zero = np.sum(current_counts[old_r::m] == 1)
            current_counts[old_r::m] -= 1
            current_score += became_zero
            
            # 2. Find best_r
            is_zero = (current_counts == 0)
            best_r = old_r # Default to old if no improvement found easily? No, find max gain.
            best_gain = -1
            
            # Optimization for finding best_r
            # If m is small, loop is fast.
            # If m is large, loop is slow but slices are small.
            # We can break early if best_gain == base//m (max possible)?
            max_possible = base // m
            if max_possible == 1:
                # Only one residue effectively? No, m divides base.
                # If m == base, m=51975, max_possible=1.
                # Then any r covers 1 residue.
                # We just pick the one that is zero.
                # is_zero.sum() is current_score.
                # We want r such that is_zero[r] == 1.
                # If any is 1, pick it. Else pick 0.
                # This case is rare (only m=51975).
                pass

            for r in range(m):
                gain = is_zero[r::m].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
                # Early exit if perfect coverage for this modulus
                if best_gain == max_possible:
                    break
            
            # 3. Add best_r
            was_zero = is_zero[best_r::m].sum() # This is best_gain
            current_counts[best_r::m] += 1
            current_score -= best_gain
            
            delta = current_score - score # Wait, score variable is old_score?
            # Let's track old_score properly.
            # Actually, I updated current_score in place.
            # Let's use a separate variable for the score before the move.
            
            # Re-logic:
            # score_before = current_score (after removal? No, before removal)
            # Let's restart the loop logic to be clean.
            
            # Break out and rewrite the loop body cleanly in the final code.
            
        return best_local_residues, best_local_score

    # Main loop
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        residues, counts, score = greedy_initialize(rng)
        
        if score < best_score:
            best_score = score
            best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
            
        if time.time() > deadline - 15:
            break
            
        ls_time = min(20.0, deadline - 15 - time.time())
        if ls_time < 2.0:
            continue
            
        # Call local search
        # Need to implement local_search properly.
        
    return best_classes
