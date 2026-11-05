import numpy as np
import time

BASE = 51975
DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Pre-sort divisors? Maybe smaller first for faster convergence?
    # Or random order.
    
    best_score = BASE
    best_classes = None
    
    # Initial random states
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize ref_count to 0
        ref_count = np.zeros(BASE, dtype=np.int8)
        
        # Assign random residues
        residues = {}
        for m in DIVISORS:
            residues[m] = np.random.randint(0, m)
            ref_count[residues[m]::m] += 1
            
        # Calculate initial score
        score = np.sum(ref_count == 0)
        
        if score < best_score:
            best_score = score
            best_classes = [[m, residues[m]] for m in DIVISORS]
            if best_score == 0:
                return best_classes

        # Local Search (Hill Climbing)
        no_improve_count = 0
        while time.time() < deadline:
            improved = False
            # Shuffle moduli for randomization
            current_divisors = DIVISORS.copy()
            np.random.shuffle(current_divisors)
            
            for m in current_divisors:
                if time.time() > deadline:
                    break
                    
                r_curr = residues[m]
                
                # Remove current class
                ref_count[r_curr::m] -= 1
                
                # Calculate loss: how many became uncovered?
                # Points where ref_count is now 0 in the slice
                loss = np.sum(ref_count[r_curr::m] == 0)
                
                # Find best new residue
                best_r = r_curr
                best_gain = -1
                # If m is small, loop all. If m is large, maybe sample?
                # But m <= 51975.
                # For m > 1000, the slice is small, so loop is fast.
                # For m < 1000, slice is large, but m is small.
                # Total work is O(N).
                
                # Optimization: If m is very large, the number of candidates is large,
                # but each check is cheap.
                # If m is small, candidates are few, but each check is expensive.
                
                for r in range(m):
                    # Gain is number of points in r::m that are currently 0
                    gain = np.sum(ref_count[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                # Apply best new class
                ref_count[best_r::m] += 1
                
                # Update score
                # Change in score = loss - best_gain
                delta = loss - best_gain
                if delta < 0:
                    score += delta
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_classes = [[m, residues[m]] for m in DIVISORS]
                        if best_score == 0:
                            return best_classes
                
                residues[m] = best_r
                
            if not improved:
                # Stuck in local minimum. Try a random perturbation?
                # Or just break and restart?
                # Let's try a few random moves to escape.
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    m = np.random.choice(current_divisors)
                    r_new = np.random.randint(0, m)
                    if r_new == residues[m]:
                        continue
                    
                    r_old = residues[m]
                    ref_count[r_old::m] -= 1
                    loss = np.sum(ref_count[r_old::m] == 0)
                    ref_count[r_new::m] += 1
                    gain = np.sum(ref_count[r_new::m] == 0)
                    # Note: gain here is calculated AFTER adding, so it includes the ones we just covered?
                    # No, ref_count[r_new::m] was incremented.
                    # We need the gain BEFORE adding?
                    # Actually, the logic "loss - gain" worked for the greedy step.
                    # For random moves, we just update score.
                    # Score change = (new_uncovered) - (old_uncovered)
                    # Old uncovered in slice r_old: count of 0s after removal = loss.
                    # New uncovered in slice r_new: count of 0s after addition.
                    # Wait, the "gain" in the greedy step was "count of 0s BEFORE addition".
                    # So delta = loss - gain.
                    # Here, we already added.
                    # Let's just recalculate score locally? No, too slow.
                    # Let's stick to the greedy logic. If we do random moves,
                    # we should calculate the delta properly.
                    
                    # Actually, simpler: just accept if it doesn't increase score too much?
                    # Or just break and restart.
                    # Given the strong greedy pull, restarts might be better.
                    break 
                break

    return best_classes
