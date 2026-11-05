"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 30
    best = None
    best_abs = None
    
    # Seed solution: [0]*28 + [1, -1] gives sum = 1/29 - 1/30 = 1/(29*30) = 1/870
    seed = [0]*28 + [1, -1]
    seed_frac = Fraction(0)
    for i, c in enumerate(seed, 1):
        seed_frac += Fraction(c, i)
    best_abs = abs(seed_frac)
    best = seed[:]
    
    def compute_abs(coeffs):
        s = Fraction(0)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        if s == 0:
            return None
        return abs(s)
    
    def random_coeffs():
        # Random sparse coefficients
        n_nonzero = random.randint(2, min(8, N))
        indices = random.sample(range(N), n_nonzero)
        coeffs = [0]*N
        for idx in indices:
            coeffs[idx] = random.choice([1, -1])
        return coeffs
    
    def local_search(start, max_steps, deadline):
        """Greedy local search with restarts, using exact fractions."""
        coeffs = start[:]
        current_abs = compute_abs(coeffs)
        if current_abs is None:
            return start, None
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            improved = False
            # Try single-bit flips
            for i in range(N):
                new_coeffs = coeffs[:]
                new_coeffs[i] = -coeffs[i] if coeffs[i] != 0 else (1 if random.random() < 0.5 else -1)
                
                # Actually, if coeffs[i] is 0, flipping to 1 or -1 adds a term
                # If coeffs[i] is 1, flipping to -1 changes by -2/i
                # If coeffs[i] is -1, flipping to 1 changes by 2/i
                # Let's just recompute for simplicity with exact fractions
                
                new_abs = compute_abs(new_coeffs)
                if new_abs is not None and (current_abs is None or new_abs < current_abs):
                    coeffs = new_coeffs
                    current_abs = new_abs
                    improved = True
                    break
                # Also try setting to 0
                if coeffs[i] != 0:
                    new_coeffs2 = coeffs[:]
                    new_coeffs2[i] = 0
                    new_abs2 = compute_abs(new_coeffs2)
                    if new_abs2 is not None and (current_abs is None or new_abs2 < current_abs):
                        coeffs = new_coeffs2
                        current_abs = new_abs2
                        improved = True
                        break
            
            if not improved:
                # Try 2-bit moves to escape local minima
                for i in range(N):
                    for j in range(i+1, N):
                        new_coeffs = coeffs[:]
                        new_coeffs[i] = -coeffs[i] if coeffs[i] != 0 else 1
                        new_coeffs[j] = -coeffs[j] if coeffs[j] != 0 else 1
                        new_abs = compute_abs(new_coeffs)
                        if new_abs is not None and (current_abs is None or new_abs < current_abs):
                            coeffs = new_coeffs
                            current_abs = new_abs
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
        
        return coeffs, current_abs
    
    # Phase 1: Random restarts with local search
    max_restarts = 50
    restarts_done = 0
    while time.time() < deadline and restarts_done < max_restarts:
        start = random_coeffs()
        coeffs, abs_val = local_search(start, max_steps=200, deadline=deadline)
        if abs_val is not None and (best_abs is None or abs_val < best_abs):
            best_abs = abs_val
            best = coeffs[:]
        restarts_done += 1
    
    # Phase 2: Simulated annealing from best found so far
    if time.time() < deadline - 5:
        coeffs = best[:]
        current_abs = best_abs
        T = 0.5
        T_min = 1e-9
        cooling_rate = 0.995
        
        for sa_step in range(50000):
            if time.time() > deadline - 2:
                break
            
            # Pick a random index to flip
            i = random.randint(0, N-1)
            old_val = coeffs[i]
            new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
            
            new_coeffs = coeffs[:]
            new_coeffs[i] = new_val
            new_abs = compute_abs(new_coeffs)
            
            if new_abs is not None:
                if current_abs is None or new_abs < current_abs:
                    coeffs = new_coeffs
                    current_abs = new_abs
                    if current_abs < best_abs:
                        best_abs = current_abs
                        best = coeffs[:]
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_abs - current_abs
                    if delta < 0:
                        coeffs = new_coeffs
                        current_abs = new_abs
                    else:
                        prob = min(1.0, max(0.0, 1.0 - delta / (T + 1e-12)))
                        if random.random() < prob:
                            coeffs = new_coeffs
                            current_abs = new_abs
            # If new_abs is None (zero sum), reject to stay in valid space
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Update best after SA
        if current_abs is not None and current_abs < best_abs:
            best_abs = current_abs
            best = coeffs[:]
    
    # Final verification
    final_abs = compute_abs(best)
    if final_abs is None:
        return seed
    
    return best
# EVOLVE-BLOCK-END
