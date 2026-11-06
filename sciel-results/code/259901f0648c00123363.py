import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    # Sort moduli
    moduli.sort()
    
    # State
    residues = {m: 0 for m in moduli}
    count_cover = np.zeros(base, dtype=np.int16)
    
    # Helper to compute score
    def get_score():
        return np.sum(count_cover == 0)
    
    # Helper to compute G(r) for a given m, using current count_cover
    # Returns array of length m where G[r] = sum(1 for x in r::m if count_cover[x]==0)
    def compute_G(m):
        # Vectorized approach using bincount
        mods = np.arange(base) % m
        uncovered = (count_cover == 0).astype(np.int16)
        G = np.bincount(mods, weights=uncovered, minlength=m)
        return G
    
    # Helper to compute L for current r_old and m
    # L = sum(1 for x in r_old::m if count_cover[x]==1)
    def compute_L(m, r_old):
        return np.sum(count_cover[r_old::m] == 1)
    
    # Initialize
    # Greedy initialization
    for m in moduli:
        G = compute_G(m)
        best_r = np.argmax(G)
        residues[m] = best_r
        count_cover[best_r::m] += 1
        
    current_score = get_score()
    best_score = current_score
    best_residues = residues.copy()
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    no_improve_steps = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Pick random modulus
        m = random.choice(moduli)
        r_old = residues[m]
        
        L = compute_L(m, r_old)
        G = compute_G(m)
        
        # Find best r
        r_best = np.argmax(G)
        G_best = G[r_best]
        delta_best = L - G_best
        
        # Decide move
        # With 90% probability, try the best move
        # With 10% probability, try a random move (to escape local minima)
        if random.random() < 0.9:
            r_new = r_best
            delta = delta_best
        else:
            r_new = random.randint(0, m - 1)
            G_new = G[r_new]
            delta = L - G_new
        
        # Acceptance
        if delta < 0:
            # Improvement
            accept = True
        else:
            # Worsening
            # Accept with probability exp(-delta / T)
            # Note: delta is increase in score (uncovered count). We want to minimize score.
            # So if delta > 0, score increases. We accept with prob exp(-delta/T).
            if T > 0:
                accept = random.random() < np.exp(-delta / T)
            else:
                accept = False
            
        if accept:
            # Update state
            count_cover[r_old::m] -= 1
            count_cover[r_new::m] += 1
            residues[m] = r_new
            
            new_score = current_score + delta
            if new_score < best_score:
                best_score = new_score
                best_residues = residues.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            current_score = new_score
        else:
            no_improve_steps += 1
            
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Restart if stuck
        if no_improve_steps > max_no_improve:
            # Reheat
            T = 5.0
            no_improve_steps = 0
            
    # Final Polish: Strict Hill Climbing
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for m in moduli:
            if time.time() > deadline - 5:
                break
            r_old = residues[m]
            L = compute_L(m, r_old)
            G = compute_G(m)
            r_best = np.argmax(G)
            G_best = G[r_best]
            delta_best = L - G_best
            
            if delta_best < 0:
                count_cover[r_old::m] -= 1
                count_cover[r_best::m] += 1
                residues[m] = r_best
                current_score += delta_best
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                improved = True

    # Return best solution
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
