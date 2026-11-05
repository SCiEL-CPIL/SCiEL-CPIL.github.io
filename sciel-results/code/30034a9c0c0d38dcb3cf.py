"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_odd_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    base = 51975
    # Precompute all odd divisors > 1 of base
    divisors = _all_odd_divisors(base)
    # Sort by increasing m - small moduli cover more residues
    moduli = sorted(divisors)
    num_moduli = len(moduli)
    
    best_score = base  # initially all uncovered
    best_classes = None
    
    def greedy_initialize(rng, order=None):
        """Greedy initialization: for each modulus in order, pick residue covering most uncovered."""
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        
        if order is None:
            order = list(range(num_moduli))
        
        for idx in order:
            m = moduli[idx]
            best_r = 0
            best_gain = -1
            # Try all residues for this modulus
            for r in range(m):
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            counts[best_r::m] += 1
        
        score = np.count_nonzero(counts == 0)
        return residues, counts, score
    
    def local_search(residues, counts, score, rng, time_limit=30.0):
        """Simulated annealing local search."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_score = score
        
        temperature = 10.0
        cooling = 0.999
        min_temp = 0.1
        best_local_score = current_score
        best_local_residues = current_residues[:]
        
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit and temperature > min_temp:
            # Pick a random modulus index
            i = rng.randint(0, num_moduli - 1)
            m = moduli[i]
            old_r = current_residues[i]
            new_r = rng.randint(0, m - 1)
            
            if new_r == old_r:
                continue
            
            # Update counts
            current_counts[old_r::m] -= 1
            current_counts[new_r::m] += 1
            
            new_score = np.count_nonzero(current_counts == 0)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_residues[i] = new_r
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_residues = current_residues[:]
            else:
                # Revert
                current_counts[new_r::m] -= 1
                current_counts[old_r::m] += 1
            
            temperature *= cooling
        
        return best_local_residues, best_local_score
    
    # Multiple restarts with different strategies
    strategies = [
        lambda rng: list(range(num_moduli)),  # natural order (small to large)
        lambda rng: list(range(num_moduli - 1, -1, -1)),  # reverse (large to small)
        lambda rng: rng.sample(range(num_moduli), num_moduli),  # random
        lambda rng: sorted(range(num_moduli), key=lambda i: moduli[i], reverse=True),  # large to small
        lambda rng: sorted(range(num_moduli), key=lambda i: -moduli[i]),  # large to small
    ]
    
    rng = random.Random(42)
    
    for restart in range(20):  # up to 20 restarts
        if time.time() > deadline - 10:
            break
        
        # Choose strategy
        strategy_idx = restart % len(strategies)
        order = strategies[strategy_idx](rng)
        
        # Greedy initialization
        residues, counts, score = greedy_initialize(rng, order)
        
        if score < best_score:
            best_score = score
            best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
        
        # Local search
        if time.time() > deadline - 15:
            break
        
        ls_time = min(20.0, deadline - 15 - time.time())
        if ls_time < 2.0:
            continue
        
        best_residues, best_ls_score = local_search(residues, counts, score, rng, ls_time)
        
        if best_ls_score < best_score:
            best_score = best_ls_score
            best_classes = [[moduli[i], best_residues[i]] for i in range(num_moduli)]
    
    # Final local search on the best solution found
    if best_classes is not None and time.time() < deadline - 5:
        final_residues = [c[1] for c in best_classes]
        final_counts = np.zeros(base, dtype=np.int8)
        for m, r in best_classes:
            final_counts[r::m] += 1
        final_score = np.count_nonzero(final_counts == 0)
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2.0:
            improved_residues, improved_score = local_search(
                final_residues, final_counts, final_score, rng, remaining_time
            )
            if improved_score < best_score:
                best_score = improved_score
                best_classes = [[moduli[i], improved_residues[i]] for i in range(num_moduli)]
    
    return best_classes
# EVOLVE-BLOCK-END
