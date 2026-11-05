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
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    base = 51975
    divisors = _all_odd_divisors(base)
    moduli = sorted(divisors)
    num_moduli = len(moduli)
    
    # Precompute step sizes for each modulus
    steps = [base // m for m in moduli]  # number of residues covered by each modulus
    
    # Precompute residue indices for each modulus
    # residue_indices[m_idx] = list of (r, np.arange(r, base, m)) for all r in range(moduli[m_idx])
    residue_indices = []
    for m in moduli:
        indices = []
        for r in range(m):
            indices.append(np.arange(r, base, m))
        residue_indices.append(indices)
    
    best_score = base
    best_classes = None
    
    def greedy_initialize(order):
        """Greedy: for each modulus in given order, pick residue covering most uncovered."""
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        
        for idx in order:
            m = moduli[idx]
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = np.count_nonzero(counts[residue_indices[idx][r]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            counts[residue_indices[idx][best_r]] += 1
        
        score = np.count_nonzero(counts == 0)
        return residues, counts, score
    
    def compute_delta_fast(counts, m_idx, old_r, new_r):
        """Compute change in uncovered count using precomputed indices."""
        if old_r == new_r:
            return 0
        old_slice = counts[residue_indices[m_idx][old_r]]
        new_slice = counts[residue_indices[m_idx][new_r]]
        lose = np.count_nonzero(old_slice == 1)
        gain = np.count_nonzero(new_slice == 0)
        return lose - gain
    
    def local_search(residues, counts, score, rng, time_limit, temp_start=8.0, cooling=0.998, min_temp=0.05, max_no_improve=5000):
        """Fast local search with efficient delta computation."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_score = score
        
        temperature = temp_start
        best_local_score = current_score
        best_local_residues = current_residues[:]
        
        start_ls = time.time()
        no_improve_count = 0
        
        while time.time() - start_ls < time_limit and temperature > min_temp:
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random modulus index
            i = rng.randint(0, num_moduli - 1)
            m = moduli[i]
            old_r = current_residues[i]
            new_r = rng.randint(0, m - 1)
            
            if new_r == old_r:
                continue
            
            # Compute delta efficiently
            delta = compute_delta_fast(current_counts, i, old_r, new_r)
            
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                # Accept move
                current_counts[residue_indices[i][old_r]] -= 1
                current_counts[residue_indices[i][new_r]] += 1
                current_residues[i] = new_r
                current_score += delta
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_residues = current_residues[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            temperature *= cooling
        
        return best_local_residues, best_local_score
    
    # Multiple restarts with different strategies
    strategies = [
        lambda rng: list(range(num_moduli)),  # natural order (small to large)
        lambda rng: list(range(num_moduli - 1, -1, -1)),  # reverse (large to small)
        lambda rng: rng.sample(range(num_moduli), num_moduli),  # random
        lambda rng: sorted(range(num_moduli), key=lambda i: moduli[i], reverse=True),  # large to small
        lambda rng: sorted(range(num_moduli), key=lambda i: -steps[i]),  # by coverage size
    ]
    
    rng = random.Random(42)
    
    for restart in range(50):  # more restarts with faster search
        if time.time() > deadline - 8:
            break
        
        # Choose strategy
        strategy_idx = restart % len(strategies)
        order = strategies[strategy_idx](rng)
        
        # Greedy initialization
        residues, counts, score = greedy_initialize(order)
        
        if score < best_score:
            best_score = score
            best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
        
        # Local search with time budget
        if time.time() > deadline - 10:
            break
        
        ls_time = min(15.0, deadline - 10 - time.time())
        if ls_time < 1.0:
            continue
        
        # Run local search with different temperature settings
        best_residues = residues
        best_ls_score = score
        
        # First pass: standard
        r1, s1 = local_search(residues, counts, score, rng, ls_time * 0.5, temp_start=8.0, cooling=0.998, min_temp=0.05)
        if s1 < best_ls_score:
            best_ls_score = s1
            best_residues = r1
        
        # Second pass: lower temperature for fine-tuning
        if time.time() < deadline - 10:
            remaining = deadline - 10 - time.time()
            if remaining > 0.5:
                r2, s2 = local_search(best_residues, counts, best_ls_score, rng, remaining, temp_start=3.0, cooling=0.999, min_temp=0.02, max_no_improve=10000)
                if s2 < best_ls_score:
                    best_ls_score = s2
                    best_residues = r2
        
        if best_ls_score < best_score:
            best_score = best_ls_score
            best_classes = [[moduli[i], best_residues[i]] for i in range(num_moduli)]
    
    # Final local search on the best solution found
    if best_classes is not None and time.time() < deadline - 3:
        final_residues = [c[1] for c in best_classes]
        final_counts = np.zeros(base, dtype=np.int8)
        for m, r in best_classes:
            final_counts[r::m] += 1
        final_score = np.count_nonzero(final_counts == 0)
        
        remaining_time = deadline - 3 - time.time()
        if remaining_time > 1.0:
            # Aggressive final polish
            improved_residues, improved_score = local_search(
                final_residues, final_counts, final_score, rng, remaining_time,
                temp_start=5.0, cooling=0.9995, min_temp=0.01, max_no_improve=20000
            )
            if improved_score < best_score:
                best_score = improved_score
                best_classes = [[moduli[i], improved_residues[i]] for i in range(num_moduli)]
    
    return best_classes
# EVOLVE-BLOCK-END
