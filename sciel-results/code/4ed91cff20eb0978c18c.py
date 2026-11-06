"""
Genetic Algorithm with vectorized local search to minimize monochromatic 4-term APs in [1, 256].
"""
import numpy as np
import time

N = 256

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256, d >= 1
def _precompute_aps():
    aps = []
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return np.array(aps, dtype=np.int32)

AP_IDX = _precompute_aps()
NUM_APS = AP_IDX.shape[0]

# For each position, store which APs it belongs to and the positions of the other 3 elements
# Store as arrays for vectorized access
_AP_POS_LISTS = []
_AP_OTHER_LISTS = []
for i in range(N):
    mask = (AP_IDX == i).any(axis=1)
    ap_indices = np.where(mask)[0]
    _AP_POS_LISTS.append(ap_indices)
    # For each AP containing i, get the other 3 positions
    others = []
    for k in ap_indices:
        idxs = AP_IDX[k]
        other = [idxs[j] for j in range(4) if idxs[j] != i]
        others.append(other)
    _AP_OTHER_LISTS.append(np.array(others, dtype=np.int32) if others else np.empty((0,3), dtype=np.int32))

def _count_mono_aps(colours):
    cols = colours[AP_IDX]
    mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    return int(np.sum(mono))

def _local_search(colours, max_sweeps=30, rng=None):
    """
    Vectorized local search: for each position, compute delta of flipping it.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    colours = colours.copy().astype(np.int8)
    cols = colours[AP_IDX]
    mono_status = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    current_count = int(np.sum(mono_status))
    
    for sweep in range(max_sweeps):
        improved = False
        order = np.arange(N)
        rng.shuffle(order)
        
        for pos in order:
            old_val = colours[pos]
            new_val = 1 - old_val
            
            ap_indices = _AP_POS_LISTS[pos]
            if len(ap_indices) == 0:
                continue
            
            other_arr = _AP_OTHER_LISTS[pos]
            if len(other_arr) == 0:
                continue
            
            # Get colors of the other 3 elements for each AP
            # other_arr shape: (num_aps, 3)
            other_cols = colours[other_arr]  # shape (num_aps, 3)
            
            # Check if all 3 others equal new_val
            all_match_new = np.all(other_cols == new_val, axis=1)
            
            # Delta: +1 for each AP that becomes mono (was not mono, now all match new_val)
            #         -1 for each AP that was mono and now isn't
            was_mono = mono_status[ap_indices]
            
            # APs that become mono: was not mono AND all others match new_val
            becomes_mono = ~was_mono & all_match_new
            # APs that break mono: was mono AND NOT all others match new_val
            # Note: if was_mono, then all 4 were same color. After flip, the flipped one differs.
            # So it breaks mono iff the other 3 don't all equal new_val... actually:
            # If was_mono, all 4 had color old_val. After flip, pos has new_val, others have old_val.
            # So it's no longer mono. Always breaks.
            breaks_mono = was_mono
            
            delta = int(np.sum(becomes_mono)) - int(np.sum(breaks_mono))
            
            if delta < 0:
                colours[pos] = new_val
                current_count += delta
                # Update mono_status for affected APs
                for j, k in enumerate(ap_indices):
                    idxs = AP_IDX[k]
                    mono_status[k] = (colours[idxs[0]] == colours[idxs[1]]) and \
                                     (colours[idxs[1]] == colours[idxs[2]]) and \
                                     (colours[idxs[2]] == colours[idxs[3]])
                improved = True
        
        if not improved:
            break
    
    return colours, current_count

def _generate_diverse_initial(rng):
    """Generate diverse initial population using structured patterns."""
    pop = []
    
    # Random
    for _ in range(8):
        pop.append(rng.randint(0, 2, size=N).astype(np.int8))
    
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        tm[i] = bin(i).count('1') % 2
    pop.append(tm)
    
    # Block patterns
    for bs in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 32]:
        pop.append(np.array([(i // bs) % 2 for i in range(N)], dtype=np.int8))
    
    # Periodic patterns
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16]:
        half = period // 2
        pop.append(np.array([(i % period) < half for i in range(N)], dtype=np.int8))
    
    # XOR patterns
    for shift in [1, 2, 3, 4]:
        pop.append(np.array([(i & 1) ^ ((i >> shift) & 1) for i in range(N)], dtype=np.int8))
    
    # Gradient-like
    pop.append(np.array([(i // 64) % 2 for i in range(N)], dtype=np.int8))
    pop.append(np.array([(i // 32) % 2 for i in range(N)], dtype=np.int8))
    pop.append(np.array([(i // 16) % 2 for i in range(N)], dtype=np.int8))
    
    # Fill with random if needed
    while len(pop) < 40:
        pop.append(rng.randint(0, 2, size=N).astype(np.int8))
    
    return pop[:40]

def construct():
    """
    Genetic algorithm with vectorized local search to minimize monochromatic 4-term APs.
    Returns 256 colours in {0, 1}.
    """
    start_time = time.time()
    time_limit = 115
    
    rng = np.random.RandomState(42)
    pop_size = 40
    
    # Initialize diverse population
    population = _generate_diverse_initial(rng)
    scores = [_count_mono_aps(c) for c in population]
    
    # Local search on top candidates
    top_k = 15
    top_indices = np.argsort(scores)[:top_k]
    for idx in top_indices:
        improved_c, improved_s = _local_search(population[idx], max_sweeps=25, rng=rng)
        population[idx] = improved_c
        scores[idx] = improved_s
    
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    best_individual = population[best_idx].copy()
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        elite_size = 6
        new_population = [population[i].copy() for i in range(elite_size)]
        new_scores = scores[:elite_size]
        
        while len(new_population) < pop_size:
            # Tournament selection
            cands = rng.randint(0, len(population), size=4)
            p1 = population[int(min(cands, key=lambda i: scores[i]))]
            p2 = population[int(min(cands, key=lambda i: scores[i]))]
            
            # Crossover: uniform with bias toward better parent
            better = p1 if scores[int(min(cands, key=lambda i: scores[i]))] <= scores[int(cands[1])] else p2
            worse = p2 if better is p1 else p1
            
            # Weighted crossover: 65% from better, 35% from worse
            mask = (rng.rand(N) < 0.65).astype(np.int8)
            child = np.where(mask == 1, better, worse).copy()
            
            # Adaptive mutation: more mutations if stuck
            num_mut = rng.randint(1, 6)
            mut_positions = rng.choice(N, size=num_mut, replace=False)
            child[mut_positions] = 1 - child[mut_positions]
            
            # Local search
            child, child_score = _local_search(child, max_sweeps=25, rng=rng)
            
            new_population.append(child)
            new_scores.append(child_score)
        
        population = new_population
        scores = new_scores
        
        # Track best
        cur_best_idx = int(np.argmin(scores))
        if scores[cur_best_idx] < best_score:
            best_score = scores[cur_best_idx]
            best_individual = population[cur_best_idx].copy()
        
        # Periodic diversity injection
        if generation % 8 == 0 and time.time() - start_time < time_limit - 15:
            for _ in range(4):
                c = rng.randint(0, 2, size=N).astype(np.int8)
                c, s = _local_search(c, max_sweeps=20, rng=rng)
                worst_idx = int(np.argmax(scores))
                population[worst_idx] = c
                scores[worst_idx] = s
                if s < best_score:
                    best_score = s
                    best_individual = c.copy()
            
            order = np.argsort(scores)
            population = [population[i] for i in order]
            scores = [scores[i] for i in order]
        
        # Also try mutating the best individual occasionally
        if generation % 5 == 0 and time.time() - start_time < time_limit - 10:
            c = best_individual.copy()
            mut_pos = rng.choice(N, size=rng.randint(3, 8), replace=False)
            c[mut_pos] = 1 - c[mut_pos]
            c, s = _local_search(c, max_sweeps=20, rng=rng)
            worst_idx = int(np.argmax(scores))
            if s < scores[worst_idx]:
                population[worst_idx] = c
                scores[worst_idx] = s
                if s < best_score:
                    best_score = s
                    best_individual = c.copy()
    
    return [int(x) for x in best_individual]

if __name__ == "__main__":
    result = construct()
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    count = _count_mono_aps(np.array(result, dtype=np.int8))
    print(f"Monochromatic 4-APs: {count}")
