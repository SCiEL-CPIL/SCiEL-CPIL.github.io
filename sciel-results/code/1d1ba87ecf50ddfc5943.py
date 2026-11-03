import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1) using Simulated Annealing
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    def is_valid_cover(chosen_indices):
        if not chosen_indices:
            return False
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def get_uncovered(chosen_indices):
        """Return set of uncovered points."""
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.where(cover_count == 0)[0]

    def greedy_initial_cover():
        """Fast greedy to get a valid starting point."""
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        while not covered.all():
            if time.time() - start_time > time_limit - 30.0:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            candidates = np.where(newly == max_new)[0]
            pick = candidates[random.randint(0, len(candidates) - 1)]
            best_global = avail_indices[pick]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
            
        return chosen

    def simulated_annealing(initial_cover):
        """
        Simulated Annealing for covering code optimization.
        State: set of codewords (as a list of indices)
        Energy: penalty for uncovered points (heavy) + number of codewords (light)
        Moves: remove, add, swap, 2-swap
        """
        current = list(initial_cover)
        current_cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            current_cover_count[neighbors[ci]] += 1
        current_uncovered = np.sum(current_cover_count == 0)
        current_energy = current_uncovered * 100 + len(current)
        
        best = list(current)
        best_energy = current_energy
        best_size = len(current)
        
        # SA parameters
        T = 50.0
        T_min = 0.01
        cooling = 0.995
        max_iters_per_T = 200
        
        rng = np.random.RandomState(42)
        
        while T > T_min and time.time() - start_time < time_limit - 5.0:
            for _ in range(max_iters_per_T):
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Choose a random move type
                move_type = rng.randint(0, 4)
                
                if move_type == 0 and len(current) > 1:
                    # Remove a random codeword
                    idx = rng.randint(0, len(current) - 1)
                    ci = current[idx]
                    # Check if removal causes uncovered points
                    new_uncovered = np.sum((current_cover_count[neighbors[ci]] == 1))
                    delta_energy = new_uncovered * 100 - 1
                    if delta_energy < 0 or (T > 0.01 and rng.random() < np.exp(-delta_energy / T)):
                        current.pop(idx)
                        current_cover_count[neighbors[ci]] -= 1
                        current_energy += delta_energy
                        current_uncovered += new_uncovered
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best = list(current)
                            best_size = len(current)
                
                elif move_type == 1:
                    # Add a random codeword
                    ci = rng.randint(0, total_points - 1)
                    if ci not in current:
                        # Check how many uncovered points it covers
                        covered_uncovered = np.sum((current_cover_count[neighbors[ci]] == 0))
                        delta_energy = -covered_uncovered * 100 + 1
                        if delta_energy < 0 or (T > 0.01 and rng.random() < np.exp(-delta_energy / T)):
                            current.append(ci)
                            current_cover_count[neighbors[ci]] += 1
                            current_energy += delta_energy
                            current_uncovered -= covered_uncovered
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best = list(current)
                                best_size = len(current)
                
                elif move_type == 2 and len(current) > 0:
                    # Swap: remove one, add another
                    idx = rng.randint(0, len(current) - 1)
                    ci_remove = current[idx]
                    ci_add = rng.randint(0, total_points - 1)
                    if ci_add == ci_remove or ci_add in current:
                        continue
                    
                    # Compute delta energy for swap
                    # Remove ci_remove: some points lose coverage
                    # Add ci_add: some points gain coverage
                    # Net effect on uncovered count
                    exposed = neighbors[ci_remove]
                    would_uncover = np.sum((current_cover_count[exposed] == 1))
                    would_cover = np.sum((current_cover_count[neighbors[ci_add]] == 0))
                    
                    # More precise: compute actual change
                    # Temporarily apply
                    old_cover = current_cover_count.copy()
                    current_cover_count[neighbors[ci_remove]] -= 1
                    current_cover_count[neighbors[ci_add]] += 1
                    new_uncovered = np.sum(current_cover_count == 0)
                    delta_energy = (new_uncovered - current_uncovered) * 100
                    
                    if delta_energy < 0 or (T > 0.01 and rng.random() < np.exp(-delta_energy / T)):
                        current[idx] = ci_add
                        current_energy += delta_energy
                        current_uncovered = new_uncovered
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best = list(current)
                            best_size = len(current)
                    else:
                        # Revert
                        current_cover_count = old_cover
                
                elif move_type == 3 and len(current) >= 2:
                    # 2-remove: try to remove two codewords at once
                    idx1, idx2 = rng.choice(len(current), size=2, replace=False)
                    ci1, ci2 = current[idx1], current[idx2]
                    
                    old_cover = current_cover_count.copy()
                    current_cover_count[neighbors[ci1]] -= 1
                    current_cover_count[neighbors[ci2]] -= 1
                    new_uncovered = np.sum(current_cover_count == 0)
                    delta_energy = (new_uncovered - current_uncovered) * 100 - 2
                    
                    if delta_energy < 0 or (T > 0.01 and rng.random() < np.exp(-delta_energy / T)):
                        # Remove both (keep higher index first to avoid shifting)
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        current.pop(idx1)
                        current.pop(idx2)
                        current_energy += delta_energy
                        current_uncovered = new_uncovered
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best = list(current)
                            best_size = len(current)
                    else:
                        current_cover_count = old_cover
                
                elif move_type == 4 and len(current) > 0:
                    # 2-swap: remove two, add two
                    idx1, idx2 = rng.choice(len(current), size=2, replace=False)
                    ci1, ci2 = current[idx1], current[idx2]
                    ci3, ci4 = rng.randint(0, total_points - 1), rng.randint(0, total_points - 1)
                    if ci3 in (ci1, ci2) or ci4 in (ci1, ci2, ci3):
                        continue
                    
                    old_cover = current_cover_count.copy()
                    current_cover_count[neighbors[ci1]] -= 1
                    current_cover_count[neighbors[ci2]] -= 1
                    current_cover_count[neighbors[ci3]] += 1
                    current_cover_count[neighbors[ci4]] += 1
                    new_uncovered = np.sum(current_cover_count == 0)
                    delta_energy = (new_uncovered - current_uncovered) * 100
                    
                    if delta_energy < 0 or (T > 0.01 and rng.random() < np.exp(-delta_energy / T)):
                        current[idx1] = ci3
                        current[idx2] = ci4
                        current_energy += delta_energy
                        current_uncovered = new_uncovered
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best = list(current)
                            best_size = len(current)
                    else:
                        current_cover_count = old_cover
                
                T *= cooling
        
        return best

    def local_search_reduce(chosen_indices):
        """Aggressive removal-based local search to minimize size."""
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1
        
        improved = True
        max_rounds = 50
        round_num = 0
        
        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 2.0:
            improved = False
            round_num += 1
            order = list(range(len(current)))
            random.shuffle(order)
            
            for i in order:
                if i >= len(current):
                    continue
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                ci = current[i]
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]
                
                if len(would_uncover) == 0:
                    cover_count[neighbors[ci]] -= 1
                    current.pop(i)
                    improved = True
        
        return current

    # Main algorithm: multiple restarts of SA + local search
    best_result = None
    best_size = float('inf')
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Generate initial cover
        initial = greedy_initial_cover()
        
        if not is_valid_cover(initial):
            continue
        
        # Simulated annealing
        sa_result = simulated_annealing(initial)
        
        # Local search to reduce further
        optimized = local_search_reduce(sa_result)
        
        # Verify validity
        if not is_valid_cover(optimized):
            # Try to repair
            uncovered = get_uncovered(optimized)
            for pt in uncovered:
                # Find a codeword that covers pt
                for ci in range(total_points):
                    if pt in neighbors[ci]:
                        if ci not in optimized:
                            optimized.append(ci)
                            break
                if time.time() - start_time > time_limit - 2.0:
                    break
        
        if not is_valid_cover(optimized):
            continue
        
        current_size = len(optimized)
        
        if current_size < best_size:
            best_size = current_size
            best_result = list(optimized)
        
        if best_size <= 410:
            break
    
    # Fallback if no valid result
    if best_result is None:
        initial = greedy_initial_cover()
        if is_valid_cover(initial):
            best_result = local_search_reduce(initial)
            best_size = len(best_result)
        else:
            # Emergency full greedy
            covered = np.zeros(total_points, dtype=bool)
            chosen = []
            available = np.ones(total_points, dtype=bool)
            while not covered.all():
                avail_indices = np.where(available)[0]
                if len(avail_indices) == 0:
                    break
                neigh_avail = neighbors[avail_indices]
                is_covered = covered[neigh_avail]
                newly = (~is_covered).sum(axis=1)
                max_new = np.max(newly)
                if max_new == 0:
                    break
                candidates = np.where(newly == max_new)[0]
                pick = candidates[random.randint(0, len(candidates) - 1)]
                best_global = avail_indices[pick]
                chosen.append(best_global)
                covered[neighbors[best_global]] = True
                available[best_global] = False
            best_result = chosen
    
    # Final verification
    if best_result is not None and not is_valid_cover(best_result):
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        while not covered.all():
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            candidates = np.where(newly == max_new)[0]
            pick = candidates[random.randint(0, len(candidates) - 1)]
            best_global = avail_indices[pick]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        best_result = chosen
    
    return [all_words[i] for i in best_result]
