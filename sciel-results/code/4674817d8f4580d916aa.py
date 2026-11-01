"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from array import array

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute neighbors
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 2187
    
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(tuple(idxs))
    
    # Greedy initialization with multiple strategies
    def greedy_init(seed, strategy='random'):
        random.seed(seed)
        counts = [0] * N
        covered = 0
        code = []
        code_set = set()
        uncovered = N
        
        if strategy == 'random':
            # Standard greedy: pick word covering most uncovered
            while uncovered > 0:
                best_idx = -1
                max_cov = 0
                for w_idx in range(N):
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_idx = w_idx
                if best_idx == -1:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    if counts[nb] == 0:
                        covered += 1
                        uncovered -= 1
                    counts[nb] += 1
        elif strategy == 'structured':
            # Use symmetry: pick words with specific patterns
            # Start with words that have many 0s
            candidates = sorted(range(N), key=lambda i: sum(1 for x in all_words[i] if x == 0), reverse=True)
            random.shuffle(candidates)
            for w_idx in candidates:
                if uncovered == 0:
                    break
                cov = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 0:
                        cov += 1
                if cov > 0:
                    code.append(all_words[w_idx])
                    code_set.add(w_idx)
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            covered += 1
                            uncovered -= 1
                        counts[nb] += 1
        elif strategy == 'mixed':
            # Mix of random and structured
            for _ in range(50):
                if uncovered == 0:
                    break
                # Pick from top 100 candidates by coverage
                scored = []
                for w_idx in random.sample(range(N), min(100, N)):
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    scored.append((cov, w_idx))
                if not scored:
                    break
                scored.sort(reverse=True)
                # Pick from top 5 with some randomness
                top_k = min(5, len(scored))
                best_cov, best_idx = scored[random.randint(0, top_k - 1)]
                if best_cov == 0:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    if counts[nb] == 0:
                        covered += 1
                        uncovered -= 1
                    counts[nb] += 1
            # Fill remaining with standard greedy
            while uncovered > 0:
                best_idx = -1
                max_cov = 0
                for w_idx in range(N):
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_idx = w_idx
                if best_idx == -1:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    if counts[nb] == 0:
                        covered += 1
                        uncovered -= 1
                    counts[nb] += 1
        
        return code, counts, covered, code_set
    
    # Simulated annealing with improved move generation
    def simulated_annealing(init_code, init_counts, init_covered, init_code_set, start_time, time_limit, seed, initial_temp=15.0):
        random.seed(seed)
        code = list(init_code)
        counts = list(init_counts)
        covered = init_covered
        size = len(code)
        code_set = set(init_code_set)
        
        P = 10.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = initial_temp
        T_min = 0.001
        cooling = 0.99992
        
        all_indices = list(range(N))
        
        # Periodic greedy reduction
        last_reduction = time.time()
        reduction_interval = 1.0
        
        # Track recent moves for diversity
        recent_moves = []
        
        while time.time() - start_time < time_limit:
            # Periodic greedy reduction: try to remove redundant codewords
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                # Try removing each codeword in random order, multiple passes
                for _ in range(3):
                    indices = list(range(size))
                    random.shuffle(indices)
                    removed_any = False
                    for idx in indices:
                        if idx >= size:
                            continue
                        w = code[idx]
                        w_idx = word_to_idx[w]
                        would_uncover = 0
                        for nb in neighbors[w_idx]:
                            if counts[nb] == 1:
                                would_uncover += 1
                        if would_uncover == 0:
                            code.pop(idx)
                            code_set.remove(w_idx)
                            for nb in neighbors[w_idx]:
                                counts[nb] -= 1
                            size -= 1
                            current_cost = cost(size, covered)
                            if size < best_size:
                                best_size = size
                                best_code = list(code)
                                best_covered = covered
                                best_cost = current_cost
                            removed_any = True
                            break
                    if not removed_any:
                        break
            
            move_type = random.random()
            
            if move_type < 0.60:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                idx_add = random.choice(all_indices)
                
                if idx_add == idx_w_remove or idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbors[idx_w_remove]
                nbs_add = neighbors[idx_add]
                
                # Calculate delta coverage
                losing = 0
                for nb in nbs_remove:
                    if counts[nb] == 1 and nb not in set(nbs_add):
                        losing += 1
                gaining = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        gaining += 1
                
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(idx_add)
                    for nb in nbs_remove: counts[nb] -= 1
                    for nb in nbs_add: counts[nb] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    recent_moves.append(('swap', idx_remove, idx_add))
                    if len(recent_moves) > 50:
                        recent_moves.pop(0)
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type < 0.85:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                w = code[idx]
                w_idx = word_to_idx[w]
                
                delta_covered = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        delta_covered -= 1
                
                new_size = size - 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors[w_idx]:
                        counts[nb] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    recent_moves.append(('remove', idx))
                    if len(recent_moves) > 50:
                        recent_moves.pop(0)
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                # Add
                if covered == N:
                    # Convert to swap when fully covered
                    if size > 0:
                        idx_remove = random.randint(0, size - 1)
                        idx_w_remove = word_to_idx[code[idx_remove]]
                        idx_add = random.choice(all_indices)
                        if idx_add == idx_w_remove or idx_add in code_set:
                            T *= cooling
                            if T < T_min: T = T_min
                            continue
                        nbs_remove = neighbors[idx_w_remove]
                        nbs_add = neighbors[idx_add]
                        
                        losing = 0
                        for nb in nbs_remove:
                            if counts[nb] == 1 and nb not in set(nbs_add):
                                losing += 1
                        gaining = 0
                        for nb in nbs_add:
                            if counts[nb] == 0:
                                gaining += 1
                        
                        delta = gaining - losing
                        new_covered = covered + delta
                        new_cost = cost(size, new_covered)
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            code[idx_remove] = all_words[idx_add]
                            code_set.remove(idx_w_remove)
                            code_set.add(idx_add)
                            for nb in nbs_remove: counts[nb] -= 1
                            for nb in nbs_add: counts[nb] += 1
                            covered = new_covered
                            current_cost = new_cost
                            
                            if covered == N and size < best_size:
                                best_size = size
                                best_covered = covered
                                best_code = list(code)
                                best_cost = current_cost
                        T *= cooling
                        if T < T_min: T = T_min
                    continue
                
                # Pick word with best marginal gain
                best_idx_add = -1
                best_gain = 0
                # Sample from a larger pool
                for _ in range(20):
                    idx_add = random.choice(all_indices)
                    if idx_add in code_set:
                        continue
                    gain = 0
                    for nb in neighbors[idx_add]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors[best_idx_add]
                delta_covered = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        delta_covered += 1
                        
                new_size = size + 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(all_words[best_idx_add])
                    code_set.add(best_idx_add)
                    for nb in nbs_add:
                        counts[nb] += 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    recent_moves.append(('add', best_idx_add))
                    if len(recent_moves) > 50:
                        recent_moves.pop(0)
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        return best_code, best_size
    
    # Iterated local search: perturb best solution and re-optimize
    def perturb(code, counts, covered, code_set, num_swaps):
        """Create a perturbed version of the code."""
        perturbed_code = list(code)
        perturbed_counts = list(counts)
        perturbed_covered = covered
        perturbed_code_set = set(code_set)
        
        for _ in range(num_swaps):
            if len(perturbed_code) == 0:
                break
            # Pick a random codeword to replace
            idx = random.randint(0, len(perturbed_code) - 1)
            old_word = perturbed_code[idx]
            old_idx = word_to_idx[old_word]
            
            # Pick a new random word
            new_idx = random.choice(range(N))
            if new_idx in perturbed_code_set:
                continue
            
            # Remove old
            for nb in neighbors[old_idx]:
                perturbed_counts[nb] -= 1
                if perturbed_counts[nb] == 0:
                    perturbed_covered -= 1
            perturbed_code_set.remove(old_idx)
            
            # Add new
            perturbed_code[idx] = all_words[new_idx]
            perturbed_code_set.add(new_idx)
            for nb in neighbors[new_idx]:
                if perturbed_counts[nb] == 0:
                    perturbed_covered += 1
                perturbed_counts[nb] += 1
        
        return perturbed_code, perturbed_counts, perturbed_covered, perturbed_code_set
    
    start_time = time.time()
    time_limit = 115.0
    
    # Phase 1: Multiple greedy initializations with different strategies
    best_overall = None
    best_overall_size = float('inf')
    
    strategies = ['random', 'structured', 'mixed', 'random', 'mixed', 'structured']
    seeds = [42, 123, 456, 789, 1011, 2023]
    
    for i, (strategy, seed) in enumerate(zip(strategies, seeds)):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 15:
            break
        init_code, init_counts, init_covered, init_code_set = greedy_init(seed, strategy)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 10:
            break
        sa_time = min(sa_remaining - 2, 15.0)
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set,
            start_time, t1 + sa_time, 100 + i, initial_temp=15.0
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    # Phase 2: Iterated local search on the best solution
    if best_overall is not None:
        # Rebuild counts for the best solution
        best_counts = [0] * N
        best_covered = 0
        best_code_set = set()
        for w in best_overall:
            idx = word_to_idx[w]
            best_code_set.add(idx)
            for nb in neighbors[idx]:
                if best_counts[nb] == 0:
                    best_covered += 1
                best_counts[nb] += 1
        
        # ILS loop
        remaining_time = time_limit - (time.time() - start_time)
        ils_count = 0
        while remaining_time > 8:
            if best_overall is None:
                break
            
            # Perturb the best solution
            num_swaps = random.randint(2, 6)
            perturbed_code, perturbed_counts, perturbed_covered, perturbed_code_set = perturb(
                best_overall, best_counts, best_covered, best_code_set, num_swaps
            )
            
            t_start = time.time()
            sa_duration = min(12.0, remaining_time - 5.0)
            if sa_duration < 2:
                break
            
            best_code, best_size = simulated_annealing(
                perturbed_code, perturbed_counts, perturbed_covered, perturbed_code_set,
                t_start, t_start + sa_duration, random.randint(1000, 9999), initial_temp=10.0
            )
            
            if best_size < best_overall_size:
                best_overall_size = best_size
                best_overall = best_code
                # Rebuild counts
                best_counts = [0] * N
                best_covered = 0
                best_code_set = set()
                for w in best_overall:
                    idx = word_to_idx[w]
                    best_code_set.add(idx)
                    for nb in neighbors[idx]:
                        if best_counts[nb] == 0:
                            best_covered += 1
                        best_counts[nb] += 1
                ils_count += 1
            
            remaining_time = time_limit - (time.time() - start_time)
    
    if best_overall is None:
        # Fallback
        init_code, _, _, _ = greedy_init(42, 'random')
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
