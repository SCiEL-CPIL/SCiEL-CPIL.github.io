"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from array import array
import numpy as np

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
    neighbor_sets = []
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
        neighbor_sets.append(frozenset(idxs))
    
    # Convert to numpy for faster operations
    neighbors_np = np.array(neighbors, dtype=np.int32)
    
    # Greedy initialization with multiple strategies
    def greedy_init(seed, strategy=0):
        random.seed(seed)
        counts = np.zeros(N, dtype=np.int32)
        covered = 0
        code = []
        code_set = set()
        uncovered = N
        used = np.zeros(N, dtype=bool)
        
        if strategy == 0:
            # Standard greedy: pick word covering most uncovered
            while uncovered > 0:
                best_idx = -1
                max_cov = 0
                for w_idx in range(N):
                    if used[w_idx]:
                        continue
                    nbs = neighbors_np[w_idx]
                    cov = np.sum(counts[nbs] == 0)
                    if cov > max_cov:
                        max_cov = cov
                        best_idx = w_idx
                if best_idx == -1 or max_cov == 0:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                used[best_idx] = True
                nbs = neighbors_np[best_idx]
                new_uncov = counts[nbs] == 0
                covered += int(np.sum(new_uncov))
                uncovered -= int(np.sum(new_uncov))
                counts[nbs] += 1
        elif strategy == 1:
            # Greedy with random tie-breaking
            while uncovered > 0:
                best_cov = 0
                candidates = []
                for w_idx in range(N):
                    if used[w_idx]:
                        continue
                    nbs = neighbors_np[w_idx]
                    cov = int(np.sum(counts[nbs] == 0))
                    if cov > best_cov:
                        best_cov = cov
                        candidates = [w_idx]
                    elif cov == best_cov and cov > 0:
                        candidates.append(w_idx)
                if not candidates or best_cov == 0:
                    break
                best_idx = random.choice(candidates)
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                used[best_idx] = True
                nbs = neighbors_np[best_idx]
                new_uncov = counts[nbs] == 0
                covered += int(np.sum(new_uncov))
                uncovered -= int(np.sum(new_uncov))
                counts[nbs] += 1
        elif strategy == 2:
            # Greedy with overlap penalty: prefer words that don't overlap much with existing
            while uncovered > 0:
                best_idx = -1
                best_score = -1
                for w_idx in range(N):
                    if used[w_idx]:
                        continue
                    nbs = neighbors_np[w_idx]
                    cov = int(np.sum(counts[nbs] == 0))
                    if cov == 0:
                        continue
                    # Penalty for words that are neighbors of many existing codewords
                    overlap = 0
                    for c_idx in code_set:
                        if w_idx in neighbor_sets[c_idx]:
                            overlap += 1
                    score = cov - 0.1 * overlap
                    if score > best_score:
                        best_score = score
                        best_idx = w_idx
                if best_idx == -1:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                used[best_idx] = True
                nbs = neighbors_np[best_idx]
                new_uncov = counts[nbs] == 0
                covered += int(np.sum(new_uncov))
                uncovered -= int(np.sum(new_uncov))
                counts[nbs] += 1
        else:
            # Randomized greedy: with probability p, pick best, else pick from top-k
            p = 0.8
            k = 5
            while uncovered > 0:
                scores = []
                for w_idx in range(N):
                    if used[w_idx]:
                        continue
                    nbs = neighbors_np[w_idx]
                    cov = int(np.sum(counts[nbs] == 0))
                    scores.append((cov, w_idx))
                scores.sort(reverse=True)
                if not scores or scores[0][0] == 0:
                    break
                if random.random() < p:
                    best_idx = scores[0][1]
                else:
                    top_k = scores[:min(k, len(scores))]
                    best_idx = random.choice([x[1] for x in top_k])
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                used[best_idx] = True
                nbs = neighbors_np[best_idx]
                new_uncov = counts[nbs] == 0
                covered += int(np.sum(new_uncov))
                uncovered -= int(np.sum(new_uncov))
                counts[nbs] += 1
        
        return code, counts, covered, code_set
    
    # Aggressive local search: try to remove as many codewords as possible
    def local_search_reduce(code, counts, code_set, start_time, time_limit):
        size = len(code)
        covered = int(np.sum(counts > 0))
        
        # Repeatedly try to remove codewords that don't uncover anything
        progress = True
        while progress and time.time() - start_time < time_limit:
            progress = False
            indices = list(range(size))
            random.shuffle(indices)
            for idx in indices:
                if idx >= size:
                    continue
                w = code[idx]
                w_idx = word_to_idx[w]
                nbs = neighbors_np[w_idx]
                # Check if any neighbor would become uncovered
                if np.any(counts[nbs] == 1):
                    continue
                # Safe to remove
                code.pop(idx)
                code_set.remove(w_idx)
                counts[nbs] -= 1
                size -= 1
                progress = True
                break  # Restart scan after removal
        
        return code, counts, code_set, size, covered
    
    # Simulated annealing
    def simulated_annealing(init_code, init_counts, init_covered, init_code_set, start_time, time_limit, seed):
        random.seed(seed)
        code = list(init_code)
        counts = init_counts.copy()
        covered = init_covered
        size = len(code)
        code_set = set(init_code_set)
        
        P = 15.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9999
        all_indices = list(range(N))
        
        # Track acceptance rates for adaptive cooling
        accept_count = 0
        total_moves = 0
        last_adapt = time.time()
        
        while time.time() - start_time < time_limit:
            # Adaptive cooling based on acceptance rate
            if time.time() - last_adapt > 2.0:
                last_adapt = time.time()
                if total_moves > 1000:
                    acc_rate = accept_count / total_moves
                    # Target acceptance rate ~0.3
                    if acc_rate > 0.5:
                        cooling = min(0.99995, cooling * 1.02)
                    elif acc_rate < 0.1:
                        cooling = max(0.9997, cooling * 0.98)
                total_moves = 0
                accept_count = 0
            
            # Periodic local search reduction
            if size > 0 and random.random() < 0.05:
                code, counts, code_set, size, covered = local_search_reduce(
                    code, counts, code_set, start_time, time_limit
                )
                current_cost = cost(size, covered)
                if covered == N and size < best_size:
                    best_size = size
                    best_covered = covered
                    best_code = list(code)
                    best_cost = current_cost
            
            move_type = random.random()
            total_moves += 1
            
            if move_type < 0.6:
                # Swap: replace one codeword with another
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
                
                nbs_remove = neighbors_np[idx_w_remove]
                nbs_add = neighbors_np[idx_add]
                nbs_add_set = neighbor_sets[idx_add]
                
                # Calculate delta coverage
                losing = int(np.sum((counts[nbs_remove] == 1) & ~np.isin(nbs_remove, nbs_add_set)))
                gaining = int(np.sum(counts[nbs_add] == 0))
                
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(idx_add)
                    counts[nbs_remove] -= 1
                    counts[nbs_add] += 1
                    covered = new_covered
                    current_cost = new_cost
                    accept_count += 1
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type < 0.85:
                # Remove a codeword
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                w = code[idx]
                w_idx = word_to_idx[w]
                nbs = neighbors_np[w_idx]
                
                delta_covered = -int(np.sum(counts[nbs] == 1))
                new_size = size - 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    counts[nbs] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    accept_count += 1
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                # Add a codeword (only if not fully covered, or as part of swap)
                if covered == N and size > 0:
                    # Convert to swap
                    idx_remove = random.randint(0, size - 1)
                    idx_w_remove = word_to_idx[code[idx_remove]]
                    idx_add = random.choice(all_indices)
                    if idx_add == idx_w_remove or idx_add in code_set:
                        T *= cooling
                        if T < T_min: T = T_min
                        continue
                    nbs_remove = neighbors_np[idx_w_remove]
                    nbs_add = neighbors_np[idx_add]
                    nbs_add_set = neighbor_sets[idx_add]
                    
                    losing = int(np.sum((counts[nbs_remove] == 1) & ~np.isin(nbs_remove, nbs_add_set)))
                    gaining = int(np.sum(counts[nbs_add] == 0))
                    
                    delta = gaining - losing
                    new_covered = covered + delta
                    new_cost = cost(size, new_covered)
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                        code[idx_remove] = all_words[idx_add]
                        code_set.remove(idx_w_remove)
                        code_set.add(idx_add)
                        counts[nbs_remove] -= 1
                        counts[nbs_add] += 1
                        covered = new_covered
                        current_cost = new_cost
                        accept_count += 1
                        
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
                for _ in range(15):
                    idx_add = random.choice(all_indices)
                    if idx_add in code_set:
                        continue
                    nbs = neighbors_np[idx_add]
                    gain = int(np.sum(counts[nbs] == 0))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors_np[best_idx_add]
                delta_covered = int(np.sum(counts[nbs_add] == 0))
                new_size = size + 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(all_words[best_idx_add])
                    code_set.add(best_idx_add)
                    counts[nbs_add] += 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    accept_count += 1
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        # Final local search reduction
        best_code, _, _, final_size, _ = local_search_reduce(
            best_code, 
            np.zeros(N, dtype=np.int32), 
            set(word_to_idx[w] for w in best_code),
            start_time, time_limit
        )
        # Recount coverage for final code
        final_counts = np.zeros(N, dtype=np.int32)
        for w in best_code:
            w_idx = word_to_idx[w]
            final_counts[neighbors_np[w_idx]] += 1
        final_covered = int(np.sum(final_counts > 0))
        if final_covered == N and final_size < best_size:
            best_size = final_size
            best_covered = final_covered
        
        return best_code, best_size
    
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-restart with different strategies
    best_overall = None
    best_overall_size = float('inf')
    
    # Try different strategies and seeds
    configs = [
        (42, 0), (43, 1), (44, 2), (45, 3),
        (100, 0), (101, 1), (102, 2), (103, 3)
    ]
    
    num_restarts = min(6, len(configs))
    time_per_restart = time_limit / num_restarts
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 10:
            break
        seed, strategy = configs[r]
        init_code, init_counts, init_covered, init_code_set = greedy_init(seed, strategy)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 5:
            break
        
        # Do local search reduction before SA
        init_code, init_counts, init_code_set, init_size, init_covered = local_search_reduce(
            init_code, init_counts, init_code_set, start_time, time_limit
        )
        
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set,
            start_time, time_limit, 200 + r
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    if best_overall is None:
        # Fallback
        init_code, _, _, _ = greedy_init(42, 0)
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
