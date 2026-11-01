"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a refined Simulated Annealing approach with multiple restarts.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precomputation for q=3, n=7, r=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N_WORDS = 3**7
    
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
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
        neighbors.append(idxs)
        
    random.seed(42)
    
    # 1. Greedy Initialization: Max Coverage Greedy
    counts = [0] * N_WORDS
    covered = 0
    code = []
    uncovered_set = set(range(N_WORDS))
    
    while uncovered_set:
        best_word_idx = -1
        max_cov = 0
        
        for w_idx in range(N_WORDS):
            cov = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered_set:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_word_idx = w_idx
        
        if best_word_idx == -1:
            break
            
        code.append(all_words[best_word_idx])
        for nb in neighbors[best_word_idx]:
            if counts[nb] == 0:
                covered += 1
                uncovered_set.remove(nb)
            counts[nb] += 1
            
    # Start Simulated Annealing with Restarts
    best_code = list(code)
    best_size = len(code)
    best_covered = covered
    
    start_time = time.time()
    time_limit = 115.0
    
    P = 10.0 # Penalty for uncovered
    
    def get_cost(size, cov):
        return size + P * (N_WORDS - cov)
        
    # We will run multiple SA cycles.
    # Each cycle runs until no improvement for a while or time runs out.
    # We restart from the best solution found so far.
    
    while time.time() - start_time < time_limit:
        # Reset SA state to best known
        current_code = list(best_code)
        current_counts = [0] * N_WORDS
        current_covered = 0
        current_size = len(current_code)
        
        for w in current_code:
            w_idx = word_to_idx[w]
            for nb in neighbors[w_idx]:
                current_counts[nb] += 1
        for nb in range(N_WORDS):
            if current_counts[nb] > 0:
                current_covered += 1
                
        current_cost = get_cost(current_size, current_covered)
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        no_improve_steps = 0
        max_steps_without_improve = 5000
        
        while time.time() - start_time < time_limit and no_improve_steps < max_steps_without_improve:
            move_type = random.random()
            
            if move_type < 0.6:
                # Try to Remove
                if not current_code:
                    continue
                idx = random.randint(0, current_size - 1)
                w = current_code[idx]
                w_idx = word_to_idx[w]
                
                delta_covered = 0
                for nb in neighbors[w_idx]:
                    if current_counts[nb] == 1:
                        delta_covered -= 1
                
                new_size = current_size - 1
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code.pop(idx)
                    for nb in neighbors[w_idx]:
                        current_counts[nb] -= 1
                    current_size = new_size
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == N_WORDS and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                else:
                    no_improve_steps += 1
                    T *= cooling
                    if T < T_min: T = T_min
                    continue

            elif move_type < 0.95:
                # Replace
                if not current_code:
                    continue
                idx = random.randint(0, current_size - 1)
                old_w = current_code[idx]
                old_w_idx = word_to_idx[old_w]
                
                # Choose new word
                if current_covered < N_WORDS:
                    u_idx = random.randint(0, N_WORDS - 1)
                    if current_counts[u_idx] == 0:
                        new_w_idx = random.choice(neighbors[u_idx])
                    else:
                        new_w_idx = random.randint(0, N_WORDS - 1)
                else:
                    new_w_idx = random.randint(0, N_WORDS - 1)
                
                new_w = all_words[new_w_idx]
                
                if new_w_idx == old_w_idx:
                    continue
                
                old_nbs = neighbors[old_w_idx]
                new_nbs = neighbors[new_w_idx]
                
                old_set = set(old_nbs)
                new_set = set(new_nbs)
                
                delta_covered = 0
                for nb in old_set:
                    if nb not in new_set:
                        if current_counts[nb] == 1:
                            delta_covered -= 1
                            
                for nb in new_set:
                    if nb not in old_set:
                        if current_counts[nb] == 0:
                            delta_covered += 1
                            
                new_size = current_size
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code[idx] = new_w
                    for nb in old_nbs:
                        current_counts[nb] -= 1
                    for nb in new_nbs:
                        current_counts[nb] += 1
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == N_WORDS and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                else:
                    no_improve_steps += 1
                    T *= cooling
                    if T < T_min: T = T_min
                    continue

            else:
                # Add
                if current_covered == N_WORDS:
                    continue
                
                u_idx = random.randint(0, N_WORDS - 1)
                if current_counts[u_idx] == 0:
                    new_w_idx = random.choice(neighbors[u_idx])
                else:
                    new_w_idx = random.randint(0, N_WORDS - 1)
                    
                new_w = all_words[new_w_idx]
                
                if new_w in current_code:
                    continue
                
                new_nbs = neighbors[new_w_idx]
                delta_covered = 0
                for nb in new_nbs:
                    if current_counts[nb] == 0:
                        delta_covered += 1
                
                new_size = current_size + 1
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code.append(new_w)
                    for nb in new_nbs:
                        current_counts[nb] += 1
                    current_size = new_size
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == N_WORDS and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                else:
                    no_improve_steps += 1
                    T *= cooling
                    if T < T_min: T = T_min
                    continue

    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
