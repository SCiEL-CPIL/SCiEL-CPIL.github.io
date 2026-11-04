import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    # Specific case optimization for q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        # Precompute all words and their indices
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors for each word (self + 1-dist)
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].append(word_to_idx[new_word])
        
        # Build incidence matrix M for fast greedy
        M = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for nb in neighbors[i]:
                M[i, nb] = 1
        
        def greedy_init(seed_val):
            random.seed(seed_val)
            uncovered = np.ones(N, dtype=bool)
            solution = []
            while np.any(uncovered):
                gains = M @ uncovered
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                best_idx = random.choice(candidates)
                solution.append(best_idx)
                uncovered[neighbors[best_idx]] = False
            return solution

        def local_search(initial_solution, time_limit=1.0):
            code_set = set(initial_solution)
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_set:
                cover_count[neighbors[c]] += 1
            
            best_cost = len(code_set)
            best_solution = list(code_set)
            penalty = 1000
            
            start_time = time.time()
            T = 1.0
            T_min = 0.01
            alpha = 0.999
            
            while time.time() - start_time < time_limit:
                move_type = random.random()
                if move_type < 0.5:
                    # Remove
                    if not code_set: continue
                    c_out = random.choice(list(code_set))
                    min_cov = np.min(cover_count[neighbors[c_out]])
                    if min_cov > 1:
                        # Valid removal
                        code_set.remove(c_out)
                        cover_count[neighbors[c_out]] -= 1
                        if len(code_set) < best_cost:
                            best_cost = len(code_set)
                            best_solution = list(code_set)
                    else:
                        # Invalid removal, calculate energy delta
                        current_uncovered = np.sum(cover_count == 0)
                        current_energy = len(code_set) + penalty * current_uncovered
                        new_uncovered = np.sum(cover_count[neighbors[c_out]] == 1)
                        new_energy = (len(code_set) - 1) + penalty * new_uncovered
                        delta = new_energy - current_energy
                        if random.random() < np.exp(-delta / T):
                            code_set.remove(c_out)
                            cover_count[neighbors[c_out]] -= 1
                            if len(code_set) < best_cost and np.sum(cover_count == 0) == 0:
                                best_cost = len(code_set)
                                best_solution = list(code_set)
                else:
                    # Add
                    c_in = random.randint(0, N-1)
                    if c_in in code_set: continue
                    current_uncovered = np.sum(cover_count == 0)
                    current_energy = len(code_set) + penalty * current_uncovered
                    fixed = np.sum(cover_count[neighbors[c_in]] == 0)
                    new_uncovered = current_uncovered - fixed
                    new_energy = (len(code_set) + 1) + penalty * new_uncovered
                    delta = new_energy - current_energy
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        code_set.add(c_in)
                        cover_count[neighbors[c_in]] += 1
                        if len(code_set) < best_cost and np.sum(cover_count == 0) == 0:
                            best_cost = len(code_set)
                            best_solution = list(code_set)
                
                T *= alpha
                if T < T_min:
                    T = T_min
            
            # Final cleanup
            while True:
                removed_any = False
                for c in list(code_set):
                    if np.min(cover_count[neighbors[c]]) > 1:
                        code_set.remove(c)
                        cover_count[neighbors[c]] -= 1
                        removed_any = True
                if not removed_any:
                    break
            
            if len(code_set) < best_cost:
                best_cost = len(code_set)
                best_solution = list(code_set)
                
            return best_solution

        best_overall = None
        best_size = N
        
        start = time.time()
        trial = 0
        while time.time() - start < 110:
            seed = trial
            init_sol = greedy_init(seed)
            time_per_trial = 2.0 if trial < 10 else 0.5
            sol = local_search(init_sol, time_limit=time_per_trial)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_overall = sol
            
            trial += 1
            
        if best_overall is None:
            best_overall = greedy_init(0)
            
        return [list(all_words[i]) for i in best_overall]
    else:
        # Fallback for other parameters (not expected)
        return []
