import random
import numpy as np
import math

def construct(q, n, radius):
    total_words = q ** n
    all_words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
    
    # Precompute coverage: list of lists of indices
    # coverage[i] contains all j such that dist(all_words[i], all_words[j]) <= radius
    coverage = []
    for i, w in enumerate(all_words):
        covered = [i]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    nw = list(w)
                    nw[pos] = val
                    nw = tuple(nw)
                    # Find index of nw
                    # Since total_words is small (2187), we can use a dictionary
                    # But building a dict every time is slow. 
                    # Let's build the dict once.
                    pass
        
    # Build word_to_idx map for fast lookup during precomputation
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for i, w in enumerate(all_words):
        covered = [i]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    nw = list(w)
                    nw[pos] = val
                    nw = tuple(nw)
                    j = word_to_idx[nw]
                    covered.append(j)
        coverage.append(covered)
        
    # Convert to numpy arrays for faster indexing? 
    # coverage is a list of lists. We can keep it as is, or use sparse matrices.
    # For N=2187, list of lists is fine.

    # 1. Randomized Greedy Initialization
    def greedy_init():
        selected = np.zeros(total_words, dtype=bool)
        cover_count = np.zeros(total_words, dtype=int)
        uncovered = np.ones(total_words, dtype=bool)
        
        while uncovered.any():
            best_c = -1
            best_count = -1
            # To add randomness, we can pick from top K candidates or add noise
            # For simplicity, standard greedy first, then we can randomize tie-breaking
            for c in range(total_words):
                if selected[c]:
                    continue
                # Count uncovered words covered by c
                cnt = np.sum(uncovered[coverage[c]])
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                elif cnt == best_count and cnt > 0:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_c = c
            
            if best_c == -1:
                break # Should not happen
                
            selected[best_c] = True
            # Update cover_count and uncovered
            for j in coverage[best_c]:
                cover_count[j] += 1
                if cover_count[j] == 1:
                    uncovered[j] = False
                    
        return selected, cover_count, int(np.sum(selected)), int(np.sum(cover_count == 0))

    # 2. Simulated Annealing
    def simulated_annealing(initial_selected, initial_cover_count, time_limit=100):
        import time
        start_time = time.time()
        
        selected = initial_selected.copy()
        cover_count = initial_cover_count.copy()
        
        current_K = int(np.sum(selected))
        current_U = int(np.sum(cover_count == 0))
        current_cost = current_K * 10000 + current_U
        
        best_selected = selected.copy()
        best_K = current_K
        best_U = current_U
        best_cost = current_cost
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random codeword to flip
            i = random.randint(0, total_words - 1)
            
            # Calculate Delta
            delta_K = 0
            delta_U = 0
            
            if selected[i]:
                # Turning OFF
                delta_K = -1
                # Words that might become uncovered are those where cover_count[j] == 1
                # and j is in coverage[i]
                for j in coverage[i]:
                    if cover_count[j] == 1:
                        delta_U += 1
            else:
                # Turning ON
                delta_K = 1
                # Words that might become covered are those where cover_count[j] == 0
                # and j is in coverage[i]
                for j in coverage[i]:
                    if cover_count[j] == 0:
                        delta_U -= 1
                        
            delta_cost = delta_K * 10000 + delta_U
            
            if delta_cost < 0:
                accept = True
            else:
                # Boltzmann distribution
                # Avoid overflow
                if delta_cost / T < 700: 
                    prob = math.exp(-delta_cost / T)
                else:
                    prob = 0.0
                accept = random.random() < prob
            
            if accept:
                # Update state
                selected[i] = not selected[i]
                for j in coverage[i]:
                    if selected[i]:
                        cover_count[j] += 1
                    else:
                        cover_count[j] -= 1
                        
                current_K += delta_K
                current_U += delta_U
                current_cost += delta_cost
                
                if current_U == 0 and current_K < best_K:
                    best_K = current_K
                    best_U = current_U
                    best_cost = current_cost
                    best_selected = selected.copy()
            
            T *= cooling_rate
            
        return best_selected, best_K

    # Run Greedy Init
    selected, cover_count, K, U = greedy_init()
    
    # Run SA
    best_selected, best_K = simulated_annealing(selected, cover_count, time_limit=110)
    
    return [list(all_words[i]) for i in range(total_words) if best_selected[i]]
