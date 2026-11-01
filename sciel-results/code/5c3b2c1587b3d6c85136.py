import itertools
import random
import numpy as np
import time
import math

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage indices
    # coverage[i] is a list of indices of words within 'radius' of all_words[i]
    coverage = []
    for i, w1 in enumerate(all_words):
        cov = []
        for j, w2 in enumerate(all_words):
            dist = sum(1 for a, b in zip(w1, w2) if a != b)
            if dist <= radius:
                cov.append(j)
        coverage.append(cov)
        
    # Initial Greedy Solution
    def greedy_construct():
        counts = [0] * total_words
        selected = [False] * total_words
        num_selected = 0
        uncovered_count = total_words
        
        # To speed up greedy, we can just pick the word that covers most uncovered.
        # Since coverage is small (13), we can iterate.
        while uncovered_count > 0:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                if selected[i]:
                    continue
                gain = 0
                for j in coverage[i]:
                    if counts[j] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break # Should not happen for radius 1
            # Select best_idx
            selected[best_idx] = True
            num_selected += 1
            for j in coverage[best_idx]:
                if counts[j] == 0:
                    uncovered_count -= 1
                counts[j] += 1
                
        return [i for i in range(total_words) if selected[i]], counts, uncovered_count

    init_cw, counts, unc = greedy_construct()
    
    # Simulated Annealing
    def simulate_annealing(initial_cw, initial_counts, initial_unc, time_limit):
        selected = [False] * total_words
        for i in initial_cw:
            selected[i] = True
            
        num_selected = len(initial_cw)
        uncovered_count = initial_unc
        current_cost = num_selected + 1000 * uncovered_count
        
        best_cost = current_cost
        best_selected = selected[:]
        
        T = 100.0
        alpha = 0.999
        min_T = 0.1
        
        start_time = time.time()
        
        # Precompute list of selected and unselected indices for faster random picking?
        # Or just pick random index and check status.
        
        while time.time() - start_time < time_limit:
            # Generate move
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove
                if num_selected == 0:
                    continue
                u = random.randint(0, total_words - 1)
                if not selected[u]:
                    continue
                # Try remove u
                # Calculate delta
                # Delta cost = -1 (num_selected) + 1000 * (change in uncovered)
                # Change in uncovered: for j in coverage[u], if counts[j] becomes 0, uncovered increases by 1
                # We need to check if removing u causes any j to become uncovered.
                # i.e., if counts[j] == 1 for any j in coverage[u].
                
                new_unc = uncovered_count
                for j in coverage[u]:
                    if counts[j] == 1:
                        new_unc += 1
                new_num = num_selected - 1
                new_cost = new_num + 1000 * new_unc
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    selected[u] = False
                    num_selected = new_num
                    uncovered_count = new_unc
                    for j in coverage[u]:
                        counts[j] -= 1
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = selected[:]
                        
            elif move_type < 0.6:
                # Add
                if num_selected == total_words:
                    continue
                v = random.randint(0, total_words - 1)
                if selected[v]:
                    continue
                # Try add v
                new_unc = uncovered_count
                for j in coverage[v]:
                    if counts[j] == 0:
                        new_unc -= 1
                new_num = num_selected + 1
                new_cost = new_num + 1000 * new_unc
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    selected[v] = True
                    num_selected = new_num
                    uncovered_count = new_unc
                    for j in coverage[v]:
                        counts[j] += 1
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = selected[:]
                        
            else:
                # Swap
                if num_selected == 0 or num_selected == total_words:
                    continue
                # Pick u from selected, v from unselected
                # Efficient way: pick random u, if not selected, retry or pick from list.
                # Given density, random pick is okay.
                u = random.randint(0, total_words - 1)
                if not selected[u]:
                    continue
                v = random.randint(0, total_words - 1)
                if selected[v]:
                    continue
                
                # Try swap u -> v
                # Effect: Remove u, Add v.
                # We need to update counts for coverage[u] (decrease) and coverage[v] (increase).
                # Overlap: words in both coverage[u] and coverage[v] have net 0 change in count?
                # No, counts[j] decreases by 1 for j in cov[u], increases by 1 for j in cov[v].
                # If j in intersection, net change 0.
                # We need to check uncovered changes carefully.
                
                # Let's just apply and revert if needed, or calculate delta.
                # Calculating delta for swap is tricky if we do it incrementally.
                # Let's do:
                # 1. Temporarily remove u
                # 2. Temporarily add v
                # But we need to know delta before accepting.
                
                # Let's compute new_unc and new_num.
                # new_num = num_selected (same)
                # new_unc:
                #   For j in cov[u] \ cov[v]: counts[j] decreases by 1. If it was 1, it becomes 0 (unc +1).
                #   For j in cov[v] \ cov[u]: counts[j] increases by 1. If it was 0, it becomes 1 (unc -1).
                #   For j in intersection: no change.
                
                # To avoid set operations, we can just iterate all j in cov[u] and cov[v].
                # cov size is ~13.
                
                # We need to know which j are in intersection.
                # Since cov is small, we can make a set of cov[v].
                set_cov_v = set(coverage[v])
                
                delta_unc = 0
                for j in coverage[u]:
                    if j not in set_cov_v:
                        if counts[j] == 1:
                            delta_unc += 1
                for j in coverage[v]:
                    if j not in set(coverage[u]): # This is inefficient, better precompute or use set for u too
                        if counts[j] == 0:
                            delta_unc -= 1
                            
                # Optimization: Precompute sets for coverage? Or just use the fact that n is small.
                # Let's just use a temporary array or logic.
                # Actually, 13 elements is tiny.
                set_cov_u = set(coverage[u])
                
                for j in coverage[v]:
                    if j not in set_cov_u:
                        if counts[j] == 0:
                            delta_unc -= 1

                new_unc = uncovered_count + delta_unc
                new_num = num_selected
                new_cost = new_num + 1000 * new_unc
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    # Apply changes
                    for j in coverage[u]:
                        counts[j] -= 1
                    for j in coverage[v]:
                        counts[j] += 1
                    selected[u] = False
                    selected[v] = True
                    uncovered_count = new_unc
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = selected[:]

            T *= alpha
            if T < min_T:
                T = min_T
                
        return [i for i in range(total_words) if best_selected[i]]

    # Run SA for a few seconds, maybe restart a few times?
    # The greedy start is usually good. One long SA run might be better than many short ones.
    # 110 seconds limit.
    cw = simulate_annealing(init_cw, counts, unc, 105)
    
    return [all_words[i] for i in cw]
