import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_sets = len(all_sets)
    
    # Precompute intersections as frozensets for fast comparison
    set_list = [set(s) for s in all_sets]
    
    def is_sunflower_triple(i, j, k):
        """Check if sets i, j, k form a sunflower (all three pairwise intersections equal)."""
        ij = set_list[i] & set_list[j]
        ik = set_list[i] & set_list[k]
        jk = set_list[j] & set_list[k]
        return ij == ik == jk
    
    def check_conflict_with_existing(new_idx, chosen_indices):
        """Check if adding new_idx creates a sunflower with any pair in chosen_indices."""
        for a in range(len(chosen_indices)):
            for b in range(a + 1, len(chosen_indices)):
                if is_sunflower_triple(new_idx, chosen_indices[a], chosen_indices[b]):
                    return True
        return False
    
    def greedy_build(order):
        """Greedy build using given order of set indices."""
        chosen = []
        for idx in order:
            if not check_conflict_with_existing(idx, chosen):
                chosen.append(idx)
        return chosen
    
    def local_search(chosen, max_iter=1000):
        """Try to improve by swapping out one set for another."""
        chosen_set = set(chosen)
        current_size = len(chosen)
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline:
            improved = False
            iterations += 1
            # Try removing one and adding another
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                trial = chosen[:i] + chosen[i+1:]
                trial_set = set(trial)
                # Try to add any set not in trial_set
                for j in range(n_sets):
                    if j in trial_set:
                        continue
                    if not check_conflict_with_existing(j, trial):
                        new_chosen = trial + [j]
                        if len(new_chosen) > current_size:
                            chosen = new_chosen
                            current_size = len(chosen)
                            chosen_set = set(chosen)
                            improved = True
                            break
                if improved:
                    break
        return chosen
    
    # Multiple random restarts with greedy + local search
    best_chosen = []
    best_size = 0
    
    # First, try a deterministic greedy to get a baseline
    base_order = list(range(n_sets))
    base_chosen = greedy_build(base_order)
    best_chosen = base_chosen[:]
    best_size = len(base_chosen)
    
    # Random restarts
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        order = list(range(n_sets))
        random.shuffle(order)
        chosen = greedy_build(order)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
        
        # Local search
        if time.time() < deadline:
            chosen_ls = local_search(chosen, max_iter=200)
            if len(chosen_ls) > best_size:
                best_size = len(chosen_ls)
                best_chosen = chosen_ls[:]
    
    # Also try local search on the best found so far
    if time.time() < deadline:
        best_ls = local_search(best_chosen, max_iter=500)
        if len(best_ls) > best_size:
            best_size = len(best_ls)
            best_chosen = best_ls[:]
    
    return [sorted(set_list[i]) for i in best_chosen]
