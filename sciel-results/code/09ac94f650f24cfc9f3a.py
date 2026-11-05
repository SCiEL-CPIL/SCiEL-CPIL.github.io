"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Precompute compatibility: compat[i][j] = True if lcm(i,j) > 256
    # For numbers 2..256
    n = 255  # indices 0..254 represent values 2..256
    vals = list(range(2, 257))
    
    # Build compatibility matrix
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        a = vals[i]
        for j in range(i + 1, n):
            b = vals[j]
            g = gcd(a, b)
            if a * b // g > 256:
                compat[i][j] = True
                compat[j][i] = True
    
    def is_valid_set(indices):
        """Check if set of indices is valid."""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if not compat[indices[i]][indices[j]]:
                    return False
        return True
    
    def reciprocal_sum(indices):
        return sum(1.0 / vals[i] for i in indices)
    
    # Greedy: pick smallest values (largest reciprocals) that are compatible
    greedy = []
    for i in range(n):
        if all(compat[i][j] for j in greedy):
            greedy.append(i)
    
    best = greedy[:]
    best_score = reciprocal_sum(best)
    
    # Local search: try single swaps and insertions
    def try_improve(current, score):
        improved = True
        while improved:
            improved = False
            current_set = set(current)
            # Try adding each unselected element
            for i in range(n):
                if i in current_set:
                    continue
                if all(compat[i][j] for j in current):
                    new_set = current + [i]
                    new_score = score + 1.0 / vals[i]
                    if new_score > score + 1e-15:
                        current = new_set
                        score = new_score
                        improved = True
                        break
            # Try single swap: remove one, add one
            if not improved:
                for rem in range(len(current)):
                    for add in range(n):
                        if add in current_set:
                            continue
                        # Check if add is compatible with all except current[rem]
                        if all(compat[add][current[k]] for k in range(len(current)) if k != rem):
                            new_score = score - 1.0 / vals[current[rem]] + 1.0 / vals[add]
                            if new_score > score + 1e-15:
                                new_current = [x for k, x in enumerate(current) if k != rem] + [add]
                                current = new_current
                                score = new_score
                                current_set = set(current)
                                improved = True
                                break
                    if improved:
                        break
        return current, score
    
    best, best_score = try_improve(best, best_score)
    
    # Try multiple random restarts with different greedy orders
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Perturb: start with a subset of greedy, then greedily fill
            if restart == 0:
                # Try greedy from different starting points
                for start_i in range(0, n, 10):
                    if time.time() > deadline - 5:
                        break
                    trial = [start_i]
                    for i in range(n):
                        if i == start_i:
                            continue
                        if all(compat[i][j] for j in trial):
                            trial.append(i)
                    trial, trial_score = try_improve(trial, reciprocal_sum(trial))
                    if trial_score > best_score:
                        best = trial
                        best_score = trial_score
            else:
                # Random perturbation of best
                if len(best) > 2:
                    trial = best[:]
                    random.shuffle(trial)
                    # Remove a random subset
                    remove_count = random.randint(1, min(3, len(trial) // 2))
                    to_remove = random.sample(trial, remove_count)
                    trial = [x for x in trial if x not in set(to_remove)]
                    trial, trial_score = try_improve(trial, reciprocal_sum(trial))
                    if trial_score > best_score:
                        best = trial
                        best_score = trial_score
    
    # Convert back to values
    return [vals[i] for i in best]

# EVOLVE-BLOCK-END
