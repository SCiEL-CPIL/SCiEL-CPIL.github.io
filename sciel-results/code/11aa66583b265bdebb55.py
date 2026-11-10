"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    N = 128
    # Precompute gcd table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(x, selected_set, selected_list):
        """Check if x can be added to selected without creating a bad triple."""
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            gx_a = gcd_table[x][a]
            for j in range(i+1, n):
                b = selected_list[j]
                if gx_a == gcd_table[x][b] == gcd_table[a][b]:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy: add elements in given order if possible."""
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    def local_improve(selected):
        """Try to improve by removing one element and adding others."""
        selected_set = set(selected)
        best = list(selected)
        best_set = set(selected)
        n_best = len(selected)
        
        for rem_idx in range(len(selected)):
            rem = selected[rem_idx]
            # Try removing rem and greedily adding others
            trial = [x for x in selected if x != rem]
            trial_set = selected_set - {rem}
            # Try to add all other elements
            for x in range(1, N+1):
                if x not in trial_set:
                    if can_add(x, trial_set, trial):
                        trial.append(x)
                        trial_set.add(x)
            if len(trial) > n_best:
                best = trial
                best_set = trial_set
                n_best = len(trial)
        
        return best
    
    # Initial deterministic greedy
    best = greedy_construct(range(1, N+1))
    best_size = len(best)
    
    # Randomized greedy passes
    rng = random.Random(42)
    passes = 0
    while time.time() < deadline - 5.0:
        order = list(range(1, N+1))
        rng.shuffle(order)
        cand = greedy_construct(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
        passes += 1
        
        # Periodically do local improvement
        if passes % 10 == 0 and len(best) > 0:
            improved = local_improve(best)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
    
    # Final local improvement if time permits
    if time.time() < deadline - 1.0 and best:
        improved = local_improve(best)
        if len(improved) > best_size:
            best = improved
    
    return best if best else [1]
# EVOLVE-BLOCK-END
