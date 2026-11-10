"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples
    # A triple (a, b, c) with a < b < c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    conflict = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    # Bad triple (a, b, c)
                    conflict[c].append((a, b))
                    conflict[b].append((a, c))
                    conflict[a].append((b, c))
                    
    if time.time() > deadline - 10:
        # Fallback to simple greedy if precomputation is too slow
        selected = []
        for x in range(1, 129):
            valid = True
            for u, v in conflict[x]:
                if u in selected and v in selected:
                    valid = False
                    break
            if valid:
                selected.append(x)
        return selected

    # Convert conflict lists to sets of tuples for faster lookup? 
    # Actually, checking "u in S and v in S" is O(1) if S is a set.
    # But iterating over conflict[x] is O(k).
    
    # Initial solution: Greedy based on conflict count (fewest conflicts first)
    conflict_count = {i: len(conflict[i]) for i in range(1, n + 1)}
    order = sorted(numbers, key=lambda x: conflict_count[x])
    
    current_set = set()
    for x in order:
        valid = True
        for u, v in conflict[x]:
            if u in current_set and v in current_set:
                valid = False
                break
        if valid:
            current_set.add(x)
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Simulated Annealing
    # State: A set of numbers.
    # Objective: Maximize size.
    # Moves: 
    #   1. Add a random element not in set (if valid)
    #   2. Remove a random element from set
    #   3. Swap: Remove one, Add one (if valid)
    
    # To make SA effective, we need a way to evaluate "validity" quickly.
    # We track the set.
    
    # Precompute for each x, the list of conflicting pairs is already done.
    
    # Let's use a slightly more structured SA:
    # Temperature starts high, decays.
    # At each step, generate a neighbor.
    # If neighbor is valid and larger, accept.
    # If neighbor is valid and same size, accept with prob exp(0/T) = 1.
    # If neighbor is invalid, we can't accept it directly. 
    # Wait, the state space is defined by VALID sets only? 
    # If we only consider valid sets, we are doing a random walk on the graph of valid sets.
    # Moves must result in a valid set.
    
    # Move generation for valid set S:
    # - Add x: Check if S U {x} is valid. If yes, it's a valid neighbor of size |S|+1.
    # - Remove x: S \ {x} is always valid (subset of valid set). Size |S|-1.
    # - Swap x (in S) with y (not in S): Check if (S \ {x}) U {y} is valid.
    
    # Since we want to maximize size, "Add" moves are strictly better. 
    # If we can always find an add move, we should take it. 
    # The current greedy already does this until no more add moves are possible (local optimum for additions).
    # So the search must involve swaps to escape local optima where no single addition is possible, 
    # but a swap (remove one blocker, add two others? No, swap is 1-for-1) or 
    # remove one to enable two additions (net +1) might be possible.
    
    # Let's implement a "Remove 1, Try Add All" move, which is effectively a 2-step improvement.
    # Or standard SA on the space of valid sets with moves:
    # 1. Try to add a random element. If valid, accept (size +1).
    # 2. Try to remove a random element. If size > 0, accept with some probability? No, we want to maximize size.
    #    Removing an element decreases size. We only accept if it leads to a state where we can add MORE elements later.
    #    This is hard to evaluate locally.
    
    # Alternative: Hill Climbing with 2-swap (remove 2, add 2) or (remove 1, add 1).
    # The current code does 1-swap. 
    # Let's try a more aggressive local search: 
    # For each element x in S, remove it. Then greedily add all possible elements. 
    # If the resulting set is larger than S, accept.
    # This explores the "basin" around each single-removal.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each element and then greedily filling
        for x in list(current_set):
            if time.time() > deadline - 5:
                break
            trial_set = set(current_set)
            trial_set.remove(x)
            
            # Greedily add any valid element
            # Order: sorted by conflict count (least constrained first)
            for y in order:
                if y in trial_set:
                    continue
                valid = True
                for u, v in conflict[y]:
                    if u in trial_set and v in trial_set:
                        valid = False
                        break
                if valid:
                    trial_set.add(y)
            
            if len(trial_set) > len(current_set):
                current_set = trial_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                improved = True
            # If no improvement, we could try random restarts from this state?
            # Or just continue.
            
    # Additional phase: Randomized Restart Local Search
    # If stuck, perturb the set and re-optimize.
    if time.time() < deadline - 10:
        for _ in range(50): # Limited restarts
            if time.time() > deadline - 5:
                break
            
            # Perturb: Remove k random elements
            k = random.randint(1, 5)
            trial_set = set(current_set)
            elements_to_remove = random.sample(list(trial_set), min(k, len(trial_set)))
            for x in elements_to_remove:
                trial_set.remove(x)
            
            # Re-optimize: Greedily add
            for y in order:
                if y in trial_set:
                    continue
                valid = True
                for u, v in conflict[y]:
                    if u in trial_set and v in trial_set:
                        valid = False
                        break
                if valid:
                    trial_set.add(y)
            
            if len(trial_set) > len(current_set):
                current_set = trial_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
    
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
