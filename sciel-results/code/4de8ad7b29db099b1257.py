"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # For this specific task: n_rows=4, k=7
    # We know the best is 134. We want to find a set with max entry as small as possible.
    
    # The greedy algorithm in the seed produces a specific result.
    # Let's first run the greedy to get a baseline 'best'.
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # We will try to improve upon this using randomized local search / backtracking with time limit.
    # Strategy:
    # 1. Try to construct rows one by one, but with randomization in tie-breaking or choice.
    # 2. Or, use a backtracking search that tries to keep the max value low.
    
    # A more effective approach for "Difference Triangle Sets" is often a backtracking search
    # that tries to fill the rows, prioritizing the smallest possible next number,
    # but allowing backtracking if it leads to a dead end (though for 4x8, greedy is often close).
    
    # However, the prompt mentions "MRV-based global selection" improved score.
    # Let's implement a smarter heuristic:
    # Instead of building row by row strictly, we can use a priority queue of "next possible moves"
    # across all rows, always picking the move that adds the smallest new number.
    # This is similar to the "greedy" but global.
    
    # Let's try a randomized restart approach with a global greedy strategy.
    
    def global_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        used = set()
        rows = [[] for _ in range(n_rows)]
        # Each row starts with 0
        for i in range(n_rows):
            rows[i].append(0)
        
        # We need to add k more numbers to each row.
        # Total numbers to add: n_rows * k
        # We maintain a priority queue of (next_candidate_value, row_index)
        # But the next candidate value depends on the current state of the row and used set.
        
        # Simpler: At each step, for each row that is not full, find the smallest x > last[row]
        # such that all differences (x - a for a in row) are not in used.
        # Pick the row that has the smallest such x. If tie, break randomly or by index.
        
        while True:
            min_x = float('inf')
            best_row = -1
            best_candidate = None
            
            candidates = []
            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                x = last + 1
                while True:
                    # Check if x is valid for row i
                    valid = True
                    diffs = []
                    for a in rows[i]:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        diffs.append(d)
                    if valid:
                        break
                    x += 1
                    # Safety: if x gets too large, we might be stuck, but for this problem size it's fine
                    if x > 1000:
                        break
                
                if x < min_x:
                    min_x = x
                    best_row = i
                    best_candidate = (x, diffs)
            
            if best_row == -1:
                break # All rows full
            
            # Apply the move
            rows[best_row].append(min_x)
            for d in best_candidate[1]:
                used.add(d)
            
            # Check time
            if time.time() > deadline:
                break
        
        # Verify all rows are full
        if all(len(r) == k + 1 for r in rows):
            return rows
        return None

    # Run global greedy multiple times with different random seeds (though the logic above is deterministic,
    # we can add randomness to tie-breaking or by permuting row order).
    # Actually, the global greedy as defined is deterministic.
    # To make it stochastic, we can randomize the order in which we check rows when min_x is tied,
    # or we can randomize the initial "bias" of which row to extend if multiple have the same min_x.
    
    # Let's modify global_greedy to accept a random tie-breaker.
    def stochastic_global_greedy(random_seed):
        random.seed(random_seed)
        used = set()
        rows = [[] for _ in range(n_rows)]
        for i in range(n_rows):
            rows[i].append(0)
        
        while True:
            min_x = float('inf')
            best_row = -1
            best_candidate = None
            tied_rows = []
            
            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                x = last + 1
                diffs = None
                while True:
                    valid = True
                    diffs = []
                    for a in rows[i]:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        diffs.append(d)
                    if valid:
                        break
                    x += 1
                    if x > 1000:
                        break
                
                if x < min_x:
                    min_x = x
                    best_row = i
                    best_candidate = (x, diffs)
                    tied_rows = [i]
                elif x == min_x:
                    tied_rows.append(i)
            
            if best_row == -1:
                break
            
            # Randomly pick from tied_rows if there's a tie
            if len(tied_rows) > 1:
                chosen_row = random.choice(tied_rows)
                # Recalculate candidate for chosen_row to be safe (or store it)
                # For efficiency, we stored best_candidate for the first min_x found.
                # If we pick a different tied row, we need its diffs.
                # Let's just re-find the candidate for the chosen row.
                last = rows[chosen_row][-1]
                x = last + 1
                while True:
                    valid = True
                    diffs = []
                    for a in rows[chosen_row]:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        diffs.append(d)
                    if valid:
                        break
                    x += 1
                rows[chosen_row].append(x)
                for d in diffs:
                    used.add(d)
            else:
                rows[best_row].append(min_x)
                for d in best_candidate[1]:
                    used.add(d)
            
            if time.time() > deadline:
                break
        
        if all(len(r) == k + 1 for r in rows):
            return rows
        return None

    # Try a few stochastic runs
    for seed in range(50):
        if time.time() > deadline:
            break
        rows = stochastic_global_greedy(seed)
        if rows:
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows
                # If we hit the theoretical bound or known best, we can stop early?
                # Known best is 134. If we find 134, we can't do better unless we find <134.
                # The counting bound is 112, but 134 is the best known.
                # If we find 134, we might still try to find <134, but it's unlikely.
                # Let's continue searching for a while.
    
    return best_rows

def _greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
