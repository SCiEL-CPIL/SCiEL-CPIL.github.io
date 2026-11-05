import time
import random
import math

_START = None

def _is_sidon_and_complete(marks):
    """Check if marks is a Sidon set containing 0 and covers differences 1..30."""
    if 0 not in marks:
        return False
    marks = sorted(marks)
    diffs = set()
    n = len(marks)
    for i in range(n):
        for j in range(i + 1, n):
            d = marks[j] - marks[i]
            if d in diffs:
                return False
            diffs.add(d)
            if d > 30:
                break
    # Check if 1..30 are covered
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _get_missing_diffs(marks):
    """Return the set of missing differences from 1..30."""
    diffs = set()
    n = len(marks)
    for i in range(n):
        for j in range(i + 1, n):
            d = marks[j] - marks[i]
            if d <= 30:
                diffs.add(d)
    return set(range(1, 31)) - diffs

def _local_search(initial_marks, max_iters=5000):
    """
    Perform local search to minimize the max element.
    Moves: 
    1. Remove a non-zero element.
    2. Add a new element.
    3. Replace an element with another.
    Objective: Minimize max(marks), subject to Sidon and Complete(1..30).
    We use a penalty method: Cost = max(marks) * 1000 + (30 - covered_count) * 10 + (sidon_violations * 100).
    Actually, simpler: Just try to reduce max. If we can't reduce max, try to improve coverage.
    Let's stick to a simple hill-climbing on the max value.
    """
    current = sorted(initial_marks)
    if not _is_sidon_and_complete(current):
        return None
        
    best = list(current)
    best_val = max(best)
    
    # We want to find a set with smaller max.
    # Strategy: Randomly remove elements to make room, then try to add smaller ones?
    # Or just random restarts of the set composition.
    
    # Let's use a simpler randomized local search:
    # State: a set of integers.
    # Validity: Sidon + 0 in set + covers 1..30.
    # Goal: Minimize max(set).
    
    # Since the space is small, we can just do random restarts of the whole set construction.
    pass

def _construct_random_candidate():
    """
    Construct a random Sidon set that covers 1..30.
    This is tricky. Instead, let's use a known heuristic or just random search over small sets.
    """
    # A known small perfect difference set (truncated) might be found by random search.
    # Let's try to build one randomly.
    # Start with 0.
    marks = [0]
    # We need to cover 1..30.
    # Randomly pick pairs (a, b) such that b-a is in 1..30.
    # This is essentially a graph problem.
    
    # Simpler: Random restarts of a greedy-like process with randomization.
    # Or just brute force search for small sets?
    # Size of set: n(n-1)/2 >= 30 => n>=9.
    # Let's try n=10, 11, 12...
    
    # Actually, the most robust "new approach" for this specific small-scale optimization is
    # **Simulated Annealing** on the set of elements.
    
    return None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Use a randomized local search / simulated annealing approach.
    # State: A set of integers S.
    # Energy: E = max(S) * 100 + (30 - |diffs(S) \cap {1..30}|) * 10 + (number of duplicate diffs) * 1000
    # We want to minimize E.
    
    best_solution = None
    best_energy = float('inf')
    
    # Random restarts
    for restart in range(500):
        if time.time() - _START > 100:
            break
            
        # Initialize a random set
        # Let's pick a random max bound D, say 150
        D = 150
        # Start with 0 and some random numbers
        # Ensure we have enough elements to cover 30 diffs.
        # Min size ~9. Let's start with 10.
        
        # Randomly select 9 more numbers from 1..D
        # But they must form a Sidon set? No, SA will fix it.
        # Start with a random set that might be invalid.
        
        size = 10
        if random.random() < 0.5:
            size = 11
        if random.random() < 0.1:
            size = 12
            
        marks = {0}
        # Add random numbers
        while len(marks) < size:
            marks.add(random.randint(1, D))
            
        current = list(marks)
        
        # Simulated Annealing
        T = 1000.0
        cooling = 0.999
        iters = 2000
        
        for i in range(iters):
            if time.time() - _START > 105:
                break
                
            # Calculate current energy
            curr_energy = _calc_energy(current)
            
            # Generate neighbor
            # Move 1: Change one element
            # Move 2: Add an element
            # Move 3: Remove an element
            
            move_type = random.random()
            new_marks = list(current)
            
            if move_type < 0.7:
                # Change one element
                idx = random.randint(0, len(new_marks) - 1)
                if new_marks[idx] == 0:
                    # Don't change 0
                    idx = random.randint(1, len(new_marks) - 1) if len(new_marks) > 1 else 0
                old_val = new_marks[idx]
                new_val = random.randint(1, D)
                if new_val in new_marks:
                    continue
                new_marks[idx] = new_val
            elif move_type < 0.85:
                # Add an element
                if len(new_marks) < 15:
                    new_val = random.randint(1, D)
                    if new_val not in new_marks:
                        new_marks.append(new_val)
                    else:
                        continue
            else:
                # Remove an element
                if len(new_marks) > 9:
                    idx = random.randint(1, len(new_marks) - 1) # Don't remove 0
                    new_marks.pop(idx)
            
            new_energy = _calc_energy(new_marks)
            
            # Acceptance probability
            if new_energy < curr_energy:
                current = new_marks
            else:
                delta = new_energy - curr_energy
                if T > 0 and random.random() < math.exp(-delta / T):
                    current = new_marks
            
            T *= cooling
            
            # Check if we found a valid solution with better max
            if _is_sidon_and_complete(current):
                max_val = max(current)
                if max_val < best_energy:
                    best_energy = max_val
                    best_solution = list(current)
        
        # After SA, try to polish
        if _is_sidon_and_complete(current):
             max_val = max(current)
             if max_val < best_energy:
                 best_energy = max_val
                 best_solution = list(current)

    if best_solution is not None:
        return best_solution
    
    # Fallback to the original greedy if SA fails (should not happen)
    return _seed_greedy()

def _calc_energy(marks):
    """
    Calculate energy for the Simulated Annealing.
    Lower is better.
    """
    if 0 not in marks:
        return 100000
    marks = sorted(marks)
    diffs = []
    n = len(marks)
    for i in range(n):
        for j in range(i + 1, n):
            d = marks[j] - marks[i]
            diffs.append(d)
            
    # Penalty for duplicate differences
    diff_set = set(diffs)
    dup_penalty = (len(diffs) - len(diff_set)) * 100
    
    # Penalty for missing differences 1..30
    covered = len(diff_set.intersection(set(range(1, 31))))
    missing_penalty = (30 - covered) * 10
    
    # Main objective: Max element
    max_val = max(marks)
    
    # Penalty for size (prefer smaller sets if possible, though not strictly required, but helps)
    size_penalty = max(0, len(marks) - 10) * 5
    
    return max_val * 100 + dup_penalty + missing_penalty + size_penalty

def _seed_greedy():
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
