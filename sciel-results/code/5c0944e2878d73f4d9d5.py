import random
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin for 150s limit
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    # Helper to check validity and get metrics
    def evaluate(A):
        if not A:
            return 10000, 10000, 0
        A_sorted = sorted(A)
        diffs = set()
        duplicate = 0
        for i in range(len(A_sorted)):
            for j in range(i+1, len(A_sorted)):
                d = A_sorted[j] - A_sorted[i]
                if d in diffs:
                    duplicate += 1
                else:
                    diffs.add(d)
        
        missing = len(target_diffs - diffs)
        max_elem = A_sorted[-1]
        
        # Penalty: prioritize validity, then minimize size
        # We want missing=0 and duplicate=0
        penalty = missing * 1000 + duplicate * 1000 + max_elem
        return missing, duplicate, max_elem, penalty

    # Generate a random initial candidate
    def random_candidate(size, max_val):
        # Ensure 0 is included
        s = {0}
        while len(s) < size:
            s.add(random.randint(1, max_val))
        return list(s)

    best_solution = None
    best_penalty = float('inf')
    best_max_elem = float('inf')
    
    # Try different sizes and max values
    # Size must be at least 9 (since binom(8,2)=28 < 30)
    # Let's try sizes 9 to 12
    # Start with small max_val and increase if no solution found
    
    # We will use a simulated annealing / hill climbing approach
    # State: A (list of ints)
    
    for size in range(9, 13):
        for max_val in range(30, 200):
            if time.time() > deadline - 5:
                break
            
            # Run a few random restarts
            for restart in range(100):
                if time.time() > deadline - 5:
                    break
                
                A = random_candidate(size, max_val)
                # Local search
                for step in range(1000):
                    if time.time() > deadline - 5:
                        break
                    
                    missing, dup, max_e, pen = evaluate(A)
                    
                    if missing == 0 and dup == 0:
                        if max_e < best_max_elem:
                            best_max_elem = max_e
                            best_solution = list(A)
                            best_penalty = pen
                        # If we found a valid one, we can try to improve it further
                        # or break if we want to try other sizes
                        # But we want min max_elem, so keep searching or refine
                        # Let's just record it and continue to see if we can find smaller max_val
                        # Actually, if we are iterating max_val from low to high,
                        # the first valid one found for a given size might not be the absolute min,
                        # but close.
                        # Let's do a dedicated refinement for valid solutions.
                        break
                    
                    # Propose a move
                    # 1. Change a random element
                    idx = random.randint(0, len(A) - 1)
                    old_val = A[idx]
                    new_val = random.randint(0, max_val)
                    
                    # Ensure distinctness
                    if new_val in A:
                        continue
                    
                    A[idx] = new_val
                    new_missing, new_dup, new_max, new_pen = evaluate(A)
                    
                    # Accept if better or with probability (annealing-like)
                    # Simple hill climbing with random restarts is often enough for small N
                    if new_pen < pen:
                        pass # Accept
                    else:
                        # Revert
                        A[idx] = old_val

                # Check if we found a valid solution in this restart
                missing, dup, max_e, pen = evaluate(A)
                if missing == 0 and dup == 0:
                    if max_e < best_max_elem:
                        best_max_elem = max_e
                        best_solution = list(A)
                        best_penalty = pen

            # If we found a valid solution with max_val, we can potentially stop increasing max_val
            # for this size, but maybe a larger size allows smaller max_val?
            # Usually smaller size allows smaller diameter for Golomb rulers?
            # Optimal Golomb ruler of order 9 has length 41.
            # Order 10 has length 55?
            # So size 9 is likely best.
            
            # If we found a solution, we can break the max_val loop for this size
            # and try the next size to see if it's better?
            # But larger size usually means larger diameter.
            # So if we find a valid solution for size 9 with max_val M,
            # it's unlikely size 10 will have max_val < M.
            # So we can break size loop if we have a good candidate?
            # Let's keep searching for a bit to ensure we get a good one.
            
            if best_solution is not None:
                # We have a candidate. Let's try to refine it to reduce max_elem further.
                # Refinement: try to move elements to smaller values.
                pass

    # If no solution found, fallback to greedy (seed-like) or a known set
    if best_solution is None:
        # Fallback: Use the seed logic or a hardcoded known set
        # A known Sidon set covering 1..30:
        # Let's try to construct one quickly
        # Or just return a large valid set to be safe
        # Using the seed program's logic is safe but suboptimal.
        # Let's use a hardcoded set if we know one.
        # If not, we must return *something* valid.
        # The seed program returns a valid set.
        # I will implement a simple greedy as fallback.
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(sorted(s)) for y in sorted(s)[i + 1:]]
        for d in range(1, 31):
            if d in diffs(a):
                continue
            found = False
            for x in range(0, 1000):
                if x in a:
                    continue
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    # Check if d is covered? 
                    # The greedy in seed adds x such that x or x-d is in a?
                    # No, seed adds x to cover d.
                    # My fallback: just find any x that keeps Sidon and adds d?
                    # Actually, just finding any Sidon set covering 1..30 is hard.
                    # But the seed program works. I will copy the seed logic here as fallback.
                    pass
            
            # Re-implementing seed logic for fallback
            if not found:
                # Try adding x = d if possible
                # This is getting messy. 
                # Better: Just use a randomized search that is guaranteed to find *a* solution quickly.
                pass

    # Since the search above should find a solution (size 9, max ~41-50),
    # and the time limit is generous, it should succeed.
    
    # Final check: ensure best_solution is valid
    if best_solution:
        missing, dup, max_e, pen = evaluate(best_solution)
        if missing == 0 and dup == 0:
            return best_solution
    
    # If all else fails, return a hardcoded valid set (if I can find one)
    # Or run the seed program's logic.
    # Let's run the seed program's logic to ensure validity.
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(sorted(s)) for y in sorted(s)[i + 1:]]
    for d in range(1, 31):
        if d in diffs(a):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
