"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    N = 284
    MAX_SUM = 2 * N  # 568

    # Initialize the set with prescribed elements
    current_set = sorted(prescribed)
    used_sums = set()
    
    # Precompute sums for prescribed elements
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            used_sums.add(current_set[i] + current_set[j])
    
    best_size = len(current_set)
    best_set = current_set[:]

    # Candidates are elements from 12 to 284
    candidates = list(range(12, N + 1))

    # Branch and Bound
    def branch_and_bound(idx, current_set, used_sums, current_size):
        nonlocal best_size, best_set, start_time, DEADLINE

        if time.time() > DEADLINE:
            return

        # Update best if current set is larger
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]

        # If no more candidates, return
        if idx >= len(candidates):
            return

        # Pruning: estimate max possible additional elements
        # Remaining candidates: candidates[idx:]
        # Max additional elements is roughly sqrt(len(candidates) - idx)
        # This is a very loose bound, but helps prune deep branches
        remaining = len(candidates) - idx
        if current_size + remaining <= best_size:
            return

        # Option 1: Skip the current candidate
        branch_and_bound(idx + 1, current_set, used_sums, current_size)

        # Option 2: Include the current candidate if it doesn't conflict
        x = candidates[idx]
        conflict = False
        for y in current_set:
            if (x + y) in used_sums:
                conflict = True
                break
        if (2 * x) in used_sums:
            conflict = True
        
        if not conflict:
            # Add x to the set
            new_set = current_set + [x]
            new_sums = used_sums.copy()
            for y in current_set:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            
            branch_and_bound(idx + 1, new_set, new_sums, current_size + 1)

    # Start the search
    branch_and_bound(0, current_set, used_sums, len(current_set))

    # Ensure prescribed elements are present (they are by construction)
    return sorted(best_set)

# EVOLVE-BLOCK-END
