"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_valid_candidate(a, selected):
        """Check if adding a to selected maintains the constraint."""
        # For each element in selected + [a], check 3-combinations of others
        candidate_set = selected + [a]
        n = len(candidate_set)
        
        for i, x in enumerate(candidate_set):
            # Get all other elements
            others = [candidate_set[j] for j in range(n) if j != i]
            # Check all 3-combinations of others
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
                    return False
        return True
    
    def count_violations(a, selected):
        """Count how many violations would be caused by adding a."""
        candidate_set = selected + [a]
        n = len(candidate_set)
        violations = 0
        
        for i, x in enumerate(candidate_set):
            others = [candidate_set[j] for j in range(n) if j != i]
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
                    violations += 1
        return violations
    
    # Phase 1: Greedy from large to small
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_candidate(x, selected):
            selected.append(x)
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # Phase 2: Local search - try to swap elements
    if time.time() < deadline:
        # Try removing one element and adding a different one
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            # Try removing selected[i] and adding a smaller number
            for replacement in range(1, selected[i]):
                if time.time() > deadline:
                    break
                new_selected = selected[:i] + selected[i+1:] + [replacement]
                if len(new_selected) != len(set(new_selected)):
                    continue
                if is_valid_candidate(replacement, selected[:i] + selected[i+1:]):
                    if len(new_selected) > best_size:
                        best_size = len(new_selected)
                        best_selected = list(new_selected)
    
    # Phase 3: Try adding more elements if possible
    if time.time() < deadline:
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if is_valid_candidate(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
