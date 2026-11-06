"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def check_incremental(a, selected):
        """
        Check if adding a to selected maintains the constraint.
        Only need to check:
        1. Does a divide any sum of 3 distinct others in selected?
        2. For each x in selected, does x divide any sum of 3 distinct from (selected - {x}) + {a}?
           This means x divides a sum that includes a, i.e., x | (a + b + c) where b,c in selected, b!=c, b!=a, c!=a
        """
        sel = selected
        n = len(sel)
        
        # Check 1: a divides sum of any 3 distinct others in selected
        for combo in combinations(sel, 3):
            if sum(combo) % a == 0:
                return False
        
        # Check 2: For each x in selected, check if x divides (a + b + c) for some b,c in selected, b!=c
        # We need b,c distinct from each other and from x, and from a
        for x in sel:
            if x == a:
                continue
            # Find all pairs (b,c) from sel where b != c, b != x, c != x, b != a, c != a
            # Then check if (a + b + c) % x == 0
            for i in range(n):
                b = sel[i]
                if b == x or b == a:
                    continue
                for j in range(i+1, n):
                    c = sel[j]
                    if c == x or c == a:
                        continue
                    if (a + b + c) % x == 0:
                        return False
        return True
    
    def full_check(candidate_set):
        """Full validation of a set."""
        n = len(candidate_set)
        for i, x in enumerate(candidate_set):
            others = [candidate_set[j] for j in range(n) if j != i]
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
                    return False
        return True
    
    # Phase 1: Greedy from large to small with fast incremental check
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if check_incremental(x, selected):
            selected.append(x)
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # Phase 2: Try to add more elements (some may have been skipped due to order)
    if time.time() < deadline:
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if check_incremental(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    # Phase 3: Local search - try swaps to increase size
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_selected)):
                if time.time() > deadline:
                    break
                for replacement in range(1, 193):
                    if time.time() > deadline:
                        break
                    if replacement in best_selected:
                        continue
                    # Try swapping best_selected[i] with replacement
                    new_selected = best_selected[:i] + best_selected[i+1:] + [replacement]
                    new_selected.sort(reverse=True)
                    if full_check(new_selected):
                        # Check if we can now add more
                        best_selected = new_selected
                        best_size = len(best_selected)
                        improved = True
                        # Try to add more
                        for x in range(192, 0, -1):
                            if time.time() > deadline:
                                break
                            if x not in best_selected:
                                if check_incremental(x, best_selected):
                                    best_selected.append(x)
                                    best_size = len(best_selected)
                        break  # Restart the outer loop
    
    # Phase 4: Final attempt to add any remaining elements
    if time.time() < deadline:
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if check_incremental(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
