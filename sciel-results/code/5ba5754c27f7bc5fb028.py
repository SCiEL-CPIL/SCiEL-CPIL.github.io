import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    n = 256
    
    # Perfect squares up to 2*256 = 512
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Step 1: Remove numbers where 2*a is a perfect square (diagonal constraint)
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
    
    # Build conflict sets for fast lookup
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    def greedy_select(order):
        """Select from order, skipping conflicts with already selected."""
        selected = []
        selected_set = set()
        for a in order:
            if a in selected_set:
                continue
            # Check if a conflicts with any already selected
            if not any(b in conflict[a] for b in selected):
                selected.append(a)
                selected_set.add(a)
        return selected
    
    # Try multiple greedy orders
    results = []
    
    # Natural order
    results.append(greedy_select(candidates))
    
    # Reverse order
    results.append(greedy_select(list(reversed(candidates))))
    
    # By conflict count (ascending) - pick least constrained first
    by_conflict_asc = sorted(candidates, key=lambda x: len(conflict[x]))
    results.append(greedy_select(by_conflict_asc))
    
    # By conflict count (descending) - pick most constrained first  
    by_conflict_desc = sorted(candidates, key=lambda x: -len(conflict[x]))
    results.append(greedy_select(by_conflict_desc))
    
    # Pick the best
    best = max(results, key=len)
    best_set = set(best)
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return best
    
    # Local improvement: try to add any candidate that doesn't conflict
    improved = list(best)
    improved_set = set(improved)
    
    for a in candidates:
        if a not in improved_set:
            if not any(b in conflict[a] for b in improved):
                improved.append(a)
                improved_set.add(a)
    
    if time.time() - start_time > deadline - 5:
        return improved
    
    # Try single-element swaps: remove one element, see if we can add more
    for i in range(len(improved)):
        rem = improved[i]
        remaining = [x for x in improved if x != rem]
        remaining_set = set(remaining)
        
        # Try to add candidates that only conflicted with rem
        new_added = []
        for a in candidates:
            if a not in remaining_set and a != rem:
                # Check if a conflicts with any in remaining
                if not any(b in conflict[a] for b in remaining):
                    new_added.append(a)
        
        if len(remaining) + len(new_added) > len(improved):
            improved = remaining + new_added
            improved_set = set(improved)
    
    return improved
# EVOLVE-BLOCK-END
