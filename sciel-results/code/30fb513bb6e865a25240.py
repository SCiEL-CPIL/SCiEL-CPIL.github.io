"""Construct large sets without subset averages using a conflict-graph / independent-set view.

Strategy:
- The dominant small-subset violations are 3-term arithmetic progressions (size-2 subsets whose mean is a third element).
- Build a conflict graph on [1..512] where an edge between a and b exists if the midpoint (a+b)/2 is an integer and lies in [1..512] (i.e., a+b is even) — that midpoint would be the mean of {a,b}. More generally, any element m that is the mean of a subset T of size >=2 creates conflicts among the elements of T and m.
- For size-2 subsets, the condition is: no three distinct elements a, m, b with a + b = 2m. So m is "bad" if there exist a,b in the set with a+b=2m.
- We can model this as: for each candidate element m, it is "forbidden" given a set S if there exist two distinct a,b in S\{m} with a+b=2m.
- A structural approach: greedily build a set while maintaining, for each element m not yet in the set, a count of how many pairs (a,b) in the current set sum to 2m. Adding x increases these counts for all m where x pairs with an existing element to form 2m.
- We want to avoid adding any x that would create a new violation for x itself (i.e., some pair in current set sums to 2x) or for any existing element (i.e., some pair involving x and an existing element sums to 2*existing).
- To maximize cardinality up to 48, we use a greedy algorithm with a priority: at each step, among all candidates 1..512 not yet selected, pick the one that creates the fewest future conflicts (i.e., minimizes the number of new "forbidden" midpoints).
- Additionally, we use a local search / repair phase: if stuck below 48, try swapping elements to free up space.

However, the most effective structural change suggested is to model the conflict graph and use a maximum independent set heuristic. Let's implement:
1. Precompute for each pair (a,b) with a<b, the midpoint m=(a+b)/2 if integer.
2. The conflict graph has vertices 1..512. An edge between a and b exists if there is some m such that a+b=2m (always true if a+b even). But the real constraint is on triples.
3. Better: think of it as a 3-uniform hypergraph where a hyperedge is {a,m,b} with a+b=2m. We want a set with no such hyperedge fully contained.
4. For greedy construction: maintain for each potential element m, a "danger" count = number of pairs (a,b) in current set with a+b=2m. If we add x, new dangers are created for m where x + y = 2m for some y in current set, i.e., m = (x+y)/2.
5. Pick x that minimizes the maximum danger created, or that doesn't create any danger for x itself.

Let's implement a fast greedy with conflict counting, then a local search to try to reach 48.
"""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 512
    MAX_SIZE = 48
    
    # Greedy construction with conflict tracking
    # For each element m, track how many pairs (a,b) in selected set have a+b = 2m
    # When adding x, for each y in selected, if (x+y) is even, m=(x+y)//2 gets +1 danger
    # x itself is valid only if danger[x] == 0 (no pair in selected sums to 2x)
    
    selected = []
    selected_set = set()
    # danger[m] = number of pairs in selected that sum to 2m
    danger = [0] * (N + 1)
    
    # Precompute for each x, the list of m values that would be affected when x is added
    # When x is added, for each y in selected, m = (x+y)//2 if (x+y) even
    # We'll compute this on the fly for efficiency
    
    for step in range(MAX_SIZE):
        if time.time() > deadline:
            break
        
        best_x = -1
        best_score = float('inf')
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            
            # Check if x itself is safe: danger[x] must be 0
            if danger[x] > 0:
                continue
            
            # Compute how many new dangers would be created
            # For each y in selected, if (x+y) even, m=(x+y)//2, danger[m] increases
            # We want to minimize the maximum new danger or the total new danger
            # Use a heuristic: minimize the number of m where danger[m] becomes > 0
            new_danger_count = 0
            for y in selected:
                s = x + y
                if s % 2 == 0:
                    m = s // 2
                    if 1 <= m <= N and danger[m] == 0:
                        new_danger_count += 1
            
            # Prefer x that creates fewest new dangers
            if new_danger_count < best_score:
                best_score = new_danger_count
                best_x = x
            
            # Early termination if perfect
            if best_score == 0:
                break
        
        if best_x == -1:
            break
        
        # Add best_x
        selected.append(best_x)
        selected_set.add(best_x)
        
        # Update danger counts
        for y in selected:
            if y == best_x:
                continue
            s = best_x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N:
                    danger[m] += 1
    
    # Local search: try to swap elements to increase size
    current_size = len(selected)
    
    while current_size < MAX_SIZE and time.time() < deadline:
        # Try to find a swap: remove one element and add two others
        # Or simply try to add one more element by removing one that causes conflicts
        
        improved = False
        selected_list = list(selected)
        selected_set = set(selected_list)
        
        # Recompute danger
        danger = [0] * (N + 1)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                s = selected_list[i] + selected_list[j]
                if s % 2 == 0:
                    m = s // 2
                    if 1 <= m <= N:
                        danger[m] += 1
        
        # Try removing each element and see if we can add two new ones
        for i in range(len(selected_list)):
            if time.time() > deadline:
                break
            
            removed = selected_list[i]
            temp_selected = [x for j, x in enumerate(selected_list) if j != i]
            temp_set = selected_set - {removed}
            
            # Recompute danger without removed
            temp_danger = [0] * (N + 1)
            for a in temp_selected:
                for b in temp_selected:
                    if a == b:
                        continue
                    s = a + b
                    if s % 2 == 0:
                        m = s // 2
                        if 1 <= m <= N:
                            temp_danger[m] += 1
            
            # Try to add two new elements
            added = []
            for x in range(1, N + 1):
                if x in temp_set or x in added:
                    continue
                if temp_danger[x] > 0:
                    continue
                
                # Check if adding x creates no new danger for x
                # And count new dangers
                valid = True
                for y in temp_selected:
                    s = x + y
                    if s % 2 == 0:
                        m = s // 2
                        if 1 <= m <= N and temp_danger[m] > 0:
                            # This would create a conflict for m
                            # But m might not be in the set, so it's okay unless m is in temp_selected or x
                            if m in temp_set or m == x:
                                valid = False
                                break
                
                if valid:
                    added.append(x)
                    # Update temp_danger
                    for y in temp_selected:
                        s = x + y
                        if s % 2 == 0:
                            m = s // 2
                            if 1 <= m <= N:
                                temp_danger[m] += 1
                    temp_set.add(x)
                
                if len(added) == 2:
                    break
            
            if len(added) == 2:
                # Perform the swap
                new_selected = temp_selected + added
                # Verify validity
                if _is_valid(new_selected):
                    selected = new_selected
                    current_size = len(selected)
                    improved = True
                    break
        
        if not improved:
            break
    
    return selected


def _is_valid(S):
    """Check if set S satisfies the condition: no element is the mean of any subset of >=2 others."""
    S = list(S)
    n = len(S)
    if n < 3:
        return True
    
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        m = len(others)
        if m < 2:
            continue
        
        # Check all subsets of others with size >= 2
        # For efficiency, check size 2 first (most common violation)
        for k in range(2, min(m, 5) + 1):  # Check up to size 5 for speed
            from itertools import combinations
            for subset in combinations(others, k):
                if sum(subset) == k * a:
                    return False
    
    return True
