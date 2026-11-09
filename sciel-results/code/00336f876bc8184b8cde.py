import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 180
    # Precompute all forbidden triples (a, b, c) with a < b, c = ab/(a+b), c < a, all distinct
    # For each pair (a, b) with a < b, check if c = ab/(a+b) is integer and c < a and c not in {a,b}
    forbidden = []  # list of (a, b, c) with c < a < b
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if c < a and c != b and c != a:
                    # c < a < b, all distinct
                    forbidden.append((a, b, c))
    
    # We want to find a large subset S of [1,180] with no forbidden triple fully contained in S.
    # This is the maximum independent set in a 3-uniform hypergraph.
    # Use greedy: start with all elements, iteratively remove the element in the most remaining forbidden triples.
    
    # Build incidence: for each element, list of triples containing it
    elem_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(forbidden):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
            
    triple_elems = [(a, b, c) for (a, b, c) in forbidden]

    def can_add(x, in_set):
        """Check if x can be added to in_set without creating a forbidden triple."""
        for idx in elem_to_triples[x]:
            a, b, c = triple_elems[idx]
            # Check if the other two elements are in in_set
            others = [y for y in (a, b, c) if y != x]
            if in_set[others[0]] and in_set[others[1]]:
                return False
        return True

    def polish(in_set, removed):
        """Try to add back removed elements if they don't cause conflicts."""
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of removed to allow modification
            for x in list(removed):
                if can_add(x, in_set):
                    in_set[x] = True
                    removed.remove(x)
                    changed = True
        return in_set, removed

    def greedy_solve(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        in_set = [True] * (n + 1)  # 1-indexed
        triple_active = [True] * len(forbidden)
        
        # Initial count: all triples are active
        elem_active_count = [0] * (n + 1)
        for idx, (a, b, c) in enumerate(forbidden):
            elem_active_count[a] += 1
            elem_active_count[b] += 1
            elem_active_count[c] += 1
        
        removed = set()
        while True:
            if time.time() > deadline:
                break
            # Find element with max active count that is still in set
            best_elem = -1
            best_count = 0
            if randomize:
                # Add some noise to break ties or explore
                # We'll just pick randomly from top k or use random tie-breaking
                max_count = 0
                for x in range(1, n + 1):
                    if in_set[x] and elem_active_count[x] > max_count:
                        max_count = elem_active_count[x]
                if max_count == 0:
                    break
                candidates = [x for x in range(1, n + 1) if in_set[x] and elem_active_count[x] == max_count]
                best_elem = random.choice(candidates)
                best_count = max_count
            else:
                for x in range(1, n + 1):
                    if in_set[x] and elem_active_count[x] > best_count:
                        best_count = elem_active_count[x]
                        best_elem = x
                if best_elem == -1 or best_count == 0:
                    break
            
            # Remove best_elem
            in_set[best_elem] = False
            removed.add(best_elem)
            
            # Update: for each triple containing best_elem that was active, mark it inactive
            for idx in elem_to_triples[best_elem]:
                if triple_active[idx]:
                    a, b, c = triple_elems[idx]
                    triple_active[idx] = False
                    for x in (a, b, c):
                        if in_set[x]:
                            elem_active_count[x] -= 1
        
        # Polish the result
        in_set, removed = polish(in_set, removed)
        
        # Return the set
        result = [x for x in range(1, n + 1) if in_set[x]]
        return result
    
    # Run greedy multiple times with different random orderings to find a better solution
    best_result = None
    best_size = 0
    
    # First, try the basic greedy
    result = greedy_solve(randomize=False)
    best_result = result
    best_size = len(result)
    
    # Try randomized greedy: break ties randomly
    # Use more trials if time permits
    trials = 0
    while time.time() < deadline - 5:
        if time.time() > deadline - 5:
            break
        
        result = greedy_solve(randomize=True, seed=trials)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
        trials += 1
        
        # If we've done a lot of trials, check time more strictly
        if trials % 10 == 0 and time.time() > deadline - 2:
            break

    if not best_result:
        best_result = list(range(91, 181))
    
    return best_result
