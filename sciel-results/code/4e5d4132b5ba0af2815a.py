"""Find large sum-free subsets using branch-and-bound exact search with greedy seeding."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_sum_free(S):
        """Greedy construction of a large sum-free subset."""
        S_list = list(S)
        # Try multiple orderings and pick the best
        best = set()
        
        # Strategy 1: sort by absolute value descending (large numbers less likely to be sums)
        for order in [
            sorted(S_list, key=lambda x: -abs(x)),
            sorted(S_list, key=lambda x: abs(x)),
            sorted(S_list, key=lambda x: x),
            sorted(S_list, key=lambda x: -x),
            sorted(S_list, key=lambda x: x % 2),  # odds first
            sorted(S_list, key=lambda x: -x % 2),  # evens first
        ]:
            candidate = set()
            for x in order:
                # Check if adding x violates sum-free property
                valid = True
                for y in candidate:
                    if x + y in candidate:
                        valid = False
                        break
                    if x + y == x:  # y=0, but 0 not in set
                        valid = False
                        break
                    if x + y == y:  # x=0, but 0 not in set
                        valid = False
                        break
                if valid:
                    candidate.add(x)
            if len(candidate) > len(best):
                best = candidate
        
        return best

    def branch_and_bound(S, time_limit=None):
        """
        Branch and bound to find maximum sum-free subset.
        S is a list of integers.
        Returns the largest sum-free subset found.
        """
        n = len(S)
        if n == 0:
            return set()
        
        # Get a good initial lower bound via greedy
        best_set = greedy_sum_free(S)
        best_size = len(best_set)
        
        # Build conflict structure: for each element, which other elements conflict with it
        # Two elements a, b conflict if there exists c in S such that a+b=c, or a+c=b, or b+c=a
        # Actually, the constraint is: we cannot have three elements a,b,c with a+b=c
        # So when building a subset, adding x to current set C is invalid if there exist
        # y, z in C such that y+z=x, or x+y=z, or x+z=y for some z in C
        
        # Precompute: for each pair (i,j), check if S[i]+S[j] is in S
        # If so, then we can't have all three S[i], S[j], S[i]+S[j]
        
        S_set = set(S)
        
        # For each element index, precompute which other indices it "conflicts" with
        # in the sense that they form a sum triple
        # conflict[i] = set of indices j such that S[i] + S[j] is in S (and S[i]+S[j] is some S[k])
        # Actually we need: for a given x, the forbidden pairs in the current set
        
        # Let's use a different approach: represent the problem as finding the largest subset
        # with no solution to a+b=c
        
        # Branch and bound with bitsets for efficiency
        # Use recursive approach with pruning
        
        # Precompute sum relationships
        # For each element s, find all pairs (a,b) in S such that a+b=s
        # Then if s is in our subset, we can't have both a and b
        
        # Build: for each element value v, list of (a,b) pairs in S where a+b=v
        sum_pairs = {}
        for v in S:
            pairs = []
            for a in S:
                b = v - a
                if b in S_set and a <= b:  # avoid duplicates, but we need ordered for checking
                    pairs.append((a, b))
            sum_pairs[v] = pairs
        
        # Also: for each element a, which elements c can be formed as a+b for some b in S?
        # i.e., c = a + b where b in S and c in S
        can_form = {}
        for a in S:
            forms = set()
            for b in S:
                c = a + b
                if c in S_set:
                    forms.add(c)
            can_form[a] = forms
        
        # For branch and bound, use a simple recursive approach with pruning
        # Sort elements to help with pruning
        
        # Use a heuristic: process elements in order of "constraint tightness"
        # Elements that are involved in many sum relations should be decided first
        
        # Count how many sum relations each element is involved in
        relation_count = {v: 0 for v in S}
        for v in S:
            for a, b in sum_pairs[v]:
                relation_count[a] += 1
                relation_count[b] += 1
                relation_count[v] += 1
        
        # Order: elements with more relations first (more constrained)
        ordered = sorted(S, key=lambda x: -relation_count[x])
        
        # Map value to position in ordered list
        pos_in_ordered = {v: i for i, v in enumerate(ordered)}
        
        # Recursive branch and bound
        # State: current subset (as set), index in ordered list to consider next
        # We can either include or exclude ordered[index]
        
        # Use iterative deepening or just DFS with pruning
        
        best = list(best_set)
        best_size = len(best_set)
        
        def dfs(idx, current_set, current_list):
            nonlocal best, best_size
            
            if time.time() > time_limit:
                return
            
            # Upper bound: current size + remaining elements
            remaining = n - idx
            if len(current_set) + remaining <= best_size:
                return
            
            if idx == n:
                if len(current_set) > best_size:
                    best = list(current_set)
                    best_size = len(current_set)
                return
            
            v = ordered[idx]
            
            # Option 1: include v
            # Check if adding v is valid
            valid = True
            for y in current_set:
                if v + y in current_set:
                    valid = False
                    break
            if valid:
                new_set = set(current_set)
                new_set.add(v)
                dfs(idx + 1, new_set, current_list + [v])
            
            # Option 2: exclude v
            dfs(idx + 1, current_set, current_list)
        
        # Start DFS with empty set
        dfs(0, set(), [])
        
        return set(best)

    def solve_instance(S, time_limit):
        """Solve one instance, returning the best sum-free subset found."""
        if not S:
            return []
        
        n = len(S)
        
        # For small instances, try exact search
        # For larger ones, use greedy + local search
        
        # First get a good greedy solution
        greedy_sol = greedy_sum_free(S)
        greedy_size = len(greedy_sol)
        
        # Try branch and bound if the instance is small enough
        if n <= 60:
            # Use branch and bound with time limit
            sol = branch_and_bound(S, time_limit)
            if len(sol) >= greedy_size:
                greedy_sol = sol
                greedy_size = len(sol)
        
        # Local search: try to improve by swapping elements
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_set = set(S)
            current_set = set(greedy_sol)
            
            for x in S_set - current_set:
                # Try adding x
                valid = True
                for y in current_set:
                    if x + y in current_set:
                        valid = False
                        break
                if valid:
                    new_set = current_set | {x}
                    greedy_sol = list(new_set)
                    greedy_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one element, add another
            for x in current_set:
                for y in S_set - current_set:
                    new_set = (current_set - {x}) | {y}
                    # Check if new_set is sum-free
                    valid = True
                    for a in new_set:
                        for b in new_set:
                            if a + b in new_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(new_set) > len(current_set):
                        greedy_sol = list(new_set)
                        greedy_size = len(new_set)
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(greedy_sol)

    # Solve each instance
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time budget: give more time to larger instances
    total_time = DEADLINE - start_time
    # Proportional to sqrt of size (since B&B is exponential-ish)
    weights = [1.0, 2.0, 4.0]
    total_weight = sum(weights)
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        time_limit = start_time + total_time * (weights[i] / total_weight)
        # Ensure we don't exceed the overall deadline
        time_limit = min(time_limit, DEADLINE - 2.0)
        
        sol = solve_instance(S, time_limit)
        results.append(sol)
        
        # Update remaining time
        elapsed = time.time() - start_time
        remaining = DEADLINE - 2.0 - elapsed
        if remaining < 0:
            # Out of time, return what we have
            break
    
    # If we ran out of time, fill remaining with greedy
    while len(results) < 3:
        i = len(results)
        size, seed = instances[i]
        S = instance(size, seed)
        sol = greedy_sum_free(S)
        results.append(sorted(sol))
    
    return results
