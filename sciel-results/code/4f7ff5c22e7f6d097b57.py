import time
from math import gcd
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    LIMIT = 512
    candidates = list(range(1, LIMIT + 1))
    
    # Precompute lcm compatibility
    compat = [set() for _ in range(LIMIT + 1)]
    for a in range(1, LIMIT + 1):
        for b in range(a, LIMIT + 1):
            g = gcd(a, b)
            if a * b // g <= LIMIT:
                compat[a].add(b)
                compat[b].add(a)
    
    def is_compatible_with_set(x, S_set):
        """Check if x is compatible with all elements in S_set."""
        for y in S_set:
            if y not in compat[x]:
                return False
        return True
    
    def greedy_add(S_set, remaining):
        """Greedily add compatible elements from remaining to S_set."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in list(remaining):
                if is_compatible_with_set(x, S_set):
                    S_set.add(x)
                    remaining.discard(x)
                    changed = True
                    break
        return S_set
    
    def local_search(S_set, remaining, max_iter=1000):
        """Try 1-for-1, 1-for-2, 2-for-1 swaps."""
        best_S = set(S_set)
        best_size = len(best_S)
        remaining = set(remaining)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # 1-for-1: remove x from S, add y from remaining
            for x in list(best_S):
                if time.time() > deadline - 5:
                    break
                S_without_x = best_S - {x}
                for y in list(remaining):
                    if y in best_S:
                        continue
                    if is_compatible_with_set(y, S_without_x):
                        new_S = S_without_x | {y}
                        if len(new_S) > best_size:
                            best_S = new_S
                            best_size = len(best_S)
                            remaining = set(candidates) - best_S
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # 1-for-2: remove x from S, add y1, y2 from remaining
                found = False
                for x in list(best_S):
                    if time.time() > deadline - 5:
                        break
                    S_without_x = best_S - {x}
                    compatible_with_S = [y for y in remaining if is_compatible_with_set(y, S_without_x)]
                    for i in range(len(compatible_with_S)):
                        for j in range(i+1, len(compatible_with_S)):
                            y1 = compatible_with_S[i]
                            y2 = compatible_with_S[j]
                            if y2 in compat[y1]:
                                new_S = S_without_x | {y1, y2}
                                if len(new_S) > best_size:
                                    best_S = new_S
                                    best_size = len(best_S)
                                    remaining = set(candidates) - best_S
                                    found = True
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                
                # 2-for-1: remove x1,x2 from S, add y from remaining
                found = False
                S_list = list(best_S)
                for i in range(len(S_list)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i+1, len(S_list)):
                        x1, x2 = S_list[i], S_list[j]
                        S_without = best_S - {x1, x2}
                        for y in remaining:
                            if y in best_S:
                                continue
                            if is_compatible_with_set(y, S_without):
                                new_S = S_without | {y}
                                if len(new_S) > best_size:
                                    best_S = new_S
                                    best_size = len(best_S)
                                    remaining = set(candidates) - best_S
                                    found = True
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                
                # No improvement, break
                break
        
        return best_S
    
    # Maximum Weight Closure on Divisor Lattice
    # Key insight: if we include a number x, we may need to include certain "witnesses"
    # to maintain LCM constraints. Model as closure problem.
    
    def max_weight_closure():
        """
        Solve maximum weight closure using min-cut.
        Nodes: 1..512
        Weight: +1 for each node (we want to maximize count)
        Edges: if including x forces us to exclude y (or include z), add directed edge.
        
        For LCM constraint: if we include both x and y, we need lcm(x,y) <= 512.
        This is a pairwise constraint, not a simple closure.
        
        Alternative: Use the fact that the compatibility graph is a "cograph" or has
        special structure. Let's try a different approach.
        """
        # Build conflict graph: edge (x,y) if lcm(x,y) > 512
        # We want maximum independent set in this graph.
        # This is NP-hard in general, but the graph has special structure.
        
        # Try branch and bound with good initial solution
        initial = greedy_add(set([1]), set(candidates) - {1})
        initial = local_search(initial, set(candidates) - initial, max_iter=500)
        
        # Branch and bound
        best_solution = set(initial)
        best_size = len(best_solution)
        
        # Order candidates by degree (high degree first for better pruning)
        degrees = [len(compat[x]) for x in range(1, LIMIT + 1)]
        sorted_candidates = sorted(range(1, LIMIT + 1), key=lambda x: -degrees[x])
        
        def branch_and_bound(remaining, current_set, depth=0):
            nonlocal best_solution, best_size
            
            if time.time() > deadline - 10:
                return
            
            # Upper bound: current_size + number of remaining that are compatible with all in current
            compatible_remaining = [x for x in remaining if is_compatible_with_set(x, current_set)]
            upper_bound = len(current_set) + len(compatible_remaining)
            
            if upper_bound <= best_size:
                return
            
            if not remaining:
                if len(current_set) > best_size:
                    best_solution = set(current_set)
                    best_size = len(current_set)
                return
            
            # Choose next vertex to branch on
            # Pick vertex with highest degree in remaining subgraph
            if not compatible_remaining:
                if len(current_set) > best_size:
                    best_solution = set(current_set)
                    best_size = len(current_set)
                return
            
            # Try including the first compatible vertex
            v = compatible_remaining[0]
            
            # Branch 1: Include v
            new_remaining = set(x for x in compatible_remaining if x != v and x in compat[v])
            branch_and_bound(new_remaining, current_set | {v}, depth + 1)
            
            # Branch 2: Exclude v
            new_remaining_excl = set(compatible_remaining) - {v}
            branch_and_bound(new_remaining_excl, current_set, depth + 1)
        
        branch_and_bound(set(sorted_candidates), set(initial))
        
        return best_solution
    
    # Multiple restarts with different strategies
    best_result = set()
    best_size = 0
    
    # Strategy 1: Greedy from scratch
    S1 = set([1])  # Always include 1
    remaining1 = set(candidates) - S1
    S1 = greedy_add(S1, remaining1)
    S1 = local_search(S1, set(candidates) - S1, max_iter=500)
    if len(S1) > best_size:
        best_size = len(S1)
        best_result = S1
    
    if time.time() > deadline - 15:
        return list(best_result)
    
    # Strategy 2: Start with powers of 2
    powers_of_2 = set()
    p = 1
    while p <= LIMIT:
        powers_of_2.add(p)
        p *= 2
    S2 = set(powers_of_2)
    remaining2 = set(candidates) - S2
    S2 = greedy_add(S2, remaining2)
    S2 = local_search(S2, set(candidates) - S2, max_iter=500)
    if len(S2) > best_size:
        best_size = len(S2)
        best_result = S2
    
    if time.time() > deadline - 15:
        return list(best_result)
    
    # Strategy 3: Randomized restarts
    import random
    rng = random.Random(42)
    
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        # Random initial set
        S_r = set()
        # Include 1
        S_r.add(1)
        # Add some random small numbers
        for x in range(2, min(30, LIMIT+1)):
            if rng.random() < 0.3:
                if is_compatible_with_set(x, S_r):
                    S_r.add(x)
        
        remaining_r = set(candidates) - S_r
        S_r = greedy_add(S_r, remaining_r)
        S_r = local_search(S_r, set(candidates) - S_r, max_iter=300)
        
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_result = S_r
    
    # Strategy 4: Branch and bound with initial solution
    if time.time() < deadline - 15:
        bb_result = max_weight_closure()
        if len(bb_result) > best_size:
            best_size = len(bb_result)
            best_result = bb_result
    
    # Final: try to improve best_result with more local search
    if time.time() < deadline - 5:
        remaining_final = set(candidates) - best_result
        best_result = local_search(best_result, remaining_final, max_iter=1000)
    
    # Ensure nonempty
    if not best_result:
        best_result = {1}
    
    return list(best_result)

