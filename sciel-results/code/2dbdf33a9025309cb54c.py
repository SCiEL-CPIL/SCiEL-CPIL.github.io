"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_mis(adj, n, time_limit=10.0):
        """
        Solves Maximum Independent Set using Tomita's algorithm.
        adj: list of lists of neighbors.
        n: number of vertices.
        """
        # Convert to 0-indexed for internal use, but we keep original indices
        # adj[i] contains neighbors of i
        
        # We use a recursive branch and bound.
        # To handle time limits in Python recursion, we track a global deadline.
        deadline = time.time() + time_limit
        
        # Precompute degrees for initial ordering? 
        # Tomita's algorithm benefits from coloring the subgraph.
        # We will implement a simplified version of Tomita's MIS.
        
        # For efficiency in Python, we represent the candidate set R as a list.
        # We maintain a list of vertices in R.
        # We also need to order them by color.
        
        # Since n is small (<=240), we can use bitmasks if n <= 64, but n=240 requires sets or lists.
        # Using lists of indices is safer.
        
        # Helper to get neighbors of v within R
        def get_neighbors_in_R(v, R_set):
            return [u for u in adj[v] if u in R_set]
        
        # Simple coloring heuristic for R
        # We want to partition R into independent sets (color classes).
        # The number of colors is an upper bound on the size of MIS in R.
        def color_subgraph(R):
            # R is a list of vertices
            # We greedily assign colors.
            # color[v] = c
            colors = {}
            next_color = 0
            # Order R by degree in the subgraph? Or just iterate.
            # For Tomita, we usually order by color.
            # Let's just do a simple greedy coloring.
            # To make it effective, we should process vertices in a specific order.
            # Let's use the order in R.
            
            # Actually, a better approach for the bound is to compute a proper coloring of R.
            # We will use a simple greedy coloring.
            # To speed up, we can sort R by degree in the induced subgraph.
            # But that's expensive. Let's just iterate R.
            
            # We want to minimize the number of colors.
            # Greedy: for each v in R, assign the smallest color not used by neighbors in R.
            
            # To make it faster, we can use a bitset for colors if n is small, but here n=240.
            # We'll use a dictionary.
            
            # Note: The order of coloring affects the bound.
            # Tomita's algorithm specifically reorders R based on the coloring.
            
            # Let's implement a simple coloring that returns the color for each vertex and the max color used.
            # We'll also return the vertices sorted by color, then by degree (heuristic).
            
            # For simplicity and speed in Python, let's use a basic greedy coloring.
            # We iterate through R.
            # To improve the bound, we should color in an order that tends to use fewer colors?
            # Actually, we want a *valid* coloring. The number of colors is the upper bound.
            # Greedy coloring with a good order (e.g., descending degree) gives a tighter bound.
            
            # Let's sort R by degree in the induced subgraph (descending).
            # This is O(|R|^2) which is fine for |R| <= 240.
            
            R_set = set(R)
            degs = {}
            for v in R:
                degs[v] = len([u for u in adj[v] if u in R_set])
            
            R_sorted = sorted(R, key=lambda x: degs[x], reverse=True)
            
            colors = {}
            max_color = 0
            # For each vertex, check neighbors in R_sorted that have already been colored.
            # We need to know which colors are used by neighbors.
            
            # To make this O(n^2), we can keep a list of used colors per vertex?
            # Or just check neighbors.
            
            for v in R_sorted:
                # Find neighbors of v that are in R and already processed (i.e., in R_sorted before v? No, just in R).
                # Wait, greedy coloring requires that when we color v, we know the colors of its neighbors.
                # If we process in R_sorted order, we must ensure neighbors are colored?
                # No, standard greedy coloring works on any order. We just pick the smallest available color.
                # But for the bound to be tight, the order matters.
                # Let's just use the R_sorted order.
                
                used_colors = set()
                for u in adj[v]:
                    if u in R_set and u in colors:
                        used_colors.add(colors[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                
                colors[v] = c
                if c > max_color:
                    max_color = c
            
            return colors, max_color, R_sorted

        best_size = 0
        best_set = set()
        
        # Global variables for recursion
        # We pass R as a list.
        
        def expand(R):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return
            
            if not R:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            # Color R
            colors, max_color, R_colored = color_subgraph(R)
            
            # Pruning: if len(current_set) + max_color <= best_size, prune
            if len(current_set) + max_color <= best_size:
                return
            
            # Process vertices in reverse color order (highest color first)
            # This is a key part of Tomita's algorithm.
            # Vertices with higher colors are "more constrained" in a sense, or rather,
            # the bound decreases faster if we pick them?
            # Actually, the standard Tomita processes from last color to first.
            
            # R_colored is sorted by degree. We need to sort by color?
            # The coloring function returned R_sorted by degree.
            # We need to group by color.
            
            # Let's re-sort R_colored by color descending, then by degree descending?
            # Or just iterate colors from max_color down to 0.
            
            # For efficiency, let's just iterate R_colored in reverse order of the greedy coloring assignment?
            # No, let's sort R_colored by color descending.
            R_by_color = sorted(R_colored, key=lambda x: colors[x], reverse=True)
            
            for v in R_by_color:
                if time.time() > deadline:
                    return
                
                # Prune: if len(current_set) + colors[v] <= best_size, break (since colors are descending)
                if len(current_set) + colors[v] <= best_size:
                    break
                
                # Branch: include v
                current_set.append(v)
                
                # New R is R \ (N(v) \cup {v})
                # R is R_by_color? No, R is the set of candidates.
                # We should maintain R as a set or list.
                # Let's use a set for R for fast lookup, but we need to pass a list to color_subgraph.
                # This is getting expensive.
                
                # Optimization: Pass R as a set.
                # Modify color_subgraph to take a set?
                # Or convert set to list.
                
                # Let's refactor to use sets for R.
                
                pass # Placeholder for logic, will implement below

        # Refactored expand to use sets for R
        # We need to be careful with performance.
        
        # Let's implement a cleaner version.
        
        def expand(R_set):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return
            
            if not R_set:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            # Color R_set
            # Convert to list for coloring
            R_list = list(R_set)
            
            # Simple coloring
            # We sort by degree in induced subgraph
            degs = {}
            for v in R_list:
                degs[v] = len([u for u in adj[v] if u in R_set])
            
            R_sorted = sorted(R_list, key=lambda x: degs[x], reverse=True)
            
            colors = {}
            max_color = 0
            for v in R_sorted:
                used_colors = set()
                for u in adj[v]:
                    if u in R_set and u in colors:
                        used_colors.add(colors[u])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c > max_color:
                    max_color = c
            
            if len(current_set) + max_color <= best_size:
                return
            
            # Sort by color descending
            R_by_color = sorted(R_sorted, key=lambda x: colors[x], reverse=True)
            
            for v in R_by_color:
                if time.time() > deadline:
                    return
                
                if len(current_set) + colors[v] <= best_size:
                    break
                
                current_set.append(v)
                
                # New R: R_set \ (neighbors of v) \ {v}
                new_R_set = set()
                for u in R_set:
                    if u != v and u not in adj[v]:
                        new_R_set.add(u)
                
                expand(new_R_set)
                
                current_set.pop()
                
                # Remove v from R_set for the next iteration of the loop?
                # In Tomita's algorithm, we remove v from R after branching.
                # This is because if we don't pick v, we still consider the other vertices.
                # But we must ensure we don't pick v again.
                # So yes, remove v from R_set.
                R_set.remove(v)
                if not R_set:
                    break

        current_set = []
        expand(set(range(n)))
        
        return list(best_set)

    def solve_one(vals, time_budget=35.0):
        n = len(vals)
        start = time.time()
        
        # Build conflict graph
        # adj[i] is a set of j such that vals[i] and vals[j] conflict.
        # Conflict if vals[i] + vals[j] is in vals_set OR 2*vals[i] == vals[j] (which is covered if we check sums carefully? 
        # No, 2*vals[i] = vals[k] means i and i conflict? No, i and i is not an edge. 
        # The condition is: no a,b,c in S such that a+b=c.
        # This implies:
        # 1. If a=b, then 2a=c. So if 2a is in S, a cannot be in S? 
        #    Wait. If a is in S, and 2a is in S, then a+a=2a is a violation.
        #    So if 2a is in the universe, and we include a, we cannot include 2a.
        #    Also, if we include 2a, we cannot include a.
        #    So there is an edge between a and 2a?
        #    Let's check: S={a, 2a}. a+a=2a. Violation.
        #    So yes, if 2a is in the universe, a and 2a cannot both be in S.
        #    This is an edge (a, 2a).
        #
        # 2. If a != b, then a+b=c.
        #    If c is in the universe, then a, b, c cannot all be in S.
        #    This is a hyperedge (a,b,c).
        #    This is NOT a simple graph edge.
        #
        # My previous graph formulation was WRONG for the ternary case.
        # A sum-free set is NOT an independent set in a simple graph where edges are (a,b) if a+b in V.
        # Because even if (a,b) is not an edge (meaning a+b not in V), we might have a+b=c where c is in V.
        # Wait. If a+b=c, then (a,b) IS a pair whose sum is in V.
        # So if we define an edge between a and b if a+b is in V, then:
        # If a and b are in S, and there is an edge (a,b), then c=a+b is in V.
        # But is c in S?
        # The violation is if c is ALSO in S.
        # So the graph MIS formulation only works if we assume that if a+b is in V, we MUST exclude c?
        # No. We want to maximize S.
        # If we have an edge (a,b), it means a and b "conflict" with c=a+b.
        # It does NOT mean a and b conflict with each other.
        # Example: V={1, 2, 3}. Edge (1,2) because 1+2=3.
        # S={1, 2} is sum-free? 1+1=2 (2 in S? Yes). Violation.
        # S={1, 3} is sum-free? 1+1=2 (not in S). 1+3=4 (not in V). 3+3=6. OK.
        # S={2, 3} is sum-free? 2+2=4. 2+3=5. 3+3=6. OK.
        # S={1, 2, 3} is NOT sum-free.
        
        # So the conflict is ternary. MIS on a simple graph is NOT equivalent.
        
        # However, we can transform it?
        # No, Max Sum-Free Subset is NP-Hard, and not equivalent to MIS on a simple graph directly.
        
        # BUT, we can use a different approach:
        # Since n is small, we can use Integer Linear Programming (ILP) or a SAT solver?
        # Or we can stick to local search but make it much better.
        
        # Let's reconsider the local search. The previous one was weak.
        # A better heuristic:
        # 1. Start with all elements.
        # 2. While there is a violation (a,b,c in S with a+b=c), remove one of a,b,c.
        #    Which one to remove?
        #    Heuristic: Remove the one with the highest "degree" in the conflict hypergraph.
        #    i.e., count how many triples each element is part of.
        #    Remove the most "connected" element.
        # 3. Try to add back elements?
        
        # This is a greedy deletion algorithm.
        
        # Let's implement a "Greedy Deletion" with "Add-back" repair.
        
        def solve_greedy_deletion(vals, time_budget=30.0, seed=42):
            n = len(vals)
            if n == 0: return []
            
            vals_set = set(vals)
            # Precompute triples
            # triple_counts[v]: number of triples (a,b,c) with a+b=c involving v.
            # Also store the triples for removal.
            
            # For efficiency, let's just iterate.
            # S = set(vals)
            # While there is a violation:
            #   Find a violation.
            #   Remove the element with highest frequency in violations.
            
            # This is O(n^3) per step?
            # We can optimize by maintaining a count.
            
            # Let's try a different angle:
            # The set of odd numbers is always sum-free.
            # The set of numbers in (n/3, n] is sum-free?
            # If a,b > n/3, then a+b > 2n/3.
            # If a+b <= n, then c is in (2n/3, n].
            # This doesn't guarantee sum-free.
            
            # Let's use the "Large Half" heuristic.
            # Sort vals.
            # Take the top half?
            # If a,b are in the top half, a+b is large.
            # If a+b is in the set, it must be even larger.
            # This works if the set is dense in [1, N].
            # Here the set is random in [-10000, 10000].
            
            # Let's go back to the Local Search but fix the "Kick" to be smarter.
            # Use "Conflict-based" kick.
            # If we are stuck, find a set of elements that are "blocking" many additions.
            # Remove them.
            
            # Actually, the best simple heuristic for Sum-Free Sets is:
            # 1. Start with empty set.
            # 2. Add elements in random order if they don't violate.
            # 3. This is the greedy init.
            
            # To improve, use "Simulated Annealing" on the size of the set.
            # State: S.
            # Move: Remove a random element from S, or Add a random element from V\S.
            # If Add causes violation, reject (or allow with penalty?).
            # If we allow violations, we need a repair step.
            
            # Let's use the "Repair" approach properly.
            # 1. Start with a large random set (e.g., 60% of elements).
            # 2. Repair: While violations exist, remove an element from a violation.
            #    To maximize size, remove the element that is in the FEWEST violations?
            #    No, remove the one that "costs" the least.
            #    Actually, removing any element fixes at least one violation.
            #    If we remove an element involved in many violations, we fix more.
            #    So remove the one with highest conflict count.
            # 3. After repair, try to add elements back.
            
            # This is a standard approach.
            
            def repair(S):
                S_set = set(S)
                changed = True
                while changed:
                    changed = False
                    # Find a violation
                    # To be fast, pick a random a, b from S and check if a+b in S.
                    # If yes, we have a violation (a,b,a+b).
                    # Count conflicts for a, b, a+b.
                    # Remove the one with highest count.
                    
                    # We need to be careful not to loop forever.
                    # Limit iterations.
                    
                    # For simplicity, just pick the first violation found.
                    # This is O(n^2) per check.
                    
                    # Let's just iterate all pairs?
                    # n=240, n^2 = 57600.
                    # If we do this 100 times, it's 5M ops. Fine.
                    
                    violation_found = False
                    # We need to count conflicts to decide what to remove.
                    # Let's just remove the first one found?
                    # That might be suboptimal.
                    
                    # Let's collect all violations in one pass.
                    # conflicts = {v: count}
                    # Then remove max.
                    
                    conflicts = {}
                    for a in S:
                        for b in S:
                            c = a + b
                            if c in S_set:
                                # Violation
                                conflicts[a] = conflicts.get(a, 0) + 1
                                conflicts[b] = conflicts.get(b, 0) + 1
                                conflicts[c] = conflicts.get(c, 0) + 1
                                # We can break early if we just want one?
                                # No, we want to remove the most connected.
                    
                    if conflicts:
                        # Remove element with max conflicts
                        max_v = max(conflicts, key=conflicts.get)
                        S.remove(max_v)
                        S_set.remove(max_v)
                        changed = True
                        # Restart the loop to find more violations
                    else:
                        break
                
                return S

            def can_add(S_set, x):
                if 2 * x in S_set:
                    return False
                for a in S_set:
                    if a + x in S_set:
                        return False
                return True

            best_S = set()
            best_size = 0
            rng = random.Random(seed)
            end_time = start_time + time_budget
            
            # Multiple restarts
            while time.time() < end_time:
                # Init: Random subset
                # Size of init: n/2?
                init_size = n // 2
                S = set(rng.sample(vals, init_size))
                
                # Repair
                S = repair(S)
                
                # Augment: Try to add elements
                improved = True
                while improved:
                    improved = False
                    # Try random additions
                    for _ in range(50):
                        x = rng.choice(vals)
                        if x not in S:
                            if can_add(S, x):
                                S.add(x)
                                improved = True
                                break
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    
                # Kick: Remove 1-2 elements and repair
                if len(S) > 0:
                    S_temp = set(S)
                    # Remove 1 or 2
                    for _ in range(rng.randint(1, 2)):
                        if S_temp:
                            x = rng.choice(list(S_temp))
                            S_temp.remove(x)
                    
                    S_temp = repair(S_temp)
                    
                    # Augment
                    improved = True
                    while improved:
                        improved = False
                        for _ in range(50):
                            x = rng.choice(vals)
                            if x not in S_temp:
                                if can_add(S_temp, x):
                                    S_temp.add(x)
                                    improved = True
                                    break
                    
                    if len(S_temp) > best_size:
                        best_size = len(S_temp)
                        best_S = set(S_temp)
                    
                    # Even if not better, accept with prob 0.1 to explore
                    elif len(S_temp) == best_size and rng.random() < 0.1:
                        best_S = set(S_temp)

            return list(best_S)

        # Use the improved heuristic
        cand = solve_greedy_deletion(vals, time_budget=time_budget, seed=42)
        
        # Fallback to odd numbers if it fails or is small
        odd = [x for x in vals if x % 2 != 0]
        if len(cand) < len(odd):
            cand = odd
            
        if not is_sum_free(cand):
            cand = odd
            
        return cand

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        
        # Allocate time
        if size <= 60:
            budget = 30.0
        elif size <= 120:
            budget = 45.0
        else:
            budget = 60.0
        
        remaining = DEADLINE - elapsed - 5
        budget = min(budget, remaining)
        
        result = solve_one(vals, time_budget=budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
