"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130
    prescribed = [1, 2, 5, 11]
    N = 284
    
    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Greedy initial solution
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    for x in range(12, N + 1):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    best_size = len(a)
    best_a = list(a)
    
    # Graph-theoretic approach: conflict graph on candidate integers
    # Two candidates x, y conflict if adding both to the current set would create a duplicate sum
    # We build the conflict graph and find a large independent set
    
    def build_conflict_graph(selected, candidates):
        """Build conflict graph between candidates.
        Returns adjacency list for candidates."""
        n_cand = len(candidates)
        adj = [set() for _ in range(n_cand)]
        
        # Precompute sums with selected elements
        sel_sums = set()
        for i, x in enumerate(selected):
            for y in selected[i:]:
                sel_sums.add(x + y)
        
        # For each pair of candidates, check if they conflict
        for i in range(n_cand):
            ci = candidates[i]
            for j in range(i + 1, n_cand):
                cj = candidates[j]
                
                # Check if ci and cj conflict with each other or with selected
                # Conflict if: (ci + cj) is in sel_sums, or ci+cj conflicts with existing sums
                # More precisely: adding both ci and cj should not create duplicate sums
                # Sums involving ci: ci + x for x in selected, 2*ci
                # Sums involving cj: cj + x for x in selected, 2*cj
                # Also ci + cj
                
                # Check ci + cj against existing sums
                if ci + cj in sel_sums:
                    adj[i].add(j)
                    adj[j].add(i)
                    continue
                
                # Check if ci's new sums conflict with cj's new sums
                ci_sums = {ci + x for x in selected} | {2 * ci}
                cj_sums = {cj + x for x in selected} | {2 * cj}
                
                # Also need to check ci+cj itself
                all_new = ci_sums | cj_sums | {ci + cj}
                
                # Check for duplicates within all_new
                if len(all_new) < len(ci_sums) + len(cj_sums) + 1:
                    adj[i].add(j)
                    adj[j].add(i)
                    continue
                
                # Check if any new sum conflicts with sel_sums
                if all_new & sel_sums:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj
    
    def greedy_independent_set(adj, n, random_order=None):
        """Greedy coloring heuristic to find independent set."""
        if random_order is None:
            random_order = list(range(n))
            random.shuffle(random_order)
        
        in_set = [False] * n
        result = []
        
        for v in random_order:
            if in_set[v]:
                continue
            # Check if v can be added (no conflict with existing in_set)
            can_add = True
            for u in adj[v]:
                if in_set[u]:
                    can_add = False
                    break
            if can_add:
                in_set[v] = True
                result.append(v)
        
        return result
    
    def local_search_independent_set(adj, n, initial_set, max_iters=5000):
        """Local search to improve independent set."""
        in_set = [False] * n
        for v in initial_set:
            in_set[v] = True
        
        current_size = len(initial_set)
        best_set = list(initial_set)
        best_size = current_size
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try to add a vertex
            added = False
            candidates_to_add = [v for v in range(n) if not in_set[v]]
            random.shuffle(candidates_to_add)
            
            for v in candidates_to_add:
                can_add = True
                for u in adj[v]:
                    if in_set[u]:
                        can_add = False
                        break
                if can_add:
                    in_set[v] = True
                    added = True
                    break
            
            if not added:
                # Try to swap: remove one, add another
                swapped = False
                removable = list(initial_set)
                random.shuffle(removable)
                
                for rm in removable:
                    in_set[rm] = False
                    # Try to add multiple vertices
                    added_any = False
                    for v in range(n):
                        if not in_set[v]:
                            can_add = True
                            for u in adj[v]:
                                if in_set[u]:
                                    can_add = False
                                    break
                            if can_add:
                                in_set[v] = True
                                added_any = True
                    
                    new_size = sum(in_set)
                    if new_size > best_size:
                        best_size = new_size
                        best_set = [v for v in range(n) if in_set[v]]
                        swapped = True
                        break
                    else:
                        in_set[rm] = True
            
            if not added and not swapped:
                break
        
        return best_set, best_size
    
    # Use graph approach to extend the solution
    random.seed(42)
    
    # Try to find larger sets using the conflict graph
    for target_k in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 20:
            break
        if target_k <= best_size:
            continue
        
        # Candidates: all numbers not in prescribed
        candidates = [x for x in range(1, N + 1) if x not in prescribed]
        n_cand = len(candidates)
        
        # Build conflict graph
        adj = build_conflict_graph(prescribed, candidates)
        
        # Find independent set of size target_k - len(prescribed)
        needed = target_k - len(prescribed)
        
        # Multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            # Get a random ordering
            order = list(range(n_cand))
            random.shuffle(order)
            
            # Greedy independent set
            indep = greedy_independent_set(adj, n_cand, order)
            
            if len(indep) >= needed:
                # Take first 'needed' elements
                chosen = [candidates[i] for i in indep[:needed]]
                sol = sorted(prescribed + chosen)
                
                if is_valid_sidon(sol):
                    if target_k > best_size:
                        best_size = target_k
                        best_a = sol
                    found = True
                    break
            
            # Local search to improve
            if len(indep) > 0:
                improved, imp_size = local_search_independent_set(adj, n_cand, indep, max_iters=2000)
                if imp_size >= needed:
                    chosen = [candidates[i] for i in improved[:needed]]
                    sol = sorted(prescribed + chosen)
                    
                    if is_valid_sidon(sol):
                        if target_k > best_size:
                            best_size = target_k
                            best_a = sol
                        found = True
                        break
        
        if found:
            break
    
    # Verify and return
    if is_valid_sidon(best_a):
        return sorted(best_a)
    if is_valid_sidon(a):
        return sorted(a)
    return sorted(prescribed)
# EVOLVE-BLOCK-END
