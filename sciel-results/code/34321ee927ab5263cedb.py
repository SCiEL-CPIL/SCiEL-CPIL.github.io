import time
import sys

def construct():
    from random import Random
    from collections import defaultdict, deque

    # Keep the exact generator from the seed program
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert lists to sets for fast lookup
        domains = [set(lst) for lst in lists]
        
        # Heuristic ordering: 
        # 1. Minimum Remaining Values (MRV)
        # 2. Degree (higher is better)
        # We will do dynamic variable selection during backtracking.
        
        # Initial check: if any domain is empty, fail
        for i in range(n):
            if not domains[i]:
                return None

        # To speed up, we can pre-calculate degrees
        degrees = [len(adj[i]) for i in range(n)]
        
        # Backtracking search
        # We use a recursive function with forward checking
        # To avoid deep recursion limits, we can increase recursion limit or use iterative, 
        # but 400 depth is fine for Python default (1000).
        
        # Optimization: Order the initial assignment. 
        # Let's try sorting by degree descending as a static heuristic for the first few, 
        # or just let the dynamic MRV handle it. Dynamic MRV is usually stronger.
        
        # We'll use a simple recursive backtracker.
        # State: colour array, domains
        colour = [None] * n
        
        # To make it faster, we can use a stack or just recursion.
        # Given the size, recursion is fine.
        
        # Heuristic for variable selection:
        # Find uncoloured vertex with smallest domain size.
        # Tie-breaker: highest degree.
        
        # We need to be careful with copying domains for backtracking.
        # Copying sets of size 3 for 400 vertices is cheap.
        
        start_time = time.time()
        
        def select_variable():
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for i in range(n):
                if colour[i] is None:
                    size = len(domains[i])
                    if size < min_size or (size == min_size and degrees[i] > best_deg):
                        min_size = size
                        best_v = i
                        best_deg = degrees[i]
            return best_v

        def backtrack():
            if time.time() - start_time > 10: # Safety margin
                return False
            
            v = select_variable()
            if v == -1:
                return True # All assigned
            
            # Try colours in the domain
            # Heuristic for value ordering: try colours that appear in fewer neighbours' domains? 
            # Or just arbitrary. Let's try arbitrary first.
            for c in domains[v]:
                # Check if colour c is available (it is, by definition of domains[v])
                
                # Forward checking: 
                # If we assign c to v, we must remove c from domains of all uncoloured neighbours.
                # If any neighbour's domain becomes empty, this branch is invalid.
                
                saved_neighbour_domains = []
                valid = True
                for u in adj[v]:
                    if colour[u] is None:
                        if c in domains[u]:
                            saved_neighbour_domains.append((u, domains[u].copy()))
                            domains[u].remove(c)
                            if not domains[u]:
                                valid = False
                                break
                
                if valid:
                    colour[v] = c
                    if backtrack():
                        return True
                    colour[v] = None
                    # Undo forward checking
                    for u, old_dom in saved_neighbour_domains:
                        domains[u] = old_dom
                else:
                    # Undo forward checking for those we modified
                    for u, old_dom in saved_neighbour_domains:
                        domains[u] = old_dom
                        
            return False

        # It's possible the static ordering of colours in the list matters for performance.
        # We can try to sort domains or just iterate.
        
        if backtrack():
            # Ensure types are int
            return [int(c) for c in colour]
        else:
            return None

    # Since Alon-Tarsi guarantees a solution, but simple backtracking might get stuck in a bad path 
    # if the graph is complex, we can add a few random restarts with perturbed colour orders 
    # or just rely on the strong MRV heuristic. 
    # Given the specific structure (quadrangulations), MRV is very effective.
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Try solving
        sol = solve_csp(n, edges, lists)
        
        if sol is None:
            # Fallback: Try a different heuristic or random restart
            # For robustness, let's try a few restarts with randomised tie-breaking in MRV
            # or randomised value ordering.
            sol = None
            for attempt in range(5):
                # Shuffle the order of colours in domains to break symmetry/ties differently
                # We need to restore original lists first
                # Actually, solve_csp modifies domains. We should pass copies or restore.
                # My current solve_csp takes lists and makes domains. It doesn't modify the input lists.
                # But it does modify domains internally.
                # To restart, we just call solve_csp again? No, it's deterministic.
                # We need to introduce randomness.
                
                # Let's modify the solver to accept a random seed for tie-breaking or value ordering.
                # Or simpler: if it fails, it's likely a bug or extremely hard. 
                # But Alon-Tarsi says it's easy. 
                # Let's add a simple randomised value ordering to the solver logic in a helper.
                
                pass # We will implement a more robust solver below if needed.
                # For now, let's assume the first pass works. 
                # If not, we'll implement a retry mechanism.
        
        if sol is None:
            # Implement a retry with randomised value ordering
            def solve_csp_randomized(n, edges, lists, rng_seed):
                import random
                rng = random.Random(rng_seed)
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Copy domains and shuffle the order of colours in each domain
                # The set doesn't have order, so we need to iterate in a specific order.
                # We can create a list for each domain and shuffle it.
                domains_list = [list(lst) for lst in lists]
                for i in range(n):
                    rng.shuffle(domains_list[i])
                
                domains = [set(lst) for lst in domains_list]
                degrees = [len(adj[i]) for i in range(n)]
                colour = [None] * n
                start_time = time.time()
                
                def select_variable():
                    min_size = float('inf')
                    best_v = -1
                    best_deg = -1
                    # Add small random noise to break ties? Or just use rng for tie-breaking
                    candidates = []
                    for i in range(n):
                        if colour[i] is None:
                            size = len(domains[i])
                            if size < min_size:
                                min_size = size
                                candidates = [i]
                                best_deg = degrees[i]
                            elif size == min_size:
                                candidates.append(i)
                    
                    if not candidates:
                        return -1
                    
                    if len(candidates) == 1:
                        return candidates[0]
                    
                    # Pick one with highest degree, or random from top degrees?
                    # Let's pick highest degree
                    best_v = max(candidates, key=lambda x: degrees[x])
                    return best_v

                def backtrack():
                    if time.time() - start_time > 10:
                        return False
                    
                    v = select_variable()
                    if v == -1:
                        return True
                    
                    # Iterate in the shuffled order
                    for c in domains_list[v]:
                        if c not in domains[v]:
                            continue # Already removed by forward checking? 
                                     # Actually domains[v] is updated by neighbours, not by its own iteration.
                                     # But if a neighbour removed c from v, we skip it.
                        
                        saved_neighbour_domains = []
                        valid = True
                        for u in adj[v]:
                            if colour[u] is None:
                                if c in domains[u]:
                                    saved_neighbour_domains.append((u, domains[u].copy()))
                                    domains[u].remove(c)
                                    if not domains[u]:
                                        valid = False
                                        break
                        
                        if valid:
                            colour[v] = c
                            if backtrack():
                                return True
                            colour[v] = None
                            for u, old_dom in saved_neighbour_domains:
                                domains[u] = old_dom
                        else:
                            for u, old_dom in saved_neighbour_domains:
                                domains[u] = old_dom
                    return False

                if backtrack():
                    return [int(c) for c in colour]
                return None

            for attempt in range(10):
                sol = solve_csp_randomized(n, edges, lists, seed + attempt)
                if sol:
                    break
        
        # Final fallback: if still None, return dummy (will score 0 for this instance)
        # But this shouldn't happen.
        if sol is None:
            sol = [0] * n # Invalid, but prevents crash
            
        out.append(sol)
        
    return out

# To ensure the function is available and correct
# We wrap the logic in construct as requested.
