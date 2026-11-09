"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Build compatibility: two nodes u,v are compatible if lcm(u,v) <= 512
    # lcm(u,v) = u*v // gcd(u,v)
    # Precompute adjacency as bitsets for speed
    n = N
    # adj[i] is a set of nodes compatible with node i (1-indexed, we'll use 0-indexed internally)
    # Use 0-indexed: node i corresponds to value i+1
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        vi = i + 1
        for j in range(i + 1, n):
            vj = j + 1
            g = gcd(vi, vj)
            lcm_val = vi * vj // g
            if lcm_val <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Also add self-compatibility (each node is compatible with itself for the clique)
    # Actually for clique finding, we just need the edges
    
    def time_check():
        return time.time() >= deadline
    
    # Strategy 1: Greedy construction
    def greedy_solve(order):
        """Build a clique greedily by processing nodes in given order."""
        clique = []
        compatible_with_all = set(range(n))
        for v in order:
            if v not in compatible_with_all:
                continue
            # Check if v is compatible with all in clique
            ok = True
            for u in clique:
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                clique.append(v)
                # Update compatible_with_all: must be compatible with v
                compatible_with_all &= adj[v]
                compatible_with_all.discard(v)
        return clique
    
    # Initial greedy: sort by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    order_by_degree = sorted(range(n), key=lambda x: -degrees[x])
    best_clique = greedy_solve(order_by_degree)
    best_size = len(best_clique)
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 2: Try multiple random orderings
    import random
    rng = random.Random(42)
    
    for trial in range(200):
        if time_check():
            break
        # Random shuffle with some structure
        order = list(range(n))
        rng.shuffle(order)
        # Sort by degree within random groups for some structure
        # Just use random order
        clique = greedy_solve(order)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 3: Local search - try to add nodes, swap nodes
    def is_valid_clique(clique):
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                if clique[j] not in adj[clique[i]]:
                    return False
        return True
    
    # Local search: try adding a node that's compatible with all
    def try_add_node(clique):
        """Try to add a node not in clique that's compatible with all in clique."""
        clique_set = set(clique)
        candidates = [v for v in range(n) if v not in clique_set]
        # Sort by how many clique members they're compatible with (descending)
        candidates.sort(key=lambda v: -sum(1 for u in clique if v in adj[u]))
        for v in candidates:
            ok = True
            for u in clique:
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                return clique + [v]
        return None
    
    # Local search: try swapping one node for another
    def local_search(clique):
        """Try to improve clique by swapping elements."""
        clique_set = set(clique)
        improved = True
        while improved:
            improved = False
            # Try adding
            new_clique = try_add_node(clique)
            if new_clique is not None:
                clique = new_clique
                clique_set = set(clique)
                improved = True
                continue
            # Try swapping: remove one, add two
            for i, u in enumerate(clique):
                if time_check():
                    return clique
                # Remove u, see if we can add two new nodes
                remaining = [clique[j] for j in range(len(clique)) if j != i]
                remaining_set = set(remaining)
                # Find nodes compatible with all in remaining but not u
                candidates = []
                for v in range(n):
                    if v in remaining_set:
                        continue
                    if v == u:
                        continue
                    ok = True
                    for w in remaining:
                        if v not in adj[w]:
                            ok = False
                            break
                    if ok:
                        candidates.append(v)
                # Sort by degree
                candidates.sort(key=lambda v: -len(adj[v]))
                # Try to find two that are compatible with each other
                for idx_a, va in enumerate(candidates[:50]):
                    if time_check():
                        return clique
                    for vb in candidates[idx_a+1:idx_a+50]:
                        if va in adj[vb]:
                            # Check both are compatible with all in remaining
                            # Already checked above
                            new_clique = remaining + [va, vb]
                            if len(new_clique) > len(clique):
                                clique = new_clique
                                clique_set = set(clique)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return clique
    
    # Run local search from best clique
    best_clique = local_search(list(best_clique))
    best_size = len(best_clique)
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 4: Branch and Bound for maximum clique
    # Use the Tomita algorithm (branch and bound with coloring)
    
    def max_clique_bnb(init_clique):
        """Find maximum clique using branch and bound, starting from init_clique."""
        best = list(init_clique)
        best_size = len(best)
        
        def color_sort(candidates):
            """Color the candidates greedily for upper bound."""
            candidates = list(candidates)
            colors = []
            color_classes = []
            remaining = set(candidates)
            color = 0
            while remaining:
                color += 1
                color_class = []
                new_remaining = set()
                for v in sorted(remaining):
                    # v can be in this color class if not adjacent to any in color_class
                    can_add = True
                    for u in color_class:
                        if v in adj[u]:
                            can_add = False
                            break
                    if can_add:
                        color_class.append(v)
                    else:
                        new_remaining.add(v)
                colors.extend([color] * len(color_class))
                color_classes.append(color_class)
                remaining = new_remaining
            # Return ordered by color (highest color first for pruning)
            order = sorted(range(len(candidates)), key=lambda i: -colors[i])
            return [candidates[i] for i in order], colors
    
        def bnb(candidates, current_clique):
            nonlocal best, best_size
            if time_check():
                return
            if not candidates:
                if len(current_clique) > best_size:
                    best = list(current_clique)
                    best_size = len(best)
                return
            
            ordered, colors = color_sort(candidates)
            # Upper bound: len(current_clique) + max_color
            max_color = max(colors) if colors else 0
            if len(current_clique) + max_color <= best_size:
                return
            
            for i in range(len(ordered)):
                if time_check():
                    return
                v = ordered[i]
                # Upper bound using color of v
                if len(current_clique) + colors[i] <= best_size:
                    continue
                # Try adding v
                new_candidates = [u for u in ordered[i+1:] if u in adj[v]]
                current_clique.append(v)
                bnb(new_candidates, current_clique)
                current_clique.pop()
        
        # Start B&B
        # Candidates are all nodes compatible with all in init_clique
        init_set = set(init_clique)
        candidates = []
        for v in range(n):
            if v in init_set:
                continue
            ok = True
            for u in init_clique:
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        
        bnb(candidates, list(init_clique))
        return best
    
    # Run B&B from best clique
    best_clique = max_clique_bnb(best_clique)
    best_size = len(best_clique)
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 5: Try B&B from scratch (full maximum clique)
    def max_clique_full():
        """Find maximum clique from scratch using B&B."""
        best = []
        best_size = 0
        
        def color_sort(candidates):
            candidates = list(candidates)
            colors = []
            remaining = set(candidates)
            color = 0
            while remaining:
                color += 1
                color_class = []
                new_remaining = set()
                for v in sorted(remaining):
                    can_add = True
                    for u in color_class:
                        if v in adj[u]:
                            can_add = False
                            break
                    if can_add:
                        color_class.append(v)
                    else:
                        new_remaining.add(v)
                colors.extend([color] * len(color_class))
                remaining = new_remaining
            order = sorted(range(len(candidates)), key=lambda i: -colors[i])
            return [candidates[i] for i in order], colors
        
        def bnb(candidates, current_clique):
            nonlocal best, best_size
            if time_check():
                return
            if not candidates:
                if len(current_clique) > best_size:
                    best = list(current_clique)
                    best_size = len(best)
                return
            
            ordered, colors = color_sort(candidates)
            max_color = max(colors) if colors else 0
            if len(current_clique) + max_color <= best_size:
                return
            
            for i in range(len(ordered)):
                if time_check():
                    return
                v = ordered[i]
                if len(current_clique) + colors[i] <= best_size:
                    continue
                new_candidates = [u for u in ordered[i+1:] if u in adj[v]]
                current_clique.append(v)
                bnb(new_candidates, current_clique)
                current_clique.pop()
        
        # Start with all nodes
        bnb(list(range(n)), [])
        return best
    
    full_clique = max_clique_full()
    if len(full_clique) > best_size:
        best_clique = full_clique
        best_size = len(full_clique)
    
    return [i + 1 for i in best_clique]
# EVOLVE-BLOCK-END
