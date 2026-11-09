import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety (evaluator kills at 90s)
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    # Build conflict graph using bitmasks for speed
    n = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # conflicts_mask[i] is a bitmask where bit j is set if node i conflicts with node j
    conflicts_mask = [0] * n
    for i in range(n):
        vi = valid_nodes[i]
        for j in range(i + 1, n):
            vj = valid_nodes[j]
            if vi + vj in squares:
                conflicts_mask[i] |= (1 << j)
                conflicts_mask[j] |= (1 << i)
    
    # Tomita's maximum clique algorithm (used for MIS lower bound on complement)
    # We find MIS in conflict graph = find Max Clique in complement graph
    # But we can also use it directly: MIS in G = Max Clique in G_complement
    
    # For branch and bound on MIS, we need a lower bound on the size of MIS
    # We can get this by finding a large independent set greedily or via max clique on complement
    
    def max_clique_tomita(candidates, current_clique, best_clique, best_size_ref):
        """
        Tomita's algorithm for maximum clique.
        candidates: list of node indices that can be added to current_clique
        current_clique: list of nodes in current clique
        best_clique: list of the best clique found so far
        best_size_ref: list containing the size of best clique (mutable)
        """
        if not candidates:
            if len(current_clique) > best_size_ref[0]:
                best_size_ref[0] = len(current_clique)
                best_clique[:] = current_clique[:]
            return
        
        # Order candidates by degree in the candidate subgraph (descending)
        cand_set = set(candidates)
        degrees = []
        for c in candidates:
            deg = 0
            for c2 in candidates:
                if c != c2 and (conflicts_mask[c] >> c2) & 1:
                    deg += 1
            degrees.append((deg, c))
        degrees.sort(reverse=True)
        ordered_candidates = [c for _, c in degrees]
        
        for idx, v in enumerate(ordered_candidates):
            # Pruning: if current_clique + remaining candidates <= best, skip
            if len(current_clique) + (len(ordered_candidates) - idx) <= best_size_ref[0]:
                break
            
            # Check if v can be added (conflicts with all in current_clique)
            can_add = True
            for u in current_clique:
                if not ((conflicts_mask[v] >> u) & 1):
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # New candidates must conflict with v and be in remaining candidates
            new_candidates = []
            for c in ordered_candidates[idx + 1:]:
                if (conflicts_mask[v] >> c) & 1:
                    new_candidates.append(c)
            
            max_clique_tomita(new_candidates, current_clique + [v], best_clique, best_size_ref)
        
        return

    def greedy_independent_set(nodes, deadline):
        """Find a large independent set greedily using degree ordering."""
        chosen = []
        remaining = list(nodes)
        remaining.sort(key=lambda x: sum(1 for y in nodes if x != y and (conflicts_mask[x] >> y) & 1))
        
        for v in remaining:
            if time.time() > deadline:
                break
            # Check if v conflicts with any chosen
            independent = True
            for u in chosen:
                if (conflicts_mask[v] >> u) & 1:
                    independent = False
                    break
            if independent:
                chosen.append(v)
        
        return chosen

    def branch_and_bound_mis(nodes, best_size, best_set, deadline):
        """
        Branch and bound for Maximum Independent Set.
        nodes: list of candidate node indices
        best_size: current best MIS size
        best_set: current best MIS (list of indices)
        deadline: time limit
        """
        if not nodes:
            return best_size, best_set
        
        if time.time() > deadline:
            return best_size, best_set
        
        # Get a lower bound using greedy
        greedy_set = greedy_independent_set(nodes, deadline)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set[:]
        
        # If nodes size <= best_size, can't improve
        if len(nodes) <= best_size:
            return best_size, best_set
        
        # Order nodes by degree (ascending for MIS - prefer low degree first)
        nodes_sorted = sorted(nodes, key=lambda x: sum(1 for y in nodes if x != y and (conflicts_mask[x] >> y) & 1))
        
        # Branch on first node
        v = nodes_sorted[0]
        
        # Branch 1: Include v
        # New candidates are nodes that don't conflict with v
        new_candidates_include = [u for u in nodes[1:] if not ((conflicts_mask[v] >> u) & 1)]
        
        size_inc, set_inc = branch_and_bound_mis(new_candidates_include, best_size, best_set, deadline)
        best_size = max(best_size, size_inc)
        
        # Branch 2: Exclude v
        new_candidates_exclude = nodes[1:]
        size_exc, set_exc = branch_and_bound_mis(new_candidates_exclude, best_size, best_set, deadline)
        best_size = max(best_size, size_exc)
        
        return best_size, best_set

    def improved_branch_and_bound(nodes, best_size, best_set, deadline):
        """
        Improved B&B with better pruning and ordering.
        """
        if not nodes:
            return best_size, best_set
        
        if time.time() > deadline:
            return best_size, best_set
        
        # Upper bound: size of nodes
        if len(nodes) <= best_size:
            return best_size, best_set
        
        # Greedy lower bound
        greedy_set = greedy_independent_set(nodes, deadline)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set[:]
        
        if len(nodes) <= best_size:
            return best_size, best_set
        
        # Color-based upper bound (greedy coloring of the subgraph)
        # For MIS, we color the conflict graph; the number of colors is an upper bound on MIS
        # Actually for MIS, we want a lower bound on clique cover of complement... 
        # Simpler: use greedy coloring as upper bound on clique number of subgraph
        # For MIS in G, upper bound = chromatic number of G[nodes]
        
        # Greedy coloring
        node_set = set(nodes)
        colors = {}
        color_list = []
        for v in nodes:
            used_colors = set()
            for u in colors:
                if u in node_set and (conflicts_mask[v] >> u) & 1:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c >= len(color_list):
                color_list.append([])
            color_list[c].append(v)
        
        upper_bound = len(color_list)
        if upper_bound <= best_size:
            return best_size, best_set
        
        # Branch on node with highest degree in the subgraph
        v = max(nodes, key=lambda x: sum(1 for y in nodes if x != y and (conflicts_mask[x] >> y) & 1))
        
        # Branch 1: Include v
        new_candidates_include = [u for u in nodes if u != v and not ((conflicts_mask[v] >> u) & 1)]
        size_inc, set_inc = improved_branch_and_bound(new_candidates_include, best_size, best_set, deadline)
        best_size = max(best_size, size_inc)
        
        # Branch 2: Exclude v
        new_candidates_exclude = [u for u in nodes if u != v]
        size_exc, set_exc = improved_branch_and_bound(new_candidates_exclude, best_size, best_set, deadline)
        best_size = max(best_size, size_exc)
        
        return best_size, best_set

    # Start with greedy solution
    best_set = greedy_independent_set(list(range(n)), deadline)
    best_size = len(best_set)
    
    # Try multiple greedy orderings
    random.seed(42)
    for _ in range(20):
        shuffled = list(range(n))
        random.shuffle(shuffled)
        gs = greedy_independent_set(shuffled, deadline)
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs
    
    # Run branch and bound
    if time.time() < deadline - 5:
        best_size, best_set = improved_branch_and_bound(list(range(n)), best_size, best_set, deadline)
    
    # Convert back to original values
    result = [valid_nodes[i] for i in best_set]
    
    if not result:
        result = [valid_nodes[0]]
    
    return result
