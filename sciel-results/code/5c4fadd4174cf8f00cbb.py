# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    trees = instance_trees(20, 611)
    num_trees = 20
    
    tree_data = []
    for i in range(num_trees):
        k = i + 1
        if k == 1:
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1, 'deg': [0]})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2, 'deg': [1, 1]})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            deg = [0] * k
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
            tree_data.append({'edges': edges, 'adj': adj, 'size': k, 'deg': deg})
            
    # Order: Largest to Smallest
    order = list(range(19, -1, -1))
    best_score = 0
    best_mappings = [None] * 20
    best_used_edges = set()
    best_used_degrees = [0] * 20

    def embed_tree_csp_fc(tree_idx, used_edges, used_degrees, is_first_tree=False):
        """
        CSP with Forward Checking and Degree Constraints.
        """
        k = tree_data[tree_idx]['size']
        if k == 1:
            # For k=1, no edges, degree 0.
            # Just pick any vertex. To be safe, pick 0? 
            # But 0 might be "used up" in terms of degree? No, degree 0 adds nothing.
            # However, we must return a distinct int.
            # Since vertices can be shared, we can return [0].
            # But wait, if we return [0], and another tree also returns [0], is that allowed?
            # "list of k distinct int host vertices" -> distinct WITHIN the tree.
            # "Embedded trees must be edge-disjoint" -> edges disjoint BETWEEN trees.
            # So yes, [0] is fine for T1.
            return [0], used_edges, used_degrees
            
        if k == 2:
            edges = tree_data[tree_idx]['edges']
            # Find an available edge
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        # Check degree constraint
                        # deg is [1, 1]
                        if used_degrees[u] + 1 <= 19 and used_degrees[v] + 1 <= 19:
                            new_used_edges = used_edges | {e_key}
                            new_used_degrees = used_degrees[:]
                            new_used_degrees[u] += 1
                            new_used_degrees[v] += 1
                            return [u, v], new_used_edges, new_used_degrees
            return None, None, None
            
        edges = tree_data[tree_idx]['edges']
        adj = tree_data[tree_idx]['adj']
        deg = tree_data[tree_idx]['deg']
        
        # Initial domains
        initial_domains = {v: set(range(20)) for v in range(k)}
        
        # Symmetry Breaking: If this is the first (largest) tree, fix tree vertex 0 to host 0
        fixed_root = None
        fixed_host = None
        if is_first_tree:
            fixed_root = 0
            fixed_host = 0
            # Remove host 0 from all other domains
            for v in range(1, k):
                initial_domains[v].discard(0)
            # Check degree constraint for root
            if used_degrees[0] + deg[0] > 19:
                return None, None, None

        def backtrack(assigned, domains, cur_used_degrees):
            if len(assigned) == k:
                # All assigned. Verify edges
                for u, v in edges:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()
            
            # MRV: Pick unassigned variable with smallest domain
            unassigned = [v for v in range(k) if v not in assigned]
            if not unassigned:
                return None
                
            min_size = 21
            best_v = -1
            max_deg = -1
            for v in unassigned:
                size = len(domains[v])
                deg_v = deg[v]
                # Tie-breaker: Higher degree in tree first (more constraining)
                if size < min_size or (size == min_size and deg_v > max_deg):
                    min_size = size
                    best_v = v
                    max_deg = deg_v
                    if size == 0:
                        break
            
            if best_v == -1 or min_size == 0:
                return None
                
            # Value Ordering: Sort by host vertex index
            values = sorted(domains[best_v])
            
            for h in values:
                # Check Degree Constraint for this assignment
                if cur_used_degrees[h] + deg[best_v] > 19:
                    continue
                    
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                
                # Forward Check
                new_domains = {}
                valid = True
                new_cur_used_degrees = cur_used_degrees[:]
                new_cur_used_degrees[h] += deg[best_v]
                
                for v in range(k):
                    if v in new_assigned:
                        continue
                    new_dom = set(domains[v])
                    
                    # 1. Injectivity: Remove h
                    new_dom.discard(h)
                    
                    # 2. Degree Constraint: Remove hosts where degree would exceed 19
                    # 3. Edge Constraint: If v is neighbor of best_v
                    is_neighbor = best_v in adj[v]
                    
                    to_remove = set()
                    for h_prime in new_dom:
                        # Degree check
                        if new_cur_used_degrees[h_prime] + deg[v] > 19:
                            to_remove.add(h_prime)
                            continue
                            
                        # Edge check
                        if is_neighbor:
                            e_key = (min(h, h_prime), max(h, h_prime))
                            if e_key in used_edges:
                                to_remove.add(h_prime)
                        
                    new_dom -= to_remove
                        
                    if not new_dom:
                        valid = False
                        break
                    new_domains[v] = new_dom
                
                if not valid:
                    continue
                    
                result = backtrack(new_assigned, new_domains, new_cur_used_degrees)
                if result is not None:
                    return result
                    
            return None
        
        # Start search
        if is_first_tree:
            initial_assigned = {0: 0}
            init_domains = {}
            valid = True
            init_used_degrees = used_degrees[:]
            init_used_degrees[0] += deg[0]
            
            for v in range(1, k):
                dom = set(range(1, 20))
                # Check edge constraint with root (host 0)
                if 0 in adj[v]:
                    to_remove = set()
                    for h_prime in dom:
                        e_key = (0, h_prime)
                        if e_key in used_edges:
                            to_remove.add(h_prime)
                    dom -= to_remove
                # Check degree constraint
                to_remove_deg = set()
                for h_prime in dom:
                    if init_used_degrees[h_prime] + deg[v] > 19:
                        to_remove_deg.add(h_prime)
                dom -= to_remove_deg
                
                if not dom:
                    valid = False
                    break
                init_domains[v] = dom
            
            if valid:
                result = backtrack(initial_assigned, init_domains, init_used_degrees)
                if result is not None:
                    mapping = [0] * k
                    for tv, hv in result.items():
                        mapping[tv] = hv
                    new_used_edges = set(used_edges)
                    new_used_degrees = init_used_degrees[:]
                    for u, v in edges:
                        hu = mapping[u]
                        hv = mapping[v]
                        e_key = (min(hu, hv), max(hu, hv))
                        new_used_edges.add(e_key)
                    # Note: new_used_degrees already updated in backtrack? 
                    # No, backtrack returns just the mapping. We need to re-calc or pass it out.
                    # In the is_first_tree block, we initialized init_used_degrees.
                    # The backtrack function modifies a *copy* in its recursion.
                    # We need the final used_degrees.
                    # Let's re-calculate used_degrees from the mapping to be safe.
                    final_used_degrees = used_degrees[:]
                    for tv, hv in result.items():
                        final_used_degrees[hv] += deg[tv]
                    return mapping, new_used_edges, final_used_degrees

        # General search
        result = backtrack({}, initial_domains, used_degrees)
        if result is not None:
            mapping = [0] * k
            for tv, hv in result.items():
                mapping[tv] = hv
            new_used_edges = set(used_edges)
            new_used_degrees = used_degrees[:]
            for u, v in edges:
                hu = mapping[u]
                hv = mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                new_used_edges.add(e_key)
            for tv, hv in result.items():
                new_used_degrees[hv] += deg[tv]
            return mapping, new_used_edges, new_used_degrees
            
        return None, None, None

    # Main Loop
    current_used_edges = set()
    current_used_degrees = [0] * 20
    current_mappings = [None] * 20
    success = True
    
    # Deterministic run
    for i, tree_idx in enumerate(order):
        if time.time() > deadline:
            success = False
            break
            
        is_first = (i == 0)
        mapping, new_used_edges, new_used_degrees = embed_tree_csp_fc(tree_idx, current_used_edges, current_used_degrees, is_first)
        if mapping is None:
            success = False
            break
        current_mappings[tree_idx] = mapping
        current_used_edges = new_used_edges
        current_used_degrees = new_used_degrees
        
    if success:
        score = 0
        for i in range(20):
            if current_mappings[i] is not None:
                score += len(tree_data[i]['edges'])
                
        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in current_mappings]
            best_used_edges = current_used_edges
            best_used_degrees = current_used_degrees[:]
            
        if best_score == 190:
            final_result = []
            for i in range(20):
                if best_mappings[i] is None:
                    final_result.append(None)
                else:
                    final_result.append([int(x) for x in best_mappings[i]])
            return final_result

    # Random Restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 6789)
        current_used_edges = set()
        current_used_degrees = [0] * 20
        current_mappings = [None] * 20
        success = True
        
        # Slight variation: Shuffle the order of smaller trees? 
        # Or just rely on the CSP's internal randomness? 
        # The CSP is deterministic given the domains. 
        # To introduce randomness, we can shuffle the order of values in `values = sorted(...)`?
        # Or shuffle the order of trees?
        # Shuffling trees might break the "largest first" heuristic which is strong.
        # Let's stick to deterministic CSP but maybe try different symmetry breaking hosts?
        # For is_first_tree, we fixed host 0. We can try host 0..19 in different restarts?
        # But symmetry breaking is powerful.
        
        # Let's just run the same deterministic logic. 
        # If it fails, it fails.
        # To get different results, we need to randomize the CSP search.
        # We can pass a `rng` to `embed_tree_csp_fc` and shuffle `values`.
        
        # Let's modify embed_tree_csp_fc to accept rng and use it for value ordering.
        # But we can't easily modify the nested function signature without refactoring.
        # Instead, let's just accept that if deterministic fails, we might need a different strategy.
        # However, the degree constraint is so strong that it might solve it deterministically.
        
        # If we want to add randomness, we can shuffle the `order` of trees?
        # No, keep largest first.
        
        # Let's try a few different fixed hosts for the first tree in restarts.
        fixed_host = restart % 20
        
        # We need to pass fixed_host to the embed function or handle it here.
        # The current embed_tree_csp_fc hardcodes host 0 for is_first_tree.
        # Let's refactor slightly to allow this.
        
        # For now, let's just run the deterministic one multiple times? No, it's deterministic.
        # We need to introduce randomness.
        
        # Let's create a wrapper that shuffles the value ordering in the CSP.
        # This requires modifying the CSP function.
        
        pass # Placeholder for logic below

    # To properly implement random restarts, we should modify embed_tree_csp_fc to take a rng.
    # Let's rewrite the main loop to use a version that supports rng.
    
    # Refactored Embed Function with RNG
    def embed_tree_csp_fc_random(tree_idx, used_edges, used_degrees, is_first_tree, rng, fixed_host=0):
        k = tree_data[tree_idx]['size']
        if k == 1:
            return [0], used_edges, used_degrees
            
        if k == 2:
            edges = tree_data[tree_idx]['edges']
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        if used_degrees[u] + 1 <= 19 and used_degrees[v] + 1 <= 19:
                            new_used_edges = used_edges | {e_key}
                            new_used_degrees = used_degrees[:]
                            new_used_degrees[u] += 1
                            new_used_degrees[v] += 1
                            return [u, v], new_used_edges, new_used_degrees
            return None, None, None
            
        edges = tree_data[tree_idx]['edges']
        adj = tree_data[tree_idx]['adj']
        deg = tree_data[tree_idx]['deg']
        
        initial_domains = {v: set(range(20)) for v in range(k)}
        
        if is_first_tree:
            # Remove fixed_host from other domains
            for v in range(1, k):
                initial_domains[v].discard(fixed_host)
            if used_degrees[fixed_host] + deg[0] > 19:
                return None, None, None

        def backtrack(assigned, domains, cur_used_degrees):
            if len(assigned) == k:
                for u, v in edges:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()
            
            unassigned = [v for v in range(k) if v not in assigned]
            if not unassigned:
                return None
                
            min_size = 21
            best_v = -1
            max_deg = -1
            for v in unassigned:
                size = len(domains[v])
                deg_v = deg[v]
                if size < min_size or (size == min_size and deg_v > max_deg):
                    min_size = size
                    best_v = v
                    max_deg = deg_v
                    if size == 0:
                        break
            
            if best_v == -1 or min_size == 0:
                return None
                
            values = list(domains[best_v])
            rng.shuffle(values)
            
            for h in values:
                if cur_used_degrees[h] + deg[best_v] > 19:
                    continue
                    
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                
                new_domains = {}
                valid = True
                new_cur_used_degrees = cur_used_degrees[:]
                new_cur_used_degrees[h] += deg[best_v]
                
                for v in range(k):
                    if v in new_assigned:
                        continue
                    new_dom = set(domains[v])
                    new_dom.discard(h)
                    
                    is_neighbor = best_v in adj[v]
                    to_remove = set()
                    for h_prime in new_dom:
                        if new_cur_used_degrees[h_prime] + deg[v] > 19:
                            to_remove.add(h_prime)
                            continue
                        if is_neighbor:
                            e_key = (min(h, h_prime), max(h, h_prime))
                            if e_key in used_edges:
                                to_remove.add(h_prime)
                    new_dom -= to_remove
                        
                    if not new_dom:
                        valid = False
                        break
                    new_domains[v] = new_dom
                
                if not valid:
                    continue
                    
                result = backtrack(new_assigned, new_domains, new_cur_used_degrees)
                if result is not None:
                    return result
                    
            return None
        
        if is_first_tree:
            initial_assigned = {0: fixed_host}
            init_domains = {}
            valid = True
            init_used_degrees = used_degrees[:]
            init_used_degrees[fixed_host] += deg[0]
            
            for v in range(1, k):
                dom = set(range(20)) - {fixed_host}
                if 0 in adj[v]:
                    to_remove = set()
                    for h_prime in dom:
                        e_key = (min(fixed_host, h_prime), max(fixed_host, h_prime))
                        if e_key in used_edges:
                            to_remove.add(h_prime)
                    dom -= to_remove
                to_remove_deg = set()
                for h_prime in dom:
                    if init_used_degrees[h_prime] + deg[v] > 19:
                        to_remove_deg.add(h_prime)
                dom -= to_remove_deg
                
                if not dom:
                    valid = False
                    break
                init_domains[v] = dom
            
            if valid:
                result = backtrack(initial_assigned, init_domains, init_used_degrees)
                if result is not None:
                    mapping = [0] * k
                    for tv, hv in result.items():
                        mapping[tv] = hv
                    new_used_edges = set(used_edges)
                    final_used_degrees = used_degrees[:]
                    for u, v in edges:
                        hu = mapping[u]
                        hv = mapping[v]
                        e_key = (min(hu, hv), max(hu, hv))
                        new_used_edges.add(e_key)
                    for tv, hv in result.items():
                        final_used_degrees[hv] += deg[tv]
                    return mapping, new_used_edges, final_used_degrees

        result = backtrack({}, initial_domains, used_degrees)
        if result is not None:
            mapping = [0] * k
            for tv, hv in result.items():
                mapping[tv] = hv
            new_used_edges = set(used_edges)
            new_used_degrees = used_degrees[:]
            for u, v in edges:
                hu = mapping[u]
                hv = mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                new_used_edges.add(e_key)
            for tv, hv in result.items():
                new_used_degrees[hv] += deg[tv]
            return mapping, new_used_edges, new_used_degrees
            
        return None, None, None

    # Run Random Restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 6789)
        current_used_edges = set()
        current_used_degrees = [0] * 20
        current_mappings = [None] * 20
        success = True
        
        fixed_host = restart % 20
        
        for i, tree_idx in enumerate(order):
            if time.time() > deadline:
                success = False
                break
                
            is_first = (i == 0)
            mapping, new_used_edges, new_used_degrees = embed_tree_csp_fc_random(tree_idx, current_used_edges, current_used_degrees, is_first, rng, fixed_host)
            if mapping is None:
                success = False
                break
            current_mappings[tree_idx] = mapping
            current_used_edges = new_used_edges
            current_used_degrees = new_used_degrees
            
        if success:
            score = 0
            for i in range(20):
                if current_mappings[i] is not None:
                    score += len(tree_data[i]['edges'])
                    
            if score > best_score:
                best_score = score
                best_mappings = [m[:] if m else None for m in current_mappings]
                
            if best_score == 190:
                break
                
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
