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
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            tree_data.append({'edges': edges, 'adj': adj, 'size': k})
            
    # Order: Largest to Smallest
    order = list(range(19, -1, -1))
    best_score = 0
    best_mappings = [None] * 20
    
    def embed_tree_csp_fc(tree_idx, used_edges, rng, is_first_tree=False):
        """
        CSP with Forward Checking.
        is_first_tree: If True, fix vertex 0 to host 0 for symmetry breaking.
        """
        k = tree_data[tree_idx]['size']
        if k == 1:
            # For k=1, just pick any available vertex
            # Since we are packing edges, vertex availability is less critical unless we run out of vertices?
            # K20 has 20 vertices. Trees sum to 1+2+...+20 = 210 vertices?
            # Wait. The trees are packed into K20.
            # The vertices of the trees are mapped to the 20 host vertices.
            # The mapping must be injective for each tree.
            # But different trees can share host vertices?
            # "Embedded trees must be edge-disjoint in K20."
            # It does NOT say vertex-disjoint.
            # So a host vertex can be used by multiple trees.
            # So for k=1, we can just pick 0.
            return [0], used_edges
            
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        return [u, v], used_edges | {e_key}
            return None, None
            
        edges = tree_data[tree_idx]['edges']
        adj = tree_data[tree_idx]['adj']
        
        # Initial domains
        # Domains are sets of possible host vertices for each tree vertex
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
            # Check if any neighbor of root has 0 removed causing empty domain?
            # No, they just can't be 0.
            # Also, edges from root (host 0) to neighbors must be free.
            # We handle this in the initial assignment step.

        def backtrack(assigned, domains):
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
            
            # Tie-breaker: Higher degree in tree
            max_deg = -1
            for v in unassigned:
                size = len(domains[v])
                deg = len(adj[v])
                if size < min_size or (size == min_size and deg > max_deg):
                    min_size = size
                    best_v = v
                    max_deg = deg
                    if size == 0:
                        break
            
            if best_v == -1 or min_size == 0:
                return None
                
            # Value Ordering: 
            # 1. Sort by host vertex index for determinism
            # 2. Could also sort by "least constraining" but index is faster
            values = sorted(domains[best_v])
            
            for h in values:
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                
                # Forward Check
                new_domains = {}
                for v in range(k):
                    if v in new_assigned:
                        continue
                    new_dom = set(domains[v])
                    
                    # 1. Injectivity: Remove h
                    new_dom.discard(h)
                    
                    # 2. Edge Constraint: If v is neighbor of best_v
                    if best_v in adj[v]:
                        to_remove = set()
                        for h_prime in new_dom:
                            e_key = (min(h, h_prime), max(h, h_prime))
                            if e_key in used_edges:
                                to_remove.add(h_prime)
                        new_dom -= to_remove
                        
                    if not new_dom:
                        new_domains = None
                        break
                    new_domains[v] = new_dom
                
                if new_domains is None:
                    continue
                    
                result = backtrack(new_assigned, new_domains)
                if result is not None:
                    return result
                    
            return None
        
        # Start search
        # If symmetry breaking is active, start with root assigned
        if is_first_tree:
            initial_assigned = {0: 0}
            init_domains = {}
            valid = True
            for v in range(1, k):
                dom = set(range(1, 20)) # 0 is already excluded
                # Check edge constraint with root (host 0)
                if 0 in adj[v]:
                    to_remove = set()
                    for h_prime in dom:
                        e_key = (0, h_prime)
                        if e_key in used_edges:
                            to_remove.add(h_prime)
                    dom -= to_remove
                if not dom:
                    valid = False
                    break
                init_domains[v] = dom
            
            if valid:
                result = backtrack(initial_assigned, init_domains)
                if result is not None:
                    mapping = [0] * k
                    for tv, hv in result.items():
                        mapping[tv] = hv
                    new_used_edges = set(used_edges)
                    for u, v in edges:
                        hu = mapping[u]
                        hv = mapping[v]
                        e_key = (min(hu, hv), max(hu, hv))
                        new_used_edges.add(e_key)
                    return mapping, new_used_edges
            # If symmetry breaking failed, fall back to general search? 
            # It shouldn't fail if a solution exists.
            
        # General search (if not first tree or symmetry breaking failed)
        # For non-first trees, we don't fix root.
        # We can still use MRV.
        # To speed up, we can try to fix root to a random available vertex?
        # No, just let MRV handle it.
        
        # Optimization: Try assigning the highest degree vertex first manually?
        # MRV will pick it if domains are equal.
        
        result = backtrack({}, initial_domains)
        if result is not None:
            mapping = [0] * k
            for tv, hv in result.items():
                mapping[tv] = hv
            new_used_edges = set(used_edges)
            for u, v in edges:
                hu = mapping[u]
                hv = mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                new_used_edges.add(e_key)
            return mapping, new_used_edges
            
        return None, None

    # Main Loop
    # Try deterministic first
    rng = Random(611)
    current_used_edges = set()
    current_mappings = [None] * 20
    success = True
    
    for i, tree_idx in enumerate(order):
        if time.time() > deadline:
            success = False
            break
            
        is_first = (i == 0)
        mapping, new_used_edges = embed_tree_csp_fc(tree_idx, current_used_edges, rng, is_first)
        if mapping is None:
            success = False
            break
        current_mappings[tree_idx] = mapping
        current_used_edges = new_used_edges
        
    if success:
        score = 0
        for i in range(20):
            if current_mappings[i] is not None:
                score += len(tree_data[i]['edges'])
                
        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in current_mappings]
            
        if best_score == 190:
            # Found perfect packing
            final_result = []
            for i in range(20):
                if best_mappings[i] is None:
                    final_result.append(None)
                else:
                    final_result.append([int(x) for x in best_mappings[i]])
            return final_result

    # If deterministic failed, try restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 6789)
        current_used_edges = set()
        current_mappings = [None] * 20
        success = True
        
        for i, tree_idx in enumerate(order):
            if time.time() > deadline:
                success = False
                break
                
            is_first = (i == 0)
            # For restarts, we can still use symmetry breaking for the first tree
            # But maybe randomize the fixed host? No, 0 is fine.
            mapping, new_used_edges = embed_tree_csp_fc(tree_idx, current_used_edges, rng, is_first)
            if mapping is None:
                success = False
                break
            current_mappings[tree_idx] = mapping
            current_used_edges = new_used_edges
            
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
