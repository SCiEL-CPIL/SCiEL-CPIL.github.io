"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    num_trees = 20
    
    # Pre-compute tree adjacency and edge lists
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
            
    # Strategy: CSP Backtracking with Forward Checking and MRV
    # We treat the problem as packing trees one by one.
    # To maximize success, we pack the largest trees first (T20, T19, ...).
    # For each tree, we use a CSP solver to find a valid embedding.
    
    # Order of trees to embed: descending by size
    order = list(range(19, -1, -1))
    
    best_score = 0
    best_mappings = [None] * 20
    
    # Let's implement a robust CSP solver for embedding a single tree given current used edges.
    
    def embed_tree_csp_random(tree_idx, used_edges, rng):
        k = tree_data[tree_idx]['size']
        if k == 1:
            for v in range(20):
                return [v], used_edges
            return None, None
        
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        return [u, v], used_edges | {e_key}
            return None, None
            
        degrees = [len(tree_data[tree_idx]['adj'][i]) for i in range(k)]
        root = max(range(k), key=lambda i: degrees[i])
        
        def backtrack(assigned, used_hosts):
            if len(assigned) == k:
                for u, v in tree_data[tree_idx]['edges']:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()
            
            unassigned = [v for v in range(k) if v not in assigned]
            best_v = -1
            min_domain_size = 21
            best_domain = []
            
            for v in unassigned:
                valid_hosts = []
                for h in range(20):
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in tree_data[tree_idx]['adj'][v]:
                        if u in assigned:
                            hu = assigned[u]
                            e_key = (min(h, hu), max(h, hu))
                            if e_key in used_edges:
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)
                
                if len(valid_hosts) < min_domain_size:
                    min_domain_size = len(valid_hosts)
                    best_v = v
                    best_domain = valid_hosts
                    if min_domain_size == 0:
                        break
            
            if best_v == -1 or min_domain_size == 0:
                return None
            
            rng.shuffle(best_domain) # Randomize tie-breaking
            
            for h in best_domain:
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                new_used_hosts = used_hosts | {h}
                result = backtrack(new_assigned, new_used_hosts)
                if result is not None:
                    return result
            return None
        
        # Shuffle root hosts
        root_hosts = list(range(20))
        rng.shuffle(root_hosts)
        
        for h in root_hosts:
            initial_assigned = {root: h}
            initial_used_hosts = {h}
            result = backtrack(initial_assigned, initial_used_hosts)
            if result is not None:
                mapping = [0] * k
                for tv, hv in result.items():
                    mapping[tv] = hv
                new_used_edges = set(used_edges)
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    new_used_edges.add(e_key)
                return mapping, new_used_edges
                
        return None, None

    # Run multiple restarts
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 6789)
        current_used_edges = set()
        current_mappings = [None] * 20
        success = True
        
        for tree_idx in order:
            if time.time() > deadline:
                success = False
                break
                
            mapping, new_used_edges = embed_tree_csp_random(tree_idx, current_used_edges, rng)
            if mapping is None:
                success = False
                break
            current_mappings[tree_idx] = mapping
            current_used_edges = new_used_edges
            
        if success:
            # Calculate score
            score = 0
            for i in range(20):
                if current_mappings[i] is not None:
                    score += len(tree_data[i]['edges'])
                    
            if score > best_score:
                best_score = score
                best_mappings = [m[:] if m else None for m in current_mappings]
                
            if best_score == 190:
                break
                
    # Final check and conversion
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
