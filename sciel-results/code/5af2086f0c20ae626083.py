from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)]); 
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    
    # Prepare tree data
    tree_data = []
    for i in range(20):
        k = i + 1
        edges = trees[i]
        if k == 1:
            order = [0]
        else:
            deg = [0] * k
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            vertices = list(range(k))
            vertices.sort(key=lambda v: deg[v], reverse=True)
            order = vertices
        tree_data.append((k, i, edges, order))
        
    # Sort by size descending
    tree_data.sort(key=lambda x: x[0], reverse=True)
    
    # All edges in K_20
    all_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            all_edges.add((u, v))
            
    mappings = [None] * 20
    
    # Backtracking solver
    def solve(idx, available_edges):
        if idx == 20:
            return True
            
        k, orig_idx, edges, order = tree_data[idx]
        
        if k == 1:
            mappings[orig_idx] = [0]
            return solve(idx + 1, available_edges)
            
        mapping_host = {}
        
        def assign_vertex(pos):
            if pos == len(order):
                used_host_edges = set()
                for u, v in edges:
                    hu = mapping_host[u]
                    hv = mapping_host[v]
                    he = (min(hu, hv), max(hu, hv))
                    if he not in available_edges:
                        return False
                    used_host_edges.add(he)
                    
                new_available = available_edges - used_host_edges
                mappings[orig_idx] = [mapping_host[i] for i in range(k)]
                
                if solve(idx + 1, new_available):
                    return True
                return False
            
            tv = order[pos]
            
            # Determine candidates
            assigned_neighbors = []
            for u, v in edges:
                if u == tv and v in mapping_host:
                    assigned_neighbors.append(mapping_host[v])
                elif v == tv and u in mapping_host:
                    assigned_neighbors.append(mapping_host[u])
            
            candidates = []
            for hv in range(20):
                if hv in mapping_host.values():
                    continue
                valid = True
                for an in assigned_neighbors:
                    he = (min(hv, an), max(hv, an))
                    if he not in available_edges:
                        valid = False
                        break
                if valid:
                    candidates.append(hv)
                    
            # Heuristic: Sort candidates by number of available edges to unassigned tree neighbors?
            # This is expensive to compute for every candidate.
            # Simple heuristic: Just iterate.
            
            for hv in candidates:
                mapping_host[tv] = hv
                if assign_vertex(pos + 1):
                    return True
                del mapping_host[tv]
                
            return False
            
        # Symmetry breaking for the first tree
        if idx == 0:
            # Fix the first vertex in order to 0
            first_tv = order[0]
            mapping_host[first_tv] = 0
            return assign_vertex(1)
        else:
            return assign_vertex(0)
            
    success = solve(0, all_edges)
    
    if not success:
        # Fallback
        return [None] * 19 + [list(range(20))]
        
    return mappings
