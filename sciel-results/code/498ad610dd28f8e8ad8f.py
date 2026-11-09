"""Packing trees T1..T20 into K20 via backtracking with greedy initialization."""
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
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    n = 20
    total_edges_k20 = n * (n - 1) // 2  # 190
    
    # Precompute edges for each tree
    tree_edges = []
    for edges in trees:
        edge_set = set()
        for (u, v) in edges:
            edge_set.add(frozenset((u, v)))
        tree_edges.append(list(edge_set))
    
    # Order: embed largest trees first (more constraints, harder to fit)
    # T20 has 19 edges, T19 has 18, etc.
    # We'll embed in order from largest to smallest for backtracking
    order = list(range(1, 20))  # indices 1..19 for T2..T20
    # Sort by number of edges descending (largest first)
    order.sort(key=lambda i: len(tree_edges[i]), reverse=True)
    
    best_result = [None] * 20
    best_score = 0
    
    # Greedy initial solution
    used_edges = set()
    greedy_result = [None] * 20
    
    for idx in order:
        k = idx + 1  # tree index 0-based -> k = idx+1
        edges_to_embed = tree_edges[idx]
        n_vertices = k
        
        # Try to find an embedding via backtracking
        # Map tree vertices 0..k-1 to host vertices 0..19
        # Such that all edges map to distinct unused edges in K20
        
        # For greedy: try random permutations
        found = False
        attempts = 0
        max_attempts = 50000
        rng_local = Random(idx * 1000 + 7)
        
        while not found and attempts < max_attempts and (time.time() - start_time) < deadline - 5:
            perm = list(range(n))
            rng_local.shuffle(perm)
            # perm[i] = host vertex for tree vertex i
            
            valid = True
            for edge in edges_to_embed:
                u, v = tuple(sorted(edge))
                hu, hv = perm[u], perm[v]
                he = frozenset((hu, hv))
                if he in used_edges:
                    valid = False
                    break
            
            if valid:
                for edge in edges_to_embed:
                    u, v = tuple(sorted(edge))
                    hu, hv = perm[u], perm[v]
                    used_edges.add(frozenset((hu, hv)))
                greedy_result[idx] = perm
                found = True
            attempts += 1
        
        if not found:
            # Try a more systematic approach for this tree
            # Use backtracking with ordering
            pass  # Leave as None for now
    
    # Calculate greedy score
    greedy_score = sum(len(tree_edges[i]) for i in range(1, 20) if greedy_result[i] is not None)
    if greedy_score > best_score:
        best_score = greedy_score
        best_result = list(greedy_result)
    
    # Now try to improve by embedding remaining trees
    # Use backtracking for trees that weren't embedded
    unembedded = [idx for idx in order if greedy_result[idx] is None]
    
    if unembedded and (time.time() - start_time) < deadline - 10:
        # Try to embed unembedded trees one by one with backtracking
        for idx in unembedded:
            k = idx + 1
            edges_to_embed = tree_edges[idx]
            n_vertices = k
            
            # Backtracking search
            perm = [-1] * n_vertices
            found = False
            
            def backtrack_bt(pos):
                nonlocal found
                if (time.time() - start_time) > deadline - 5:
                    return True  # timeout signal
                if pos == n_vertices:
                    # Check all edges
                    for edge in edges_to_embed:
                        u, v = tuple(sorted(edge))
                        he = frozenset((perm[u], perm[v]))
                        if he in used_edges:
                            return False
                    found = True
                    return True
                # Try assigning host vertex to tree vertex pos
                for hv in range(n):
                    if hv in perm[:pos]:
                        continue
                    perm[pos] = hv
                    # Pruning: check edges from pos to already-assigned vertices
                    conflict = False
                    for edge in edges_to_embed:
                        u, v = tuple(sorted(edge))
                        if u == pos:
                            other = v
                        elif v == pos:
                            other = u
                        else:
                            continue
                        if perm[other] != -1:
                            he = frozenset((hv, perm[other]))
                            if he in used_edges:
                                conflict = True
                                break
                    if not conflict:
                        if backtrack_bt(pos + 1):
                            return True
                perm[pos] = -1
                return False
            
            if not backtrack_bt(0) and found:
                for edge in edges_to_embed:
                    u, v = tuple(sorted(edge))
                    used_edges.add(frozenset((perm[u], perm[v])))
                best_result[idx] = perm
                best_score = sum(len(tree_edges[i]) for i in range(1, 20) if best_result[i] is not None)
            
            if (time.time() - start_time) > deadline - 5:
                break
    
    # Final: try to embed any remaining None entries with quick greedy
    for idx in range(1, 20):
        if best_result[idx] is None and (time.time() - start_time) < deadline - 2:
            k = idx + 1
            edges_to_embed = tree_edges[idx]
            n_vertices = k
            rng_local = Random(idx * 2000 + 13)
            for _ in range(20000):
                perm = list(range(n))
                rng_local.shuffle(perm)
                valid = True
                for edge in edges_to_embed:
                    u, v = tuple(sorted(edge))
                    he = frozenset((perm[u], perm[v]))
                    if he in used_edges:
                        valid = False
                        break
                if valid:
                    for edge in edges_to_embed:
                        u, v = tuple(sorted(edge))
                        used_edges.add(frozenset((perm[u], perm[v])))
                    best_result[idx] = perm
                    best_score = sum(len(tree_edges[i]) for i in range(1, 20) if best_result[i] is not None)
                    break
    
    # Ensure types are correct
    final_result = []
    for i in range(20):
        if best_result[i] is None:
            final_result.append(None)
        else:
            # Convert to list of Python ints
            final_result.append([int(x) for x in best_result[i]])
    
    return final_result
