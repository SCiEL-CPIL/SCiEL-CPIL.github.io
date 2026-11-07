import time
import itertools
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _greedy_partition(edges, n, deadline, order_seed=0):
    """Greedy biclique partition with configurable edge ordering."""
    E_uncov = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        E_uncov.add((u, v))
    
    if not E_uncov:
        return []
    
    adj = [set() for _ in range(n)]
    for u, v in E_uncov:
        adj[u].add(v)
        adj[v].add(u)
    
    bicliques = []
    
    while E_uncov:
        if time.time() > deadline:
            break
            
        best_cov = 0
        best_L = None
        best_R = None
        
        # Size 1: L={u}, R=N(u)
        for u in range(n):
            if not adj[u]:
                continue
            R = set(adj[u])
            cov = len(R)
            if cov > best_cov:
                best_cov = cov
                best_L = [u]
                best_R = list(R)
        
        # Size 2: L={u,v}, R=N(u)∩N(v)\{u,v}
        for u in range(n):
            if not adj[u]:
                continue
            for v in adj[u]:
                if u >= v:
                    continue
                R = adj[u] & adj[v]
                R.discard(u)
                R.discard(v)
                if not R:
                    continue
                cov = 2 * len(R)
                if cov > best_cov:
                    best_cov = cov
                    best_L = [u, v]
                    best_R = list(R)
        
        # Size 3: L={u,v,w}, R=N(u)∩N(v)∩N(w)\{u,v,w}
        for u in range(n):
            if len(adj[u]) < 2:
                continue
            for v in adj[u]:
                if u >= v:
                    continue
                if len(adj[v]) < 2:
                    continue
                common_uv = adj[u] & adj[v]
                common_uv.discard(u)
                common_uv.discard(v)
                if not common_uv:
                    continue
                for w in common_uv:
                    if v >= w:
                        continue
                    R = common_uv & adj[w]
                    R.discard(u)
                    R.discard(v)
                    R.discard(w)
                    if not R:
                        continue
                    cov = 3 * len(R)
                    if cov > best_cov:
                        best_cov = cov
                        best_L = [u, v, w]
                        best_R = list(R)
        
        # Size 4: L={u,v,w,x}, R=N(u)∩N(v)∩N(w)∩N(x)\{u,v,w,x}
        # Only try if we have time and the graph is small enough
        if time.time() < deadline - 10:
            for u in range(n):
                if len(adj[u]) < 3:
                    continue
                for v in adj[u]:
                    if u >= v:
                        continue
                    if len(adj[v]) < 3:
                        continue
                    common_uv = adj[u] & adj[v]
                    common_uv.discard(u)
                    common_uv.discard(v)
                    if len(common_uv) < 2:
                        continue
                    for w in common_uv:
                        if v >= w:
                            continue
                        common_uvw = common_uv & adj[w]
                        common_uvw.discard(u)
                        common_uvw.discard(v)
                        common_uvw.discard(w)
                        if len(common_uvw) < 1:
                            continue
                        for x in common_uvw:
                            if w >= x:
                                continue
                            R = common_uvw & adj[x]
                            R.discard(u)
                            R.discard(v)
                            R.discard(w)
                            R.discard(x)
                            if not R:
                                continue
                            cov = 4 * len(R)
                            if cov > best_cov:
                                best_cov = cov
                                best_L = [u, v, w, x]
                                best_R = list(R)
        
        if best_cov == 0:
            u, v = next(iter(E_uncov))
            best_L = [u]
            best_R = [v]
            best_cov = 1
            
        bicliques.append([best_L, best_R])
        
        for u in best_L:
            for v in best_R:
                edge = (min(u, v), max(u, v))
                if edge in E_uncov:
                    E_uncov.remove(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
    # Fallback: cover remaining edges individually
    if E_uncov:
        for e in E_uncov:
            bicliques.append([[e[0]], [e[1]]])
    
    return bicliques


def _validate_partition(bicliques, edges, n):
    """Check if partition is valid and return count."""
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    covered = set()
    for L, R in bicliques:
        L_set = set(L)
        R_set = set(R)
        if not L or not R:
            return -1
        if L_set & R_set:
            return -1
        for u in L:
            for v in R:
                edge = (min(u, v), max(u, v))
                if edge not in edge_set:
                    return -1
                covered.add(edge)
    
    if covered != edge_set:
        return -1
    return len(bicliques)


def _local_search(bicliques, edges, n, deadline):
    """Try to improve partition by merging adjacent bicliques or swapping vertices."""
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    current = bicliques
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                # Try merging biclique i and j
                L_i, R_i = current[i]
                L_j, R_j = current[j]
                
                # Try: combined L = L_i ∪ L_j, R = R_i ∪ R_j
                # Check if all cross edges exist
                L_comb = set(L_i) | set(L_j)
                R_comb = set(R_i) | set(R_j)
                if L_comb & R_comb:
                    continue
                
                valid = True
                for u in L_comb:
                    for v in R_comb:
                        edge = (min(u, v), max(u, v))
                        if edge not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check that merged biclique covers exactly the union of edges from i and j
                edges_i = set()
                for u in L_i:
                    for v in R_i:
                        edges_i.add((min(u, v), max(u, v)))
                edges_j = set()
                for u in L_j:
                    for v in R_j:
                        edges_j.add((min(u, v), max(u, v)))
                
                edges_merged = set()
                for u in L_comb:
                    for v in R_comb:
                        edges_merged.add((min(u, v), max(u, v)))
                
                if edges_merged == edges_i | edges_j:
                    new_current = list(current)
                    new_current[i] = [list(L_comb), list(R_comb)]
                    new_current.pop(j)
                    
                    # Validate
                    if _validate_partition(new_current, edges, n) == len(new_current):
                        current = new_current
                        improved = True
                        break
            if improved:
                break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try multiple strategies and keep the best
    best_result = None
    best_count = float('inf')
    
    # Strategy 1: Standard greedy
    result1 = _greedy_partition(edges, n, deadline)
    count1 = _validate_partition(result1, edges, n)
    if count1 > 0 and count1 < best_count:
        best_count = count1
        best_result = result1
    
    # Strategy 2: Greedy with different vertex ordering (reverse)
    # We can try by reversing the order vertices are considered
    # This is approximated by trying a slightly different greedy
    if time.time() < deadline - 20:
        # Try local search on the first result
        improved = _local_search(result1, edges, n, deadline)
        count2 = _validate_partition(improved, edges, n)
        if count2 > 0 and count2 < best_count:
            best_count = count2
            best_result = improved
    
    # Strategy 3: Try greedy with size-4 emphasis (already in _greedy_partition)
    # Strategy 4: Multi-restart with random tie-breaking
    if time.time() < deadline - 10:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            # Slight perturbation: remove a random edge first, then greedy
            E_copy = list(edges)
            if E_copy:
                idx = random.randint(0, len(E_copy) - 1)
                removed = E_copy.pop(idx)
                result3 = _greedy_partition(E_copy, n, deadline)
                # Add back the removed edge as a single biclique if not covered
                # Actually, _greedy_partition handles all edges passed to it
                # So we need to include all edges
                result3 = _greedy_partition(edges, n, deadline)
                count3 = _validate_partition(result3, edges, n)
                if count3 > 0 and count3 < best_count:
                    best_count = count3
                    best_result = result3
    
    if best_result is None:
        # Ultimate fallback: one biclique per edge
        best_result = [[ [u], [v] ] for u, v in edges]
    
    # Convert to required format with exact types
    result = []
    for L, R in best_result:
        result.append([[int(x) for x in L], [int(x) for x in R]])
    
    return result
