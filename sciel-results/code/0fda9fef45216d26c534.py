import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def is_valid(edge_set, n):
        """Check if no simple three-edge path has endpoints of equal degree."""
        if not edge_set:
            return True
            
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # Optimization: Group vertices by degree
        deg_map = {}
        for i in range(n):
            if deg[i] not in deg_map:
                deg_map[deg[i]] = []
            deg_map[deg[i]].append(i)
        
        # For any edge (u, v), check paths s-u-v-t
        for (u, v) in edge_set:
            nu = [s for s in adj[u] if s != v]
            nv = [t for t in adj[v] if t != u]
            
            # Check for s in nu, t in nv with s != t and deg[s] == deg[t]
            # We can iterate over s in nu
            for s in nu:
                ds = deg[s]
                # Find t in nv with deg[t] == ds and t != s
                # nv is small, just check
                for t in nv:
                    if t == s:
                        continue
                    if deg[t] == ds:
                        return False
        return True

    # Known valid solution: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = [(a, b) for a in A for b in B]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    best_set = base_set
    best_count = len(best_set)
    
    # Candidate edges to add: edges within A and within B
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            candidates.append((B[i], B[j]))
    
    # Local Search with Swaps
    # Current state
    current_set = set(base_set)
    
    # List of all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    # Try to improve by adding edges
    # If adding fails, try swapping: add e_add, remove e_rem
    # This keeps edge count same, but might allow further additions?
    # Or just try random restarts of greedy.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(candidates)
        for e in candidates:
            if time.time() > deadline:
                break
            e_norm = tuple(sorted(e))
            if e_norm in current_set:
                continue
            
            test_set = set(current_set)
            test_set.add(e_norm)
            
            if is_valid(test_set, n):
                current_set = test_set
                improved = True
                if len(current_set) > best_count:
                    best_set = current_set
                    best_count = len(best_set)
                break # Restart scan
    
    # If stuck, try random restarts with different shuffles
    for _ in range(20):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        current_set = set(base_set)
        
        for e in candidates:
            if time.time() > deadline:
                break
            e_norm = tuple(sorted(e))
            test_set = set(current_set)
            test_set.add(e_norm)
            if is_valid(test_set, n):
                current_set = test_set
        
        if len(current_set) > best_count:
            best_set = current_set
            best_count = len(best_set)

    # Final validation and formatting
    if not is_valid(best_set, n):
        best_set = base_set
        best_count = len(base_set)

    result = [list(e) for e in best_set]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
