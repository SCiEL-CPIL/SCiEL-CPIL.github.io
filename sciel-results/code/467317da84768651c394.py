import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = P['n']
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    # Adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    
    # Precompute all cliques and independent sets of size >= 2
    # For n=14, this is feasible
    all_cliques = []
    all_independent = []
    
    # Generate all subsets and check
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            s = set(subset)
            # Check if clique
            is_clique = True
            for u in s:
                for v in s:
                    if u != v and v not in adj[u]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                all_cliques.append(tuple(subset))
            
            # Check if independent
            is_ind = True
            for u in s:
                for v in s:
                    if u != v and v in adj[u]:
                        is_ind = False
                        break
                if not is_ind:
                    break
            if is_ind:
                all_independent.append(tuple(subset))
    
    # Add singletons
    for v in range(n):
        all_cliques.append((v,))
        all_independent.append((v,))
    
    # Combine all possible parts (cliques and independent sets)
    # We need to find a partition into minimum number of parts
    # where each part is either a clique or independent set
    
    # Strategy: Try from 1 part upward
    # For each total number of parts, try all combinations of clique/independent types
    
    # Precompute: for each vertex, which parts (cliques and independents) contain it
    # This helps with backtracking
    
    # Actually, let's use a different approach:
    # Backtracking with pruning. Assign vertices one by one to parts.
    # Parts have types. We try to minimize total parts.
    
    # Better approach: Use the fact that n=14 is small.
    # We can try all possible numbers of parts from 1 to some max.
    # For each, we try all type assignments.
    
    # Even better: Since we need minimum parts, and the seed gives 7,
    # let's try 1, 2, 3, 4, 5, 6 parts.
    
    # For a given number of parts p and type assignment, we need to check
    # if we can partition vertices into p parts with those types.
    
    # Use backtracking with vertex ordering by degree (highest first)
    degrees = [len(adj[i]) for i in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    def can_partition(part_types):
        """Check if vertices can be partitioned into parts with given types.
        Returns the partition as list of lists, or None."""
        p = len(part_types)
        parts = [set() for _ in range(p)]
        assignment = [-1] * n
        
        # Precompute: for each vertex and each part type, which vertices it's compatible with
        # For clique part: vertex v can join part S if v is adjacent to all in S
        # For independent part: vertex v can join part S if v is not adjacent to any in S
        
        def feasible(v, pi):
            part = parts[pi]
            if not part:
                return True
            if part_types[pi] == 'clique':
                for u in part:
                    if u not in adj[v]:
                        return False
            else:
                for u in part:
                    if u in adj[v]:
                        return False
            return True
        
        # Use symmetry breaking: for the first vertex, try each part type position
        # but only up to the first occurrence of each type to avoid symmetric solutions
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = vertex_order[idx]
            
            # Try assigning v to each part
            # Symmetry breaking: if parts have same type and same size, only try first
            tried_types = {}
            for pi in range(p):
                pt = part_types[pi]
                # Symmetry: if we've already tried a part of this type with same size, skip
                size_key = (pt, len(parts[pi]))
                if size_key in tried_types:
                    continue
                tried_types[size_key] = True
                
                if feasible(v, pi):
                    parts[pi].add(v)
                    assignment[v] = pi
                    if backtrack(idx + 1):
                        return True
                    parts[pi].discard(v)
                    assignment[v] = -1
            return False
        
        if backtrack(0):
            return [list(parts[i]) for i in range(p)]
        return None
    
    best_result = None
    
    # Try total parts from 1 upward
    for total in range(1, 8):
        if time.time() > deadline:
            break
        # c = number of clique parts, total - c = number of independent parts
        for c in range(0, total + 1):
            if time.time() > deadline:
                break
            part_types = ['clique'] * c + ['independent'] * (total - c)
            result = can_partition(part_types)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = set(part) | {v}
                # Check clique
                is_c = True
                for u in trial:
                    for w in trial:
                        if u != w and w not in adj[u]:
                            is_c = False
                            break
                    if not is_c:
                        break
                # Check independent
                is_i = True
                for u in trial:
                    for w in trial:
                        if u != w and w in adj[u]:
                            is_i = False
                            break
                    if not is_i:
                        break
                if is_c or is_i:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_result = parts
    
    # Convert to list of lists with Python ints
    final = []
    for part in best_result:
        final.append([int(v) for v in part])
    return final
