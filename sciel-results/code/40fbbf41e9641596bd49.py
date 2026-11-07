"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency set for quick lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    def is_biclique(left, right):
        """Check if all cross pairs are edges."""
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
        return True
    
    def get_edges_for_biclique(left, right):
        """Get the set of edges covered by this biclique."""
        es = set()
        for u in left:
            for v in right:
                if v in adj[u]:
                    es.add(frozenset((u, v)))
        return es
    
    # Generate candidate bicliques
    # We'll generate all bicliques of various sizes
    candidates = []
    
    # Single edges
    for u, v in edges:
        if u < v:
            candidates.append(([u], [v]))
        else:
            candidates.append(([v], [u]))
    
    # Pairs on one side
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            for v in common:
                if is_biclique([i, j], [v]):
                    candidates.append(([i, j], [v]))
    
    # Pairs on both sides (small bicliques)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                if is_biclique([i, j], [k]):
                    # Already covered above, but let's also try 2x2
                    pass
    
    # Try to grow bicliques from edges
    for u, v in edges:
        # Start with [u], [v]
        # Try to add more to left
        for x in range(n):
            if x == u:
                continue
            # Check if x can be added to left
            if v in adj[x]:
                new_left = [u, x]
                new_right = [v]
                if is_biclique(new_left, new_right):
                    candidates.append((new_left, new_right))
                    # Try to add to right too
                    for y in range(n):
                        if y == v or y == x:
                            continue
                        if y in adj[u] and y in adj[x]:
                            if is_biclique(new_left, [v, y]):
                                candidates.append((new_left, [v, y]))
    
    # Remove duplicates by edge set
    seen_edge_sets = set()
    unique_candidates = []
    for left, right in candidates:
        es = get_edges_for_biclique(left, right)
        key = frozenset(es)
        if key not in seen_edge_sets:
            seen_edge_sets.add(key)
            unique_candidates.append((left, right, es))
    
    # Greedy set cover to get initial solution
    remaining = set(edge_set)
    solution = []
    
    while remaining:
        best_cand = None
        best_covered = set()
        for left, right, es in unique_candidates:
            covered = es & remaining
            if len(covered) > len(best_covered):
                best_covered = covered
                best_cand = (left, right)
        
        if best_cand is None or not best_covered:
            # Fallback: pick any remaining edge
            edge = next(iter(remaining))
            u, v = list(edge)
            solution.append(([u], [v]))
            remaining.remove(edge)
        else:
            solution.append(list(best_cand))
            remaining -= best_covered
    
    initial_count = len(solution)
    
    # Try to improve with local search
    # Merge bicliques when possible
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                Li, Ri = solution[i]
                Lj, Rj = solution[j]
                
                # Try merging: (Li ∪ Lj, Ri) or (Li, Ri ∪ Rj) or (Li ∪ Lj, Ri ∪ Rj)
                for L_new, R_new in [
                    (list(set(Li) | set(Lj)), list(Ri)),
                    (list(Li), list(set(Ri) | set(Rj))),
                    (list(set(Li) | set(Lj)), list(set(Ri) | set(Rj))),
                ]:
                    if not L_new or not R_new:
                        continue
                    if is_biclique(L_new, R_new):
                        es_new = get_edges_for_biclique(L_new, R_new)
                        es_i = get_edges_for_biclique(Li, Ri)
                        es_j = get_edges_for_biclique(Lj, Rj)
                        
                        # Check if merged covers both
                        if es_i | es_j <= es_new:
                            new_solution = []
                            for k in range(len(solution)):
                                if k != i and k != j:
                                    new_solution.append(solution[k])
                            new_solution.append([L_new, R_new])
                            solution = new_solution
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Final validation
    def validate_partition(part):
        covered = set()
        for left, right in part:
            if not left or not right:
                return False
            if set(left) & set(right):
                return False
            if not is_biclique(left, right):
                return False
            es = get_edges_for_biclique(left, right)
            if covered & es:
                return False
            covered |= es
        return covered == edge_set
    
    if not validate_partition(solution):
        # Fallback to seed solution
        solution = [[[u], [v for a, v in edges if a == u]] for u in range(n) if any(a == u for a, v in edges)]
    
    # Ensure types
    result = []
    for left, right in solution:
        result.append([[int(x) for x in left], [int(x) for x in right]])
    
    return result

# EVOLVE-BLOCK-END
