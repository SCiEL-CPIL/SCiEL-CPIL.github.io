import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set(frozenset(e) for e in edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85  # safety margin
    
    def validate(bicliques):
        """Check if biclique partition is valid."""
        covered = set()
        for left, right in bicliques:
            if not left or not right:
                return False
            # Check disjoint sides
            if set(left) & set(right):
                return False
            # Check all cross pairs are edges
            for u in left:
                for v in right:
                    e = frozenset((u, v))
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edge_set
    
    def greedy_biclique(remaining_edges, adj, n, max_size=None):
        """Find a large biclique from remaining edges."""
        if not remaining_edges:
            return None, None, 0
        
        # Try different starting pairs
        best_L, best_R, best_count = None, None, 0
        edge_list = list(remaining_edges)
        
        # Sample a few edges to start from
        if len(edge_list) > 10:
            candidates = random.sample(edge_list, 10)
        else:
            candidates = edge_list
        
        for u0, v0 in candidates:
            L = {u0}
            R = {v0}
            # Iteratively expand
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for w in adj[u0]:
                    if w not in L and w not in R:
                        if all(frozenset((w, r)) in remaining_edges for r in R):
                            L.add(w)
                            changed = True
                # Try adding to R
                for w in adj[v0]:
                    if w not in R and w not in L:
                        if all(frozenset((l, w)) in remaining_edges for l in L):
                            R.add(w)
                            changed = True
            
            count = len(L) * len(R)
            if count > best_count:
                best_L, best_R, best_count = set(L), set(R), count
        
        if best_L is None:
            # Fallback: single edge
            u, v = edge_list[0]
            return {u}, {v}, 1
        
        return best_L, best_R, best_count
    
    def greedy_partition(remaining_edges, adj, n):
        """Greedy: repeatedly find largest biclique."""
        bicliques = []
        remaining = set(remaining_edges)
        
        while remaining:
            L, R, count = greedy_biclique(remaining, adj, n)
            if L is None:
                # Single edge
                e = next(iter(remaining))
                bicliques.append([list(e), [list(e)[1]]])
                remaining.discard(e)
                continue
            
            # Remove covered edges
            covered = set()
            for u in L:
                for v in R:
                    covered.add(frozenset((u, v)))
            remaining -= covered
            bicliques.append([sorted(L), sorted(R)])
        
        return bicliques
    
    def local_search(bicliques, remaining_edges, adj, n, max_iters=200):
        """Try to merge bicliques or move edges to reduce count."""
        bicliques = [list(b) for b in bicliques]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try merging pairs
            for i in range(len(bicliques)):
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                    new_L = set(L1) | set(L2)
                    new_R = set(R1) | set(R2)
                    if not new_L or not new_R:
                        continue
                    if new_L & new_R:
                        continue
                    # Check all cross pairs are edges
                    valid = True
                    for u in new_L:
                        for v in new_R:
                            if frozenset((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        bicliques[i] = [sorted(new_L), sorted(new_R)]
                        bicliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
            # Try moving vertices between bicliques
            if not improved:
                for i in range(len(bicliques)):
                    L1, R1 = bicliques[i]
                    for j in range(len(bicliques)):
                        if i == j:
                            continue
                        L2, R2 = bicliques[j]
                        # Try moving a vertex from L1 to L2
                        for v in list(L1):
                            new_L1 = set(L1) - {v}
                            new_L2 = set(L2) | {v}
                            if not new_L1 or not new_L2:
                                continue
                            if new_L1 & set(R1) or new_L2 & set(R2):
                                continue
                            valid1 = all(frozenset((u, r)) in edge_set for u in new_L1 for r in R1)
                            valid2 = all(frozenset((u, r)) in edge_set for u in new_L2 for r in R2)
                            if valid1 and valid2:
                                bicliques[i] = [sorted(new_L1), list(R1)]
                                bicliques[j] = [sorted(new_L2), list(R2)]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return bicliques
    
    # Initial greedy solution
    all_edges = set(frozenset(e) for e in edges)
    best_bicliques = greedy_partition(all_edges, adj, n)
    best_count = len(best_bicliques)
    
    # Local search
    if time.time() < deadline:
        improved_bicliques = local_search(best_bicliques, all_edges, adj, n, max_iters=500)
        if len(improved_bicliques) < best_count:
            best_bicliques = improved_bicliques
            best_count = len(best_bicliques)
    
    # Try multiple random restarts with different random seeds
    if time.time() < deadline:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            random.seed(restart * 12345)
            bc = greedy_partition(all_edges, adj, n)
            if len(bc) < best_count:
                best_bicliques = bc
                best_count = len(bc)
            if time.time() < deadline - 10:
                bc = local_search(bc, all_edges, adj, n, max_iters=300)
                if len(bc) < best_count:
                    best_bicliques = bc
                    best_count = len(bc)
    
    # Validate and return
    if not validate(best_bicliques):
        # Fallback to seed solution
        best_bicliques = [[[u], [v for a, v in edges if a == u]] for u in range(n) if any(a == u for a, v in edges)]
    
    # Convert to proper types
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
