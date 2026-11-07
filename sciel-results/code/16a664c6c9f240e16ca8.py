import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix for O(1) edge checks
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        
    # Helper to check if two bicliques can be merged
    # Biclique 1: L1, R1. Biclique 2: L2, R2.
    # Merged L: L1 U L2, Merged R: R1 U R2.
    # Valid if:
    # 1. All pairs in (L1 U L2) x (R1 U R2) are edges in the graph.
    # This implies:
    # - L1 x R1 (already edges)
    # - L2 x R2 (already edges)
    # - L1 x R2 must be edges
    # - L2 x R1 must be edges
    def can_merge(L1, R1, L2, R2):
        # Check cross terms
        for u in L1:
            for v in R2:
                if not adj[u][v]:
                    return False
        for u in L2:
            for v in R1:
                if not adj[u][v]:
                    return False
        return True

    def merge(L1, R1, L2, R2):
        new_L = list(set(L1) | set(L2))
        new_R = list(set(R1) | set(R2))
        return new_L, new_R

    # 1. Generate an initial valid partition using a simple greedy heuristic
    # This ensures we start with a valid solution.
    remaining_edges = set(edge_set)
    initial_bicliques = []
    
    def greedy_partition():
        rem = set(remaining_edges)
        bics = []
        while rem:
            # Pick an edge
            u, v = next(iter(rem))
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for x in range(n):
                    if x in L: continue
                    # x must be connected to all in R
                    if all(adj[x][y] for y in R):
                        # And edges must be in rem
                        if all((min(x, y), max(x, y)) in rem for y in R):
                            L.add(x)
                            changed = True
                # Try adding to R
                for y in range(n):
                    if y in R: continue
                    if all(adj[x][y] for x in L):
                        if all((min(x, y), max(x, y)) in rem for x in L):
                            R.add(y)
                            changed = True
            b_edges = set((min(u, v), max(u, v)) for u in L for v in R)
            bics.append([list(L), list(R)])
            rem -= b_edges
        return bics

    current_bicliques = greedy_partition()
    
    # 2. Local Search: Iteratively try to merge pairs of bicliques
    # We repeat passes until no merges are found or time is up.
    # To avoid O(N^2 * N^2) checks being too slow, we limit passes and use randomization if needed.
    # For N=30, the number of bicliques is likely small (< 10-15), so N^2 checks are cheap.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to break symmetry and potentially find different merges
        indices = list(range(len(current_bicliques)))
        random.shuffle(indices)
        
        # We iterate through pairs. If a merge is successful, we restart the pass
        # because the list changed.
        for i_idx in range(len(indices)):
            i = indices[i_idx]
            if i >= len(current_bicliques): continue
            for j_idx in range(i_idx + 1, len(indices)):
                j = indices[j_idx]
                if j >= len(current_bicliques): continue
                
                L1, R1 = current_bicliques[i]
                L2, R2 = current_bicliques[j]
                
                if can_merge(L1, R1, L2, R2):
                    new_L, new_R = merge(L1, R1, L2, R2)
                    # Replace i and j with the merged biclique
                    # Remove j first to keep indices stable for i
                    current_bicliques.pop(j)
                    current_bicliques[i] = [new_L, new_R]
                    improved = True
                    # Restart the outer loop to check new combinations
                    break
            if improved:
                break
                
    # 3. Final validation and type conversion
    # Check validity to be safe
    covered_edges = set()
    for L, R in current_bicliques:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in covered_edges:
                    return [] # Should not happen if logic is correct, but safety
                covered_edges.add(e)
                if not adj[u][v]:
                    return [] # Invalid biclique
    
    if covered_edges != edge_set:
        return [] # Missing edges
        
    final_result = []
    for L, R in current_bicliques:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])
        
    return final_result

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found {len(res)} bicliques")
