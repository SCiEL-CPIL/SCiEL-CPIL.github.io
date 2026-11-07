import time
import random
from collections import defaultdict
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np
import scipy.sparse as sp

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency matrix and edge set
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))

    # Assign each edge to the smaller-degree endpoint's side
    # This ensures each edge appears in exactly one biclique
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    # Determine side assignment for each vertex
    # Vertices with lower degree go to left, higher to right
    # For ties, use vertex index as tiebreaker
    side = [0] * n  # 0 = left, 1 = right
    for v in range(n):
        # Count how many neighbors have lower/equal degree
        left_count = 0
        right_count = 0
        for u in range(n):
            if u == v:
                continue
            if adj[v][u]:
                if deg[u] < deg[v] or (deg[u] == deg[v] and u < v):
                    left_count += 1
                else:
                    right_count += 1
        # If more neighbors are "left" (lower degree), put v on right
        # Otherwise put v on left
        if right_count > left_count:
            side[v] = 1
        else:
            side[v] = 0

    # Build bicliques based on side assignment
    # For each vertex v on left, its biclique is {v} x {all neighbors on right}
    # For each vertex v on right, its biclique is {all neighbors on left} x {v}
    bicliques = []
    covered_edges = set()

    for v in range(n):
        if side[v] == 0:  # Left side
            R = [u for u in range(n) if adj[v][u] and side[u] == 1]
            if R:
                L = [v]
                # Check all cross edges exist and are not already covered
                valid = True
                for u in R:
                    e = (min(v, u), max(v, u))
                    if e in covered_edges:
                        valid = False
                        break
                if valid:
                    bicliques.append([L, R])
                    for u in R:
                        covered_edges.add((min(v, u), max(v, u)))
        else:  # Right side
            L = [u for u in range(n) if adj[v][u] and side[u] == 0]
            if L:
                R = [v]
                # Check all cross edges exist and are not already covered
                valid = True
                for u in L:
                    e = (min(u, v), max(u, v))
                    if e in covered_edges:
                        valid = False
                        break
                if valid:
                    bicliques.append([L, R])
                    for u in L:
                        covered_edges.add((min(u, v), max(u, v)))

    # Handle any uncovered edges (shouldn't happen with perfect side assignment)
    uncovered = edge_set - covered_edges
    for e in uncovered:
        u, v = e
        bicliques.append([[u], [v]])
        covered_edges.add(e)

    def validate(bicliques):
        """Validate and convert bicliques."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return None
            for u in L:
                if u in R:
                    return None  # Sides must be disjoint
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return None  # Overlap
                    if not adj[u][v]:
                        return None  # Not an edge
                    covered.add(e)
        if covered != edge_set:
            return None
        result = []
        for L, R in bicliques:
            result.append([[int(x) for x in L], [int(x) for x in R]])
        return result

    def can_merge(L1, R1, L2, R2):
        """Check if two bicliques can be merged."""
        for u in L1:
            for v in R2:
                if not adj[u][v]:
                    return False
        for u in L2:
            for v in R1:
                if not adj[u][v]:
                    return False
        return True

    def local_search(bicliques, time_limit):
        """Aggressive local search: merge, split, and swap bicliques."""
        best = bicliques
        best_count = len(bicliques)
        no_improve_count = 0

        while time.time() < time_limit and no_improve_count < 5:
            improved = False
            no_improve_count += 1

            # Try merging pairs
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    L1, R1 = best[i]
                    L2, R2 = best[j]
                    if can_merge(L1, R1, L2, R2):
                        new_L = list(set(L1 + L2))
                        new_R = list(set(R1 + R2))
                        if not new_L or not new_R:
                            continue
                        # Verify all cross edges exist
                        valid = True
                        for u in new_L:
                            for v in new_R:
                                if not adj[u][v]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            new_bics = [best[k] for k in range(len(best)) if k != i and k != j]
                            new_bics.append([new_L, new_R])
                            result = validate(new_bics)
                            if result is not None and len(result) < best_count:
                                best = new_bics
                                best_count = len(result)
                                improved = True
                                no_improve_count = 0
                                break
                if improved:
                    break

            if improved:
                continue

            # Try splitting a biclique into two
            for i in range(len(best)):
                L, R = best[i]
                if len(L) > 1 or len(R) > 1:
                    # Try to split by removing one vertex from L or R
                    # and reassigning its edges to another biclique
                    for u in L:
                        if len(L) <= 1:
                            continue
                        # Create new biclique with u and a subset of R
                        # Find which other bicliques can take u's edges
                        new_L = [u]
                        new_R = []
                        remaining_L = [x for x in L if x != u]
                        remaining_R = R

                        # Try to find a subset of R that forms a valid biclique with u
                        # and the rest can be covered by other bicliques
                        for r in R:
                            e = (min(u, r), max(u, r))
                            # Check if e is covered by another biclique
                            covered_by_other = False
                            for k in range(len(best)):
                                if k == i:
                                    continue
                                Lk, Rk = best[k]
                                if u in Lk and r in Rk:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                new_R.append(r)

                        if new_R and remaining_L:
                            # Check if remaining_L x remaining_R is valid
                            valid_rem = True
                            for x in remaining_L:
                                for y in remaining_R:
                                    if not adj[x][y]:
                                        valid_rem = False
                                        break
                                if not valid_rem:
                                    break
                            if valid_rem:
                                new_bics = [best[k] for k in range(len(best)) if k != i]
                                new_bics.append([remaining_L, remaining_R])
                                new_bics.append([new_L, new_R])
                                result = validate(new_bics)
                                if result is not None and len(result) < best_count:
                                    best = new_bics
                                    best_count = len(result)
                                    improved = True
                                    no_improve_count = 0
                                    break
                    if improved:
                        break

            if not improved:
                break

        return best

    # Validate initial solution
    result = validate(bicliques)
    if result is None:
        # Fallback: individual edges
        result = [[[u], [v]] for u, v in sorted(edge_set)]
    else:
        # Local search to improve
        bicliques_for_ls = result
        result = local_search(bicliques_for_ls, min(time.time() + 30, deadline))
        if result is None:
            result = [[[u], [v]] for u, v in sorted(edge_set)]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
