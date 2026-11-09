"""Rewritten program: exhaustive search over 3-regular bipartite graphs on 12 vertices."""
# EVOLVE-BLOCK-START
import itertools
import time
import sys

def _count_distinct_cycle_lengths(adj, n):
    """Count distinct simple cycle lengths in a graph.
    Uses a DFS that explores all simple cycles, tracking visited nodes
    and their depths. For a bipartite graph on 12 vertices, this is fast.
    """
    cycle_lengths = set()
    
    def dfs(start, current, depth, visited, path):
        for nb in adj[current]:
            if nb == start and depth >= 3:
                cycle_lengths.add(depth + 1)
            elif nb > start and nb not in visited:
                # Only explore cycles where start is the smallest vertex
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, depth + 1, visited, path)
                path.pop()
                visited.remove(nb)
    
    for s in range(n):
        visited = {s}
        dfs(s, s, 0, visited, [s])
    
    return len(cycle_lengths)

def construct():
    """
    Search over all 3-regular bipartite graphs on 12 vertices (two parts of 6).
    A 3-regular bipartite graph on 6+6 vertices is a 3-regular bipartite graph,
    which is a 3-regular graph on 12 vertices. We enumerate all possible
    3-regular bipartite graphs (equivalent to all 3-regular graphs on 12
    vertices that are bipartite with parts of size 6), find the one with
    the fewest distinct cycle lengths, and return its edges.
    """
    deadline = time.time() + 80
    n = 12
    
    # The bipartition: part A = {0,1,2,3,4,5}, part B = {6,7,8,9,10,11}
    # Each vertex in A must connect to exactly 3 vertices in B.
    # Each vertex in B must have exactly 3 neighbors in A.
    # This is equivalent to finding a 6x6 0-1 matrix with all row sums = 3
    # and all column sums = 3.
    # We enumerate all such matrices by trying all subsets of 3 for each row,
    # then checking column sums.
    
    # All subsets of size 3 from {6,7,8,9,10,11}
    part_b = [6, 7, 8, 9, 10, 11]
    all_subsets_3 = list(itertools.combinations(part_b, 3))
    
    best_edges = None
    best_count = 100
    
    # For each assignment of subsets to the 6 rows, check column sums
    # This is 20^6 = 64,000,000 which is too slow.
    # Instead, use a smarter approach: backtracking with column sum tracking.
    
    def backtrack(row, col_sums, assignment):
        nonlocal best_edges, best_count
        if time.time() > deadline:
            return
        if row == 6:
            if all(c == 3 for c in col_sums):
                # Build adjacency and count cycle lengths
                adj = [[] for _ in range(n)]
                for r, subset in enumerate(assignment):
                    u = r  # vertex in part A
                    for v in subset:
                        idx_b = v - 6
                        adj[u].append(v)
                        adj[v].append(u)
                cnt = _count_distinct_cycle_lengths(adj, n)
                if cnt < best_count:
                    best_count = cnt
                    best_edges = []
                    for r, subset in enumerate(assignment):
                        u = r
                        for v in subset:
                            best_edges.append([u, v])
            return
        
        col_sums_new = list(col_sums)
        for subset in all_subsets_3:
            # Check if adding this subset keeps column sums <= 3
            valid = True
            for v in subset:
                idx_b = v - 6
                if col_sums_new[idx_b] >= 3:
                    valid = False
                    break
            if not valid:
                continue
            for v in subset:
                idx_b = v - 6
                col_sums_new[idx_b] += 1
            assignment.append(subset)
            backtrack(row + 1, col_sums_new, assignment)
            assignment.pop()
            for v in subset:
                idx_b = v - 6
                col_sums_new[idx_b] -= 1
    
    backtrack(0, [0]*6, [])
    
    if best_edges is not None:
        result = [[int(u), int(v)] for u, v in best_edges]
        return result
    
    # Fallback: two disjoint K_{3,3}
    edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    return [[int(u), int(v)] for u, v in edges]
# EVOLVE-BLOCK-END
