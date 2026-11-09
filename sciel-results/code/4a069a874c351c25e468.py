import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def _instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

_TREES = _instance_trees()

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 20
    # Edge availability bitmask per vertex: bit j set means edge (v, j) is free
    # Represent as list of int bitmasks
    edge_avail = [0] * N
    for v in range(N):
        mask = 0
        for u in range(N):
            if u != v:
                mask |= (1 << u)
        edge_avail[v] = mask

    # Embedding order: largest trees first (most edges = hardest)
    # Order: T20, T19, ..., T1
    order = list(range(20, 0, -1))  # tree indices 19 down to 0 (T20 down to T1)

    result = [None] * 20  # result[k-1] for tree k

    def try_embed(tree_idx, assignment, depth):
        """Backtrack: assign host vertices to tree vertices 0..k-1."""
        if time.time() > deadline:
            return False
        k = tree_idx + 1
        if depth == k:
            # All vertices assigned; check all edges are available
            edges = _TREES[tree_idx]
            for (a, b) in edges:
                ha, hb = assignment[a], assignment[b]
                # Check edge (ha, hb) is still available
                if not (edge_avail[ha] & (1 << hb)):
                    return False
                if not (edge_avail[hb] & (1 << ha)):
                    return False
            # Commit edges
            for (a, b) in edges:
                ha, hb = assignment[a], assignment[b]
                edge_avail[ha] &= ~(1 << hb)
                edge_avail[hb] &= ~(1 << ha)
            result[tree_idx] = list(assignment)
            return True
        return False

    def backtrack(tree_idx, assignment, depth):
        if time.time() > deadline:
            return False
        k = tree_idx + 1
        if depth == k:
            # All vertices assigned; verify and commit
            edges = _TREES[tree_idx]
            for (a, b) in edges:
                ha, hb = assignment[a], assignment[b]
                if not (edge_avail[ha] & (1 << hb)):
                    return False
            # Commit
            for (a, b) in edges:
                ha, hb = assignment[a], assignment[b]
                edge_avail[ha] &= ~(1 << hb)
                edge_avail[hb] &= ~(1 << ha)
            result[tree_idx] = list(assignment)
            return True

        # Choose next tree vertex to assign
        # Use degree-based heuristic: assign high-degree vertices first
        # But we go in order for simplicity; use MRV on host choices
        tv = depth  # tree vertex index
        
        # Find candidate host vertices
        used = set(assignment[:depth])
        candidates = []
        for h in range(N):
            if h in used:
                continue
            # Check: can this vertex connect to already-assigned neighbors?
            # Get neighbors of tv in the tree
            tree_edges = _TREES[tree_idx]
            neighbors_assigned = []
            for (a, b) in tree_edges:
                if a == tv and b < depth:
                    neighbors_assigned.append(assignment[b])
                elif b == tv and a < depth:
                    neighbors_assigned.append(assignment[a])
            
            # All assigned neighbors must have edge available to h
            feasible = True
            for nh in neighbors_assigned:
                if not (edge_avail[nh] & (1 << h)):
                    feasible = False
                    break
            if not feasible:
                continue
            # Count remaining available edges from h (heuristic: prefer vertices with more options)
            # Actually prefer vertices that are "tighter" - fewer available edges to unassigned
            candidates.append(h)

        if not candidates:
            return False

        # Sort candidates: prefer those with fewer available edges (MRV-like)
        # Actually for packing, prefer vertices that are less "popular" to save popular ones
        # Let's just try in order but with a small heuristic
        # Sort by number of available edges (ascending = more constrained first)
        candidates.sort(key=lambda h: bin(edge_avail[h]).count('1'))

        for h in candidates:
            # Check all already-assigned neighbors have edge to h
            tree_edges = _TREES[tree_idx]
            ok = True
            for (a, b) in tree_edges:
                if a == tv and b < depth:
                    if not (edge_avail[assignment[b]] & (1 << h)):
                        ok = False
                        break
                elif b == tv and a < depth:
                    if not (edge_avail[assignment[a]] & (1 << h)):
                        ok = False
                        break
            if not ok:
                continue

            # Also check: if both endpoints are assigned, edge must be free
            # (already checked above for assigned neighbors)

            assignment[depth] = h
            if backtrack(tree_idx, assignment, depth + 1):
                return True
            assignment[depth] = -1

        return False

    # Process trees in order (largest first)
    success_count = 0
    for tree_idx in order:
        if time.time() > deadline:
            break
        k = tree_idx + 1
        if k == 1:
            # Single vertex, no edges - just pick any unused... actually any vertex
            # But we need distinct vertices within the tree; with 1 vertex, pick vertex 0
            result[tree_idx] = [0]
            success_count += 1
            continue
        if k == 2:
            # Two vertices, one edge
            # Find any available edge
            found = False
            for u in range(N):
                for v in range(u + 1, N):
                    if (edge_avail[u] & (1 << v)) and (edge_avail[v] & (1 << u)):
                        edge_avail[u] &= ~(1 << v)
                        edge_avail[v] &= ~(1 << u)
                        result[tree_idx] = [u, v]
                        found = True
                        break
                if found:
                    break
            if found:
                success_count += 1
            continue

        # For k >= 3, use backtracking
        assignment = [-1] * k
        if backtrack(tree_idx, assignment, 0):
            success_count += 1

    # Convert to proper types
    final_result = []
    for i in range(20):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in result[i]])
    return final_result
