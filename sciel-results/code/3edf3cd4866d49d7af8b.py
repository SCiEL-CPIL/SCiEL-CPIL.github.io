"""
Path decomposition minimizing the number of paths.

Strategy:
1. Generate the three fixed graphs.
2. For each graph:
   a. Find a spanning tree and do a DFS to create an Eulerian-like traversal.
   b. Use a greedy merging approach: repeatedly merge two paths that can be
      connected via a shared vertex, reducing the path count.
   c. Use a more sophisticated approach: model as a minimum path cover problem
      on a derived structure, or use iterative improvement.

Better approach:
- The minimum number of paths in a path decomposition is related to the
  number of odd-degree vertices. Specifically, at least ceil(#odd/2) paths.
- We can use the following algorithm:
  1. Find all odd-degree vertices.
  2. Pair them up (using a matching or greedy).
  3. For each pair, find a path between them.
  4. The union of these paths plus any remaining Eulerian components gives
     the decomposition.

Actually, a cleaner approach:
- The minimum path decomposition of a graph has size at least ceil(k/2) where
  k is the number of odd-degree vertices.
- We can achieve this bound by:
  1. Finding a maximum matching on the odd-degree vertices (in the graph).
  2. Adding virtual edges to make all degrees even.
  3. Finding an Eulerian circuit in the augmented graph.
  4. Removing the virtual edges to get the path decomposition.

But we need the paths to be simple and use only original edges. Let's think differently.

Alternative: Use a greedy algorithm that:
1. Starts with each edge as a separate path.
2. Repeatedly merges two paths if they can be connected at a shared endpoint
   without creating a cycle (i.e., the result is still a simple path).

This is essentially the "iterative merging" approach mentioned in the knowledge.

Let me implement this carefully:
- Represent paths as lists of vertices.
- Two paths can be merged if one ends at a vertex where the other begins,
  and the concatenation is still simple (no repeated vertices).
- We want to maximize merges to minimize path count.

For efficiency, we can:
- Maintain a set of used edges.
- For each path, track its endpoints.
- Try to merge paths by checking if endpoint of one matches endpoint of another,
  and the merged path is simple.

Let me code this up.
"""

from random import Random
import time


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []

        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Start with each edge as a path
        paths = [[u, v] for u, v in edges]
        used_edges = set(edges)

        # Greedy merging: repeatedly try to merge two paths
        # Two paths can be merged if:
        #   - path1 ends at vertex x, path2 starts at vertex x (or vice versa)
        #   - The merged path has no repeated vertices
        #   - All edges in the merged path are in the graph (they are, by construction)

        # For efficiency, index paths by their endpoints
        # We'll use a simple iterative approach

        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try all pairs of paths
            n_paths = len(paths)
            for i in range(n_paths):
                if changed:
                    break
                for j in range(i + 1, n_paths):
                    if changed:
                        break
                    p1 = paths[i]
                    p2 = paths[j]

                    # Try merging: p1 then p2, or p2 then p1, with possible reversal
                    for rev1 in [False, True]:
                        for rev2 in [False, True]:
                            a = p1[::-1] if rev1 else p1
                            b = p2[::-1] if rev2 else p2

                            # Check if a ends where b starts
                            if a[-1] == b[0]:
                                merged = a + b[1:]
                                # Check simplicity
                                if len(set(merged)) == len(merged):
                                    # Verify all edges exist
                                    valid = True
                                    for k in range(len(merged) - 1):
                                        e = tuple(sorted((merged[k], merged[k+1])))
                                        if e not in used_edges:
                                            valid = False
                                            break
                                    if valid:
                                        paths[i] = merged
                                        paths[j] = None
                                        changed = True
                                        break
                        if changed:
                            break
                    if changed:
                        break

            paths = [p for p in paths if p is not None]

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() >= deadline:
            # Fallback: return single-edge paths
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
