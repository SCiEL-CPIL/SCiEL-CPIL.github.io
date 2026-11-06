"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_greedy(edges, time_limit=30.0):
        """
        Greedy path decomposition: repeatedly extract a simple path using
        a Warnsdorff-like heuristic that favors continuing through vertices
        with low onward options, to avoid trapping edges.
        Multiple restarts with different random seeds for tie-breaking.
        """
        if not edges:
            return []

        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        best_paths = None
        best_count = float('inf')
        start_local = time.time()
        max_restarts = 2000

        for r in range(max_restarts):
            if time.time() - start_local > time_limit:
                break

            rng = random.Random(r * 7919 + 31)
            cur_edges = set(edge_set)
            paths = []

            while cur_edges:
                # Compute remaining degrees
                deg = defaultdict(int)
                for u, v in cur_edges:
                    deg[u] += 1
                    deg[v] += 1

                # Pick start vertex: prefer high degree to consume bottlenecks
                max_d = max(deg.values())
                cands = [v for v, d in deg.items() if d == max_d]
                # Add randomness
                start_v = rng.choice(cands)

                path = [start_v]
                visited = {start_v}
                last = start_v

                while True:
                    # Find valid next vertices
                    opts = []
                    for nb in adj[last]:
                        if nb not in visited:
                            e = (min(last, nb), max(last, nb))
                            if e in cur_edges:
                                # Count onward moves from nb (excluding last)
                                onward = 0
                                for nn in adj[nb]:
                                    if nn != last and nn not in visited:
                                        e2 = (min(nb, nn), max(nb, nn))
                                        if e2 in cur_edges:
                                            onward += 1
                                opts.append((onward, nb))

                    if not opts:
                        break

                    # Warnsdorff: pick neighbor with fewest onward moves
                    opts.sort(key=lambda x: (x[0], x[1]))
                    # Randomize among top-2 to add diversity
                    if len(opts) > 1 and rng.random() < 0.3:
                        next_v = opts[1][1]
                    else:
                        next_v = opts[0][1]

                    path.append(next_v)
                    visited.add(next_v)
                    e = (min(last, next_v), max(last, next_v))
                    cur_edges.discard(e)
                    last = next_v

                if len(path) > 1:
                    paths.append(path)

            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                if best_count == 1:
                    break

        if best_paths is None:
            return [[u, v] for u, v in edges]
        return best_paths

    def solve_sa(edges, time_limit=30.0):
        """
        Simulated annealing over edge assignments to paths.
        Representation: each edge is assigned to a path and a position.
        Moves: reassign an edge to a different path, or swap positions.
        Objective: minimize number of paths (each path must be simple and valid).
        
        This is complex to validate. Instead, use a simpler local search:
        Given a path decomposition, try to merge two paths by finding a
        vertex that appears in both where we can splice them.
        """
        if not edges:
            return []

        # First get a greedy solution
        paths = solve_greedy(edges, time_limit=min(5.0, time_limit * 0.2))
        if paths is None:
            paths = [[u, v] for u, v in edges]

        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        start_local = time.time()
        best_paths = paths[:]
        best_count = len(paths)

        # Local search: try to merge two paths
        # Two paths P1 and P2 can be merged if there exist vertices a in P1, b in P2
        # such that (a,b) is an edge, and the merge maintains simplicity.
        # Actually, we can only merge if we can find an edge (a,b) where a is an
        # endpoint of P1 and b is an endpoint of P2, then concatenate.
        # Or if a is internal in P1 and b is internal in P2, we can't simply merge.
        # The simplest merge: P1 ends at a, P2 starts at b, and (a,b) is an edge.
        # Then P1 + P2[1:] is a valid path if it's simple.

        def try_merge(path1, path2):
            """Try to merge path1 and path2 into a single path."""
            # Check all combinations of endpoints
            # path1 endpoints: path1[0], path1[-1]
            # path2 endpoints: path2[0], path2[-1]
            # We can concatenate if an endpoint of path1 connects to an endpoint of path2
            
            # Option 1: path1 + path2 (if path1[-1] connects to path2[0])
            # Option 2: path1 + reversed(path2) (if path1[-1] connects to path2[-1])
            # Option 3: reversed(path1) + path2 (if path1[0] connects to path2[0])
            # Option 4: reversed(path1) + reversed(path2) (if path1[0] connects to path2[-1])
            
            candidates = [
                (path1, path2),
                (path1, path2[::-1]),
                (path1[::-1], path2),
                (path1[::-1], path2[::-1]),
            ]
            
            for p1, p2 in candidates:
                if len(p1) < 2 or len(p2) < 2:
                    continue
                a = p1[-1]
                b = p2[0]
                e = (min(a, b), max(a, b))
                if e not in edge_set:
                    continue
                # Check if (a,b) is actually used in either path
                # p1 uses edges between consecutive vertices
                # We need (a,b) to NOT be already in p1 or p2
                # p1's last edge is (p1[-2], p1[-1]) = (p1[-2], a)
                # p2's first edge is (p2[0], p2[1]) = (b, p2[1])
                # So (a,b) is not in p1 unless p1[-2]==b, and not in p2 unless p2[1]==a
                if len(p1) >= 2 and p1[-2] == b:
                    continue
                if len(p2) >= 2 and p2[1] == a:
                    continue
                merged = p1 + p2[1:]
                # Check simplicity
                if len(set(merged)) == len(merged):
                    # Check all edges are valid
                    valid = True
                    for i in range(len(merged) - 1):
                        ee = (min(merged[i], merged[i+1]), max(merged[i], merged[i+1]))
                        if ee not in edge_set:
                            valid = False
                            break
                    if valid:
                        return merged
            return None

        # Iterate: try to merge pairs
        improved = True
        while improved and time.time() - start_local < time_limit:
            improved = False
            n_paths = len(best_paths)
            for i in range(n_paths):
                if time.time() - start_local > time_limit:
                    break
                for j in range(i + 1, n_paths):
                    merged = try_merge(best_paths[i], best_paths[j])
                    if merged is not None:
                        # Verify the merge is valid and doesn't create edge conflicts
                        # We need to ensure the merged path uses edges that were in path i or j
                        # and that we're not reusing an edge
                        # Actually, we need to be more careful: the merged path must
                        # use exactly the edges of path i and path j plus the connecting edge
                        # But the connecting edge (a,b) must be in the original graph
                        # and must not have been used in either path
                        
                        # Let's verify by checking edge usage
                        used_in_i = set()
                        for k in range(len(best_paths[i]) - 1):
                            used_in_i.add((min(best_paths[i][k], best_paths[i][k+1]), max(best_paths[i][k], best_paths[i][k+1])))
                        used_in_j = set()
                        for k in range(len(best_paths[j]) - 1):
                            used_in_j.add((min(best_paths[j][k], best_paths[j][k+1]), max(best_paths[j][k], best_paths[j][k+1])))
                        
                        all_used = used_in_i | used_in_j
                        merged_edges = set()
                        for k in range(len(merged) - 1):
                            merged_edges.add((min(merged[k], merged[k+1]), max(merged[k], merged[k+1])))
                        
                        # merged_edges should be all_used + {(a,b)}
                        # But wait, we removed the edge (a,b) from consideration?
                        # Actually, (a,b) is a new edge that was NOT in either path
                        # So merged_edges should have len(all_used) + 1
                        # But that means we're using an extra edge that wasn't in our decomposition
                        # This is wrong! We can only use edges from the original graph,
                        # but each edge must be used exactly once.
                        # The connecting edge (a,b) must be one of the edges in the graph
                        # that was NOT used in path i or path j.
                        # But then we're not covering all edges with the new set of paths.
                        # We'd need to also remove (a,b) from wherever it was.
                        # This is getting complicated. Let me simplify.
                        
                        # Actually, the merge should only work if (a,b) is an edge
                        # that is in the graph but NOT in path i or path j.
                        # In that case, we replace path i and path j with the merged path,
                        # and we've used one more edge. But we need to account for it.
                        # This approach is flawed because it doesn't maintain edge-disjointness
                        # and full coverage.
                        
                        # Let me use a simpler and correct approach:
                        # Just keep the greedy result. The local search merge is tricky.
                        pass
            # Fallback: just return greedy
            break

        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        if time_left < 3:
            paths = solve_greedy(g_edges, time_limit=2.0)
        else:
            # Allocate time proportional to graph size, with a cap
            time_alloc = min(40.0, time_left * 0.3)
            paths = solve_greedy(g_edges, time_limit=time_alloc)
        results.append(paths)

        if time.time() > deadline - 5:
            for rem_spec in specs[i + 1:]:
                rem_edges = instance(*rem_spec)
                results.append(solve_greedy(rem_edges, time_limit=1.0))
            break

    return results
