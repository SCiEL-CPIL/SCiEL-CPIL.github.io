import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_euler_trail(adj, start=None):
        # Hierholzer's algorithm
        if not adj:
            return []
        if start is None:
            # Start at odd-degree vertex if exists, else any vertex
            odd = [v for v, nbrs in adj.items() if len(nbrs) % 2 == 1]
            start = odd[0] if odd else next(iter(adj))
        
        stack = [start]
        trail = []
        adj = {v: list(nbrs) for v, nbrs in adj.items()}
        
        while stack:
            v = stack[-1]
            if adj.get(v) and len(adj[v]) > 0:
                u = adj[v].pop()
                stack.append(u)
                if adj.get(u):
                    adj[u].remove(v)
            else:
                trail.append(stack.pop())
        
        trail.reverse()
        return trail

    def eulerian_decompose(edges):
        """Decompose an Eulerian graph (all even degree) into cycles, then merge into paths."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if not adj:
            return []
        
        # Find all cycles
        cycles = []
        remaining = {u: list(nbrs) for u, nbrs in adj.items()}
        
        while any(len(nbrs) > 0 for nbrs in remaining.values()):
            # Find a vertex with remaining edges
            start = None
            for v in remaining:
                if len(remaining[v]) > 0:
                    start = v
                    break
            if start is None:
                break
            
            # Find a cycle using Hierholzer
            stack = [start]
            cycle = []
            while stack:
                v = stack[-1]
                if len(remaining[v]) > 0:
                    u = remaining[v].pop()
                    stack.append(u)
                    if len(remaining.get(u, [])) > 0:
                        remaining[u].remove(v)
                else:
                    cycle.append(stack.pop())
            
            cycle.reverse()
            # Remove the closing edge (first==last)
            if len(cycle) > 1 and cycle[0] == cycle[-1]:
                cycle = cycle[:-1]
            if len(cycle) >= 2:
                cycles.append(cycle)
        
        if not cycles:
            return []
        
        # Merge cycles that share vertices into paths
        # Start with all cycles as separate paths
        paths = [list(c) for c in cycles]
        
        # Try to merge: if cycle A has vertex v and cycle B has vertex v,
        # we can splice them
        changed = True
        while changed:
            changed = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    vi = set(paths[i])
                    vj = set(paths[j])
                    shared = vi & vj
                    if not shared:
                        continue
                    # Try to merge at a shared vertex
                    for v in shared:
                        # Find v in path i
                        idx_i = paths[i].index(v)
                        idx_j = paths[j].index(v)
                        
                        # Path i: ... a, v, b ...
                        # Path j: ... c, v, d ...
                        # Merge: ... a, v, d ... + ... c, v, b ...
                        # or: ... a, v, c ... + ... d, v, b ...
                        
                        pi = paths[i]
                        pj = paths[j]
                        
                        # Split pi at v
                        pi_before = pi[:idx_i + 1]  # includes v
                        pi_after = pi[idx_i + 1:]
                        
                        # Split pj at v
                        pj_before = pj[:idx_j + 1]  # includes v
                        pj_after = pj[idx_j + 1:]
                        
                        # Option 1: pi_before + pj_after + [v] + pj_before[1:] + pi_after
                        # Actually let's think more carefully
                        # pi: [..., x, v, y, ...]
                        # pj: [..., w, v, z, ...]
                        # Merged: [..., x, v, z, ...] and [..., w, v, y, ...]
                        # That gives 2 paths, not 1.
                        
                        # To get 1 path, we need to splice:
                        # [..., x, v, z, ..., w, v, y, ...] -- but v appears twice!
                        # So we can't simply merge two cycles sharing one vertex into one path.
                        
                        # We CAN merge if we break at v:
                        # Path 1: [..., x, v, z, ..., w] (from pi_before + pj_after)
                        # Path 2: [v, y, ...] + [w, v] -- no, this is getting complex.
                        
                        # Actually, the correct merge is:
                        # Take pi and pj, both contain v.
                        # pi = A + [v] + B  (where A and B are sequences)
                        # pj = C + [v] + D
                        # New path 1: A + [v] + D + C  (reversed) -- wait no
                        # A + [v] + D  gives path from start of A to end of D
                        # C + [v] + B  gives path from start of C to end of B
                        # That's still 2 paths.
                        
                        # The trick: if we have cycles, we can do:
                        # A + [v] + D + reverse(C)  -- but reverse(C) ends at v
                        # So: A + [v] + D + reverse(C) ends at v, and we need to connect to B
                        # This only works if D connects to reverse(C) properly.
                        
                        # Let me just try: merge by splicing
                        # pi: p0, p1, ..., pk=v, pk+1, ..., pm
                        # pj: q0, q1, ..., ql=v, ql+1, ..., qn
                        # Spliced: p0...pk, ql+1...qn, qn-1...ql+1... wait
                        
                        # Simplest: if pi and pj share vertex v, we can create:
                        # path1 = pi[:idx_i+1] + pj[idx_j+1:]  (ends where pj ends)
                        # path2 = pj[:idx_j+1] + pi[idx_i+1:]  (ends where pi ends)
                        # But path1 and path2 may not be simple (v appears in both)
                        # Actually v appears once in each, so they are simple.
                        # But we went from 2 paths to 2 paths. No improvement.
                        
                        # The real gain comes from: if a cycle shares a vertex with a path,
                        # we can absorb the cycle into the path.
                        pass
                    break
                if changed:
                    break
            # If no merges helped, just return cycles as paths (each cycle becomes a path by removing one vertex)
        
        # Convert cycles to paths by just removing the last vertex (since it's a cycle)
        # Actually each cycle c0,c1,...,ck,c0 is already represented as c0,c1,...,ck
        # which is a valid path using edges c0-c1, c1-c2, ..., ck-1-ck
        # But we're missing the edge ck-c0. So we need to include it.
        # Let me redo: represent cycle as the full sequence including the return
        return paths

    def solve_graph(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        edge_set = set(tuple(sorted(e)) for e in edges)
        num_edges = len(edge_set)
        
        # Compute degrees
        deg = defaultdict(int)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        
        odd_vertices = [v for v in deg if deg[v] % 2 == 1]
        num_odd = len(odd_vertices)
        
        # Lower bound: max(1, num_odd // 2)
        lower_bound = max(1, num_odd // 2) if num_edges > 0 else 0
        
        # Strategy: pair up odd vertices, add virtual edges between pairs,
        # find Eulerian trail in the augmented graph, then remove virtual edges
        # to get the path decomposition.
        
        # For each pairing of odd vertices, we create a multigraph where
        # we add edges between paired odd vertices. This makes all degrees even.
        # Then we find an Eulerian circuit, and the virtual edges tell us
        # where to split into paths.
        
        # Number of pairs = num_odd / 2
        # We need to find the best pairing.
        
        # If num_odd == 0, the graph is Eulerian, so 1 path (Eulerian circuit)
        if num_odd == 0:
            adj = defaultdict(list)
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
            trail = find_euler_trail(adj)
            if trail and len(trail) >= 2:
                return [trail]
            return [list(e) for e in edge_set]
        
        # For odd vertices, try different pairings
        # Start with a greedy pairing
        rng = random.Random(seed)
        
        def try_pairing(pairs):
            """Given pairs of odd vertices to connect, find path decomposition."""
            # Create augmented adjacency
            adj = defaultdict(list)
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
            
            # Add virtual edges
            virtual_edges = []
            for a, b in pairs:
                adj[a].append(b)
                adj[b].append(a)
                virtual_edges.append((a, b))
            
            # Find Eulerian circuit
            trail = find_euler_trail(adj)
            if not trail or len(trail) < 2:
                return None
            
            # Now split the trail at virtual edges
            # A virtual edge (a,b) in the trail means we split the path there
            # The trail is a sequence of vertices. Each consecutive pair is either
            # a real edge or a virtual edge.
            
            # Identify which transitions are virtual
            virtual_set = set()
            for a, b in virtual_edges:
                virtual_set.add((a, b))
                virtual_set.add((b, a))
            
            # Split trail into segments
            segments = []
            current = [trail[0]]
            for i in range(len(trail) - 1):
                u, v = trail[i], trail[i + 1]
                current.append(v)
                if (u, v) in virtual_set:
                    if len(current) >= 2:
                        segments.append(current)
                    current = [v]
            if len(current) >= 2:
                segments.append(current)
            
            if not segments:
                return None
            
            # Verify all segments use only real edges
            valid = True
            for seg in segments:
                for i in range(len(seg) - 1):
                    e = tuple(sorted((seg[i], seg[i + 1])))
                    if e not in edge_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                return None
            
            # Verify all real edges are covered
            covered = set()
            for seg in segments:
                for i in range(len(seg) - 1):
                    e = tuple(sorted((seg[i], seg[i + 1])))
                    covered.add(e)
            
            if covered != edge_set:
                return None
            
            # Verify simplicity
            for seg in segments:
                if len(set(seg)) != len(seg):
                    return None
            
            return segments
        
        # Try greedy pairing first
        def greedy_pairing(odd_verts, rng):
            verts = list(odd_verts)
            rng.shuffle(verts)
            pairs = []
            while len(verts) >= 2:
                a = verts.pop(0)
                b = verts.pop(0)
                pairs.append((a, b))
            return pairs
        
        best_result = None
        best_count = float('inf')
        
        # Try multiple random pairings
        max_tries = 500
        for attempt in range(max_tries):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                pairs = greedy_pairing(odd_vertices, rng)
            else:
                # Random permutation of odd vertices, pair consecutively
                perm = list(odd_vertices)
                rng.shuffle(perm)
                pairs = [(perm[i], perm[i + 1]) for i in range(0, num_odd, 2)]
            
            result = try_pairing(pairs)
            if result is not None:
                if len(result) < best_count:
                    best_count = len(result)
                    best_result = result
                    if best_count == lower_bound:
                        break
        
        if best_result is not None:
            return best_result
        
        # Fallback: greedy decomposition
        remaining = set(edge_set)
        paths = []
        while remaining:
            if time.time() - start_time > time_limit:
                break
            # Find longest path greedily
            adj_rem = defaultdict(set)
            for u, v in remaining:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            if not adj_rem:
                break
            
            # Start from highest degree vertex
            start = max(adj_rem.keys(), key=lambda x: len(adj_rem[x]))
            path = [start]
            visited = {start}
            current = start
            while True:
                neighbors = [nb for nb in adj_rem.get(current, []) if nb not in visited]
                if not neighbors:
                    break
                # Warnsdorff
                neighbors.sort(key=lambda nb: len([x for x in adj_rem.get(nb, []) if x not in visited]))
                current = neighbors[0]
                path.append(current)
                visited.add(current)
            
            if len(path) >= 2:
                paths.append(path)
                for i in range(len(path) - 1):
                    remaining.discard(tuple(sorted((path[i], path[i + 1]))))
            else:
                # Single edge
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.discard(e)
        
        for e in remaining:
            paths.append(list(e))
        
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for n, p, seed in specs:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        time_limit_for_graph = deadline - elapsed - 5
        paths = solve_graph(n, p, seed, time_limit_for_graph)
        results.append(paths)
    
    return results
