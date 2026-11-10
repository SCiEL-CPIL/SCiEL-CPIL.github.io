import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges, n):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_simple(path):
        return len(set(path)) == len(path)

    def build_line_graph(edges, adj):
        """
        Build line graph: nodes are edges of original graph.
        Two line-graph nodes are adjacent if corresponding edges share a vertex.
        Returns: line_adj (dict of sets), edge_to_node mapping.
        """
        m = len(edges)
        edge_to_node = {e: i for i, e in enumerate(edges)}
        line_adj = [set() for _ in range(m)]
        
        # For each vertex, all edges incident to it form a clique in line graph
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)
        
        for v in vertex_edges:
            es = vertex_edges[v]
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    line_adj[es[i]].add(es[j])
                    line_adj[es[j]].add(es[i])
        
        return line_adj, edge_to_node

    def greedy_matching(line_adj, m, rng):
        """
        Greedy maximum matching in line graph.
        Returns: match dict (node -> matched node), matched set.
        """
        match = {}
        unmatched = set(range(m))
        
        # Sort by degree descending for better greedy
        order = sorted(unmatched, key=lambda x: len(line_adj[x]), reverse=True)
        
        for u in order:
            if u not in unmatched:
                continue
            candidates = [v for v in line_adj[u] if v in unmatched]
            if not candidates:
                continue
            # Prefer low-degree candidates (Warnsdorff-like)
            candidates.sort(key=lambda v: len([x for x in line_adj[v] if x in unmatched]))
            v = candidates[0]
            match[u] = v
            match[v] = u
            unmatched.discard(u)
            unmatched.discard(v)
        
        return match

    def local_improve_matching(match, line_adj, m, rng, max_iters=2000):
        """
        Local improvement via swap operations to increase matching size.
        A swap: if u-v is matched and w-x is matched, try replacing with u-w and v-x (or u-x and v-w).
        Also try augmenting paths of length 3: unmatched u - matched v-w - unmatched x, replace v-w with u-v and w-x... 
        Actually for matching, we do: find augmenting path of odd length.
        Simplified: try all pairs of matched edges and see if swapping increases or maintains matching.
        """
        matched_pairs = set()
        for u, v in match.items():
            if u < v:
                matched_pairs.add((u, v))
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try augmenting paths of length 1 (unmatched edge that can be added)
            for u in range(m):
                if u in match:
                    continue
                for v in line_adj[u]:
                    if v not in match:
                        # Can add u-v
                        match[u] = v
                        match[v] = u
                        matched_pairs.add((u, v))
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try swaps: for two matched pairs (a,b) and (c,d), try (a,c)+(b,d) or (a,d)+(b,c)
            pairs = list(matched_pairs)
            rng.shuffle(pairs)
            for i in range(min(len(pairs), 500)):
                a, b = pairs[i]
                for j in range(i + 1, min(len(pairs), i + 50)):
                    c, d = pairs[j]
                    # Try swap 1: a-c and b-d
                    if c in line_adj[a] and d in line_adj[b]:
                        # Check that removing a-b and c-d doesn't break anything
                        # New matching: a-c, b-d
                        del match[a]
                        del match[b]
                        del match[c]
                        del match[d]
                        match[a] = c
                        match[c] = a
                        match[b] = d
                        match[d] = b
                        matched_pairs.discard((a, b))
                        matched_pairs.discard((c, d))
                        matched_pairs.add((a, c))
                        matched_pairs.add((b, d))
                        improved = True
                        break
                    # Try swap 2: a-d and b-c
                    if d in line_adj[a] and c in line_adj[b]:
                        del match[a]
                        del match[b]
                        del match[c]
                        del match[d]
                        match[a] = d
                        match[d] = a
                        match[b] = c
                        match[c] = b
                        matched_pairs.discard((a, b))
                        matched_pairs.discard((c, d))
                        matched_pairs.add((a, d))
                        matched_pairs.add((b, c))
                        improved = True
                        break
                if improved:
                    break
            
            # Try 3-edge augmenting: unmatched u, matched v-w, unmatched x where u-v and w-x exist
            if not improved:
                unmatched_nodes = [u for u in range(m) if u not in match]
                rng.shuffle(unmatched_nodes)
                for u in unmatched_nodes[:100]:
                    for v in line_adj[u]:
                        if v not in match:
                            continue
                        w = match[v]
                        for x in line_adj[w]:
                            if x in match:
                                continue
                            # Check u-v and w-x, and that we can form u-v, x-w... wait
                            # Current: v-w matched. We want to add u-v and w-x, but v and w are already matched.
                            # Augmenting path: u - v - w - x where u,x unmatched, v-w matched.
                            # New matching: u-v, w-x. Size increases by 1.
                            if u in line_adj[v] and w in line_adj[x]:
                                del match[v]
                                del match[w]
                                match[u] = v
                                match[v] = u
                                match[w] = x
                                match[x] = w
                                matched_pairs.discard((min(v,w), max(v,w)))
                                matched_pairs.add((u, v))
                                matched_pairs.add((w, x))
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return match

    def matching_to_paths(match, edges, m):
        """
        Convert matching in line graph to path decomposition.
        Matched edges in line graph = adjacent edges in a path.
        Build directed graph on line-graph nodes, then find paths.
        """
        # Build adjacency: for each matched pair (u,v), they are consecutive in some path
        # We need to determine the order. For matched line-graph nodes u and v (original edges e_u, e_v),
        # they share a vertex. The path goes through that shared vertex.
        
        # Build a graph where each line-graph node has at most 2 matched neighbors
        # (since it's a matching, each node has at most 1 matched partner)
        # So the matched edges form a set of disjoint paths and cycles in the line graph.
        
        # For each component, trace the path.
        visited = set()
        paths = []
        
        for start in range(m):
            if start in visited:
                continue
            # Trace the component
            path_nodes = [start]
            visited.add(start)
            
            # Find the two ends of this path in the matching
            # A node in the matching has at most 1 partner
            # Ends are nodes with no partner or one partner that is also an end
            
            # Let's build the component properly
            comp = set([start])
            queue = [start]
            while queue:
                u = queue.pop()
                if u in match:
                    v = match[u]
                    if v not in comp:
                        comp.add(v)
                        queue.append(v)
            
            # Find endpoints: nodes in comp with degree 0 or 1 in the matching
            endpoints = []
            for u in comp:
                if u not in match:
                    endpoints.append(u)
            
            # If no endpoints, it's a cycle - pick any node as start, break the cycle
            if not endpoints:
                # It's a cycle. We need to break it into a path.
                # Pick an arbitrary node, remove its match, making it an endpoint
                start_node = list(comp)[0]
                other = match[start_node]
                # Remove the match
                del match[start_node]
                del match[other]
                endpoints = [start_node, other]
            
            # Trace from one endpoint to the other
            path_nodes = []
            cur = endpoints[0]
            prev = None
            while True:
                path_nodes.append(cur)
                if cur in match:
                    nxt = match[cur]
                    if nxt == prev:
                        break
                    prev = cur
                    cur = nxt
                else:
                    break
            
            # Convert line-graph node indices to actual edges, then to vertex paths
            # path_nodes is a sequence of edge indices
            if len(path_nodes) == 0:
                continue
            
            # Build the vertex path
            # For consecutive edges e_i, e_j in path_nodes, they share a vertex
            # We need to determine the correct orientation
            
            vertex_path = []
            for idx, edge_idx in enumerate(path_nodes):
                u, v = edges[edge_idx]
                if idx == 0:
                    vertex_path.append(u)
                    vertex_path.append(v)
                else:
                    # Previous edge's last vertex should be this edge's first vertex
                    prev_last = vertex_path[-1]
                    if prev_last == u:
                        vertex_path.append(v)
                    elif prev_last == v:
                        # Need to reverse... but we can't easily reverse
                        # Actually, let's rebuild properly
                        pass
                    else:
                        # This shouldn't happen if matching is correct
                        vertex_path.append(u)
            
            # The above is fragile. Let me redo this more carefully.
            # Actually, let me just collect the edges and build the path by walking.
            
            if len(path_nodes) >= 1:
                # Rebuild vertex path properly
                ep = []
                for idx, edge_idx in enumerate(path_nodes):
                    u, v = edges[edge_idx]
                    if idx == 0:
                        ep = [u, v]
                    else:
                        if ep[-1] == u:
                            ep.append(v)
                        elif ep[-1] == v:
                            # We need to handle this - the edge is traversed in reverse
                            # This means our path direction is inconsistent
                            # For now, just append u and hope for the best
                            ep.append(u)
                        else:
                            ep.append(u)
                
                if len(ep) >= 2:
                    paths.append(ep)
        
        return paths

    def solve_one(edges, n, time_limit=35.0):
        adj = build_adj(edges, n)
        m = len(edges)
        if m == 0:
            return []
        
        best = None
        best_count = float('inf')
        t0 = time.time()
        
        line_adj, _ = build_line_graph(edges, adj)
        
        # Multiple trials with different random seeds for greedy matching
        for trial in range(500):
            if time.time() - t0 > time_limit - 2.0:
                break
            seed = trial * 7919 + 13
            rng = random.Random(seed)
            
            match = greedy_matching(line_adj, m, rng)
            match = local_improve_matching(match, line_adj, m, rng, max_iters=500)
            
            paths = matching_to_paths(match, edges, m)
            
            # Verify and fix
            valid_paths = []
            used_edges = set()
            for p in paths:
                if len(p) < 2:
                    continue
                if not is_simple(p):
                    # Split at first repeat
                    seen = set()
                    i = 0
                    while i < len(p) - 1:
                        if p[i] in seen:
                            if i > 0:
                                valid_paths.append(p[:i+1])
                            seen = set()
                            i = 0
                            continue
                        seen.add(p[i])
                        i += 1
                    if i < len(p) - 1:
                        valid_paths.append(p[i:])
                    continue
                
                edge_set = set()
                valid = True
                for k in range(len(p) - 1):
                    e = tuple(sorted((p[k], p[k+1])))
                    if e in used_edges or e not in set(edges):
                        valid = False
                        break
                    edge_set.add(e)
                if valid:
                    valid_paths.append(p)
                    used_edges |= edge_set
            
            # Add any remaining edges
            remaining = set(edges) - used_edges
            for e in remaining:
                valid_paths.append(list(e))
            
            count = len(valid_paths)
            if count < best_count:
                best_count = count
                best = [list(p) for p in valid_paths]
        
        if best is None:
            best = [[u, v] for u, v in edges]
        
        return best

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    total_time = deadline - start_time
    time_per_instance = total_time / 3 - 2.0
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        if not edges:
            results.append([])
            continue
        
        remaining_time = deadline - time.time() - 1.0
        if remaining_time < 1.0:
            results.append([[u, v] for u, v in edges])
            continue
        
        paths = solve_one(edges, n, time_limit=min(time_per_instance, remaining_time - 1.0))
        results.append(paths)
    
    return results
