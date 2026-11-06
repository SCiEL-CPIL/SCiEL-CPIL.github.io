"""Path decomposition of random connected graphs via edge-disjoint path cover."""
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, time_limit=40.0):
    """
    Decompose a set of edges into a minimum (or near-minimum) number of
    edge-disjoint paths that cover all edges.
    
    Strategy:
    1. Start with each edge as a separate path.
    2. Repeatedly try to merge pairs of paths if the endpoint of one is 
       adjacent to the endpoint of the other (in the remaining graph sense,
       i.e., the connecting edge exists and is not yet used).
    3. Use a greedy approach with randomized restarts to find good merges.
    
    A path is represented as a list of vertices [v0, v1, ..., vk].
    The edges used are consecutive pairs.
    """
    # Build adjacency for quick lookup
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set(edges)
    
    def try_merge(paths, used_edges, max_iters=50000):
        """
        Greedily merge paths. paths is a list of paths (lists of vertices).
        used_edges is a set of frozenset edges already in paths.
        Returns the merged list of paths.
        """
        # We'll maintain a list of paths. Each path has two endpoints.
        # We try to find a pair of paths (p1, p2) such that an endpoint of p1
        # is connected to an endpoint of p2 via an edge not in used_edges.
        # When we merge, we add that connecting edge to used_edges.
        
        # For efficiency, we track which edges are used.
        # Actually, all edges in the original edge_set must be covered.
        # Let's track remaining edges.
        
        remaining = set(edge_set)
        for p in paths:
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                remaining.discard(e)
        
        # Now we have a set of remaining edges. We want to extend paths.
        # A path can be extended at either end if there's a remaining edge
        # from that endpoint to some vertex, and that vertex is an endpoint
        # of another path (or we create a new path, but we want to minimize count).
        
        # Actually, the standard approach: we have a set of paths. We want to
        # reduce the number of paths. Two paths can be merged if an endpoint
        # of one is connected to an endpoint of the other by an edge that is
        # in the remaining set (not yet used in any path).
        
        # Let's represent each path by its two endpoints.
        # path_endpoints[i] = (start, end) for path i
        
        # We'll use a different approach: iterative merging.
        # Start with all edges as paths of length 1.
        # Then repeatedly:
        #   Find two paths whose endpoints are connected by a remaining edge.
        #   Merge them, removing that edge from remaining.
        #   Stop when no more merges are possible.
        
        # But this is what we're doing. Let's implement it carefully.
        
        # paths: list of lists of vertices
        # remaining: set of edges not yet in any path
        
        # For each path, we know its two endpoints.
        # We want to find i, j, and an endpoint a of path i and endpoint b of path j
        # such that (a, b) is in remaining.
        
        # To make this efficient, for each vertex, keep track of which paths
        # have that vertex as an endpoint.
        
        endpoint_paths = defaultdict(list)  # vertex -> list of (path_idx, is_start)
        
        for idx, p in enumerate(paths):
            endpoint_paths[p[0]].append((idx, 0))  # start
            endpoint_paths[p[-1]].append((idx, 1))  # end
        
        merged = True
        while merged:
            merged = False
            # Try to find a merge
            for a, b in sorted(remaining):  # deterministic order for reproducibility
                if a in endpoint_paths and b in endpoint_paths:
                    # Find paths that have a and b as endpoints
                    paths_a = endpoint_paths[a]
                    paths_b = endpoint_paths[b]
                    for (ia, sa) in paths_a:
                        for (ib, sb) in paths_b:
                            if ia == ib:
                                continue
                            # Merge path ia and path ib using edge (a, b)
                            pa = paths[ia]
                            pb = paths[ib]
                            
                            # Determine orientation
                            # We need to connect endpoint a of pa to endpoint b of pb
                            # or endpoint b of pa to endpoint a of pb, etc.
                            
                            # Case 1: a is start of pa, b is start of pb
                            #   pa: a -> ... -> end_a
                            #   pb: b -> ... -> end_b
                            #   Merge: a -> ... -> end_a ... but we need a-b edge
                            #   Actually, if a is start of pa, then pa starts at a.
                            #   If b is start of pb, then pb starts at b.
                            #   We can do: reverse pa so it ends at a, then a-b, then pb
                            #   Or: keep pa as is, but we need to go from a to b.
                            #   If a is start of pa, we can reverse pa: end_a -> ... -> a
                            #   Then a-b, then pb: b -> ... -> end_b
                            #   Result: end_a -> ... -> a -> b -> ... -> end_b
                            
                            # Let's just try all 4 combinations of orientations
                            # and pick one that works (i.e., the edge (a,b) connects
                            # the chosen endpoints).
                            
                            # Actually, since (a,b) is in remaining, and a is an endpoint
                            # of pa, b is an endpoint of pb, we just need to orient
                            # pa so that a is at the "connecting" end, and pb so that
                            # b is at the "connecting" end.
                            
                            # If a is start of pa (sa=0), we reverse pa so a is at end.
                            # If a is end of pa (sa=1), we keep pa as is.
                            # Similarly for b and pb.
                            
                            pa_new = pa[::-1] if sa == 0 else pa
                            pb_new = pb if sb == 0 else pb[::-1]
                            
                            # Now pa_new ends at a, pb_new starts at b.
                            # Merge: pa_new + pb_new, but we need to insert the edge a-b.
                            # pa_new: ... -> a
                            # pb_new: b -> ...
                            # Merged: ... -> a -> b -> ...
                            new_path = pa_new + pb_new
                            
                            # Remove the two old paths and add the new one
                            # We'll mark them for removal
                            del paths[ia]
                            del paths[ib]
                            paths.append(new_path)
                            
                            # Update remaining
                            remaining.discard(tuple(sorted((a, b))))
                            
                            # Update endpoint_paths
                            # Remove old entries
                            endpoint_paths[a] = [(x, y) for (x, y) in endpoint_paths[a] if x not in (ia, ib)]
                            endpoint_paths[b] = [(x, y) for (x, y) in endpoint_paths[b] if x not in (ia, ib)]
                            # But indices have shifted! This is problematic.
                            
                            # Let's restart the endpoint_paths rebuild for simplicity.
                            # This is O(n) per merge, which might be too slow.
                            # Let's just rebuild endpoint_paths from scratch.
                            endpoint_paths = defaultdict(list)
                            for idx2, p2 in enumerate(paths):
                                endpoint_paths[p2[0]].append((idx2, 0))
                                endpoint_paths[p2[-1]].append((idx2, 1))
                            
                            merged = True
                            break
                        if merged:
                            break
                    if merged:
                        break
            
        return paths, remaining
    
    # Initial paths: each edge as a path
    paths = [[u, v] for u, v in edges]
    
    # Run the merging
    paths, remaining = try_merge(paths, None)
    
    # If there are remaining edges, they couldn't be merged. This shouldn't happen
    # if the graph is connected, but let's handle it.
    # Actually, if remaining is non-empty, we need to add them as new paths.
    for e in remaining:
        paths.append(list(e))
    
    return paths


def _decompose_greedy(edges, time_limit=40.0):
    """
    A more careful greedy decomposition.
    """
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Start with each edge as a path
    paths = [[u, v] for u, v in edges]
    
    # Track which edges are used
    used = set(tuple(sorted(e)) for e in edges)
    
    # For each path, we can extend it at either end.
    # We want to merge paths to reduce the count.
    
    # Let's use a priority queue or just iterate.
    # For efficiency, we'll try to merge as much as possible.
    
    start_time = time.time()
    
    # Repeatedly try to merge
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        # For each pair of paths, check if they can be merged
        n_paths = len(paths)
        for i in range(n_paths):
            if changed:
                break
            pi = paths[i]
            # Endpoints of pi
            start_i, end_i = pi[0], pi[-1]
            for j in range(i + 1, n_paths):
                pj = paths[j]
                start_j, end_j = pj[0], pj[-1]
                
                # Try to connect an endpoint of pi to an endpoint of pj
                # Check all 4 combinations
                for (a, b) in [(start_i, start_j), (start_i, end_j), (end_i, start_j), (end_i, end_j)]:
                    e = tuple(sorted((a, b)))
                    if e in used:
                        continue
                    # Check if the edge exists
                    if b not in adj[a]:
                        continue
                    # Merge
                    # Determine orientation
                    # We want to connect a (from pi) to b (from pj)
                    # If a is start_i, we keep pi as is (a is at start)
                    # If a is end_i, we reverse pi so a is at end... wait, we want a to be at the connecting end.
                    
                    # Let's think: we have path pi and path pj.
                    # We want to create a new path that includes both, connected by edge (a, b).
                    # a must be an endpoint of pi, b must be an endpoint of pj.
                    
                    # If a == start_i, then pi is oriented as [start_i, ..., end_i], so a is at position 0.
                    #   We can reverse pi to get [end_i, ..., start_i], so a is at the end.
                    #   Then append pj (oriented so b is at the start).
                    # If a == end_i, then pi is oriented as [start_i, ..., end_i], so a is at the end.
                    #   We keep pi as is.
                    #   Then append pj (oriented so b is at the start).
                    
                    # Similarly for b:
                    # If b == start_j, pj is [start_j, ..., end_j], so b is at position 0.
                    #   We keep pj as is.
                    # If b == end_j, we reverse pj so b is at the end... wait, we want b at the start of the appended part.
                    #   So if b == end_j, we reverse pj to get [end_j, ..., start_j], so b is at position 0.
                    
                    pi_new = pi if a == end_i else pi[::-1]
                    pj_new = pj if b == start_j else pj[::-1]
                    
                    # Now pi_new ends at a, pj_new starts at b.
                    new_path = pi_new + pj_new
                    
                    # Update
                    del paths[i]
                    del paths[j]
                    paths.append(new_path)
                    used.add(e)
                    changed = True
                    break
                if changed:
                    break
    
    return paths


def construct():
    start = time.time()
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        paths = _decompose_greedy(edges, time_limit=40.0)
        results.append(paths)
    return results
