"""Combined approach: randomized greedy longest-path with Warnsdorff + aggressive merging."""
import time
import sys
from collections import defaultdict, deque
import heapq
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != all_edges:
            return False
        return True

    def greedy_trail_decomposition(edges, adj, n, rng, max_time=None):
        """
        Decompose graph into edge-disjoint simple paths using greedy longest-path search.
        Repeatedly find a long simple path in the remaining graph and remove its edges.
        Uses Warnsdorff-like heuristic: prefer neighbors with fewer remaining options.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Build adjacency for remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        iteration = 0
        while remaining:
            if max_time is not None and time.time() > max_time:
                break
            
            iteration += 1
            
            # Find vertices with highest degree in remaining graph for diverse starts
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if neigh}
            if not degs:
                break
            
            # Try multiple starting points for diversity
            num_starts = min(12, len(degs))
            candidates = sorted(degs.keys(), key=lambda v: (-degs[v], rng.random()))[:num_starts]
            
            best_path = None
            best_len = 0
            
            for cand_start in candidates:
                if not remaining:
                    break
                
                # Greedy walk with Warnsdorff heuristic to find a long simple path
                path = [cand_start]
                visited_set = {cand_start}
                current = cand_start
                
                while True:
                    extended = False
                    # Get neighbors with remaining edges
                    neighbors = list(rem_adj[current])
                    if not neighbors:
                        break
                    
                    # Sort by Warnsdorff: prefer neighbor with fewest remaining unvisited neighbors
                    # This tends to produce longer paths by not trapping ourselves
                    def warnsdorff_score(nb):
                        if nb in visited_set:
                            return float('inf')
                        # Count unvisited neighbors of nb in remaining graph
                        unvisited_count = sum(1 for x in rem_adj[nb] if x not in visited_set)
                        return unvisited_count
                    
                    neighbors.sort(key=warnsdorff_score)
                    
                    for nb in neighbors:
                        e = tuple(sorted((current, nb)))
                        if e in remaining and nb not in visited_set:
                            path.append(nb)
                            visited_set.add(nb)
                            current = nb
                            extended = True
                            break
                    
                    if not extended:
                        break
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
            
            if best_path is None or len(best_path) < 2:
                # Fallback: pick any edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
                continue
            
            # Add the path and remove its edges
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                if e in remaining:
                    remaining.remove(e)
                    rem_adj[e[0]].discard(e[1])
                    rem_adj[e[1]].discard(e[0])
        
        # If any edges remain (due to time limit), add them as single-edge paths
        if remaining:
            for e in remaining:
                paths.append([e[0], e[1]])
        
        return paths

    def merge_paths(paths, edges, n):
        """
        Try to merge pairs of paths to reduce total count.
        Two paths can be merged if they share an endpoint vertex and the merged path is simple.
        Also try merging via an unused edge connecting endpoints.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        max_rounds = 200
        round_num = 0
        while improved and round_num < max_rounds:
            improved = False
            round_num += 1
            
            # Build endpoint map
            endpoint_map = defaultdict(list)
            for i, p in enumerate(current):
                if len(p) >= 2:
                    endpoint_map[p[0]].append((i, 'head'))
                    endpoint_map[p[-1]].append((i, 'tail'))
            
            # Try to merge two paths at a shared endpoint
            for v in list(endpoint_map.keys()):
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue
                
                for i in range(len(entries)):
                    idx1, role1 = entries[i]
                    for j in range(i+1, len(entries)):
                        idx2, role2 = entries[j]
                        if idx1 == idx2:
                            continue
                        
                        p1 = current[idx1]
                        p2 = current[idx2]
                        
                        new_path = None
                        if role1 == 'tail' and role2 == 'head':
                            if not any(x in p1 for x in p2[1:]):
                                new_path = p1 + p2[1:]
                        elif role1 == 'head' and role2 == 'tail':
                            if not any(x in p2 for x in p1[1:]):
                                new_path = p1[1:] + p2
                        elif role1 == 'tail' and role2 == 'tail':
                            if not any(x in p1 for x in p2[:-1]):
                                new_path = p1 + p2[::-1][1:]
                        elif role1 == 'head' and role2 == 'head':
                            if not any(x in p2 for x in p1[1:]):
                                new_path = p1[::-1][1:] + p2
                        
                        if new_path and len(set(new_path)) == len(new_path):
                            valid = True
                            for k in range(len(new_path)-1):
                                e = tuple(sorted((new_path[k], new_path[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_current = []
                                for idx, p in enumerate(current):
                                    if idx != idx1 and idx != idx2:
                                        new_current.append(p)
                                new_current.append(new_path)
                                
                                if verify_decomposition(new_current, edges, n):
                                    current = new_current
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            
            # Try merging via unused edge connecting endpoints
            if not improved:
                for i in range(len(current)):
                    p1 = current[i]
                    if len(p1) < 2:
                        continue
                    for j in range(len(current)):
                        if i == j or len(current[j]) < 2:
                            continue
                        p2 = current[j]
                        
                        # Try all 4 endpoint combinations
                        for v1 in [p1[0], p1[-1]]:
                            for v2 in [p2[0], p2[-1]]:
                                e = tuple(sorted((v1, v2)))
                                if e not in edge_set:
                                    continue
                                
                                # Check if this edge is already used
                                used = False
                                for path in current:
                                    for k in range(len(path)-1):
                                        if tuple(sorted((path[k], path[k+1]))) == e:
                                            used = True
                                            break
                                    if used:
                                        break
                                if used:
                                    continue
                                
                                # Check vertex disjointness
                                if set(p1) & set(p2):
                                    continue
                                
                                # Build merged path
                                # Determine orientation
                                if v1 == p1[-1] and v2 == p2[0]:
                                    new_path = p1 + p2
                                elif v1 == p1[0] and v2 == p2[-1]:
                                    new_path = p1[::-1] + p2
                                elif v1 == p1[-1] and v2 == p2[-1]:
                                    new_path = p1 + p2[::-1]
                                elif v1 == p1[0] and v2 == p2[0]:
                                    new_path = p1[::-1] + p2
                                else:
                                    continue
                                
                                if len(set(new_path)) != len(new_path):
                                    continue
                                
                                valid = True
                                for k in range(len(new_path)-1):
                                    e2 = tuple(sorted((new_path[k], new_path[k+1])))
                                    if e2 not in edge_set:
                                        valid = False
                                        break
                                if valid:
                                    new_current = []
                                    for idx, p in enumerate(current):
                                        if idx != i and idx != j:
                                            new_current.append(p)
                                    new_current.append(new_path)
                                    
                                    if verify_decomposition(new_current, edges, n):
                                        current = new_current
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return current

    def local_search_improve(paths, edges, n, rng, time_limit):
        """
        Local search: try to swap segments between paths to enable further merging.
        Specifically, try to rearrange paths so that more endpoints coincide.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try: for each pair of paths, see if reversing one helps with merging
            for i in range(len(current)):
                p1 = current[i]
                if len(p1) < 2:
                    continue
                for j in range(len(current)):
                    if i == j or len(current[j]) < 2:
                        continue
                    p2 = current[j]
                    
                    # Try reversing p1
                    rev_p1 = p1[::-1]
                    
                    # Check if reversed p1 can merge with p2
                    for v1 in [rev_p1[0], rev_p1[-1]]:
                        for v2 in [p2[0], p2[-1]]:
                            if v1 == v2:
                                # Shared endpoint after reversal
                                new_path = None
                                if rev_p1[-1] == p2[0]:
                                    if not any(x in rev_p1 for x in p2[1:]):
                                        new_path = rev_p1 + p2[1:]
                                elif rev_p1[0] == p2[-1]:
                                    if not any(x in p2 for x in rev_p1[1:]):
                                        new_path = rev_p1[1:] + p2
                                elif rev_p1[-1] == p2[-1]:
                                    if not any(x in rev_p1 for x in p2[:-1]):
                                        new_path = rev_p1 + p2[::-1][1:]
                                elif rev_p1[0] == p2[0]:
                                    if not any(x in p2 for x in rev_p1[1:]):
                                        new_path = rev_p1[::-1][1:] + p2
                                
                                if new_path and len(set(new_path)) == len(new_path):
                                    valid = True
                                    for k in range(len(new_path)-1):
                                        e = tuple(sorted((new_path[k], new_path[k+1])))
                                        if e not in edge_set:
                                            valid = False
                                            break
                                    if valid:
                                        new_current = []
                                        for idx, p in enumerate(current):
                                            if idx != i and idx != j:
                                                new_current.append(p)
                                        new_current.append(new_path)
                                        if verify_decomposition(new_current, edges, n):
                                            current = new_current
                                            improved = True
                                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current

    def solve_one(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        rng = Random(seed * 1000 + 7)
        
        # Phase 1: Multiple randomized greedy decompositions, keep the best
        best_paths = None
        best_count = float('inf')
        
        num_restarts = 8
        per_restart_time = (time_limit - 10) / num_restarts
        
        for r in range(num_restarts):
            restart_time = time.time() + per_restart_time
            paths = greedy_trail_decomposition(edges, adj, n, rng, max_time=restart_time)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
            
            if time.time() > time_limit - 15:
                break
        
        if best_paths is None:
            # Fallback: each edge is its own path
            best_paths = [list(e) for e in edges]
        
        # Phase 2: Merge improvement
        if time.time() < time_limit - 5:
            best_paths = merge_paths(best_paths, edges, n)
        
        # Phase 3: Local search
        if time.time() < time_limit - 2:
            best_paths = local_search_improve(best_paths, edges, n, rng, time_limit - 2)
        
        # Final verification
        if not verify_decomposition(best_paths, edges, n):
            # Fallback: each edge is its own path
            best_paths = [list(e) for e in edges]
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        n, p, seed = spec
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            # Not enough time, use fallback
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        paths = solve_one(n, p, seed, time.time() + remaining_time - 5)
        results.append(paths)

    return results
