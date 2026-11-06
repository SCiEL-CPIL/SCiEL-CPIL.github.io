import time
import random
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    def merge_paths(paths):
        """Merge paths that share an endpoint to reduce count."""
        if not paths:
            return []
        
        # Represent paths as frozensets of edges for quick lookup? No, we need vertex sequence.
        # We'll use a greedy merging approach.
        # To make it faster, we can try to merge paths repeatedly.
        
        changed = True
        while changed:
            changed = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                p_i = paths[i]
                if len(p_i) < 2:
                    new_paths.append(p_i)
                    continue
                    
                merged = False
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    p_j = paths[j]
                    
                    # Try to merge p_i and p_j
                    # Endpoints: p_i[0], p_i[-1], p_j[0], p_j[-1]
                    
                    # Case 1: p_i end == p_j start
                    if p_i[-1] == p_j[0]:
                        candidate = p_i + p_j[1:]
                        if len(candidate) == len(set(candidate)):
                            new_paths.append(candidate)
                            used[i] = True
                            used[j] = True
                            changed = True
                            merged = True
                            break
                    
                    # Case 2: p_i end == p_j end
                    if not merged and p_i[-1] == p_j[-1]:
                        candidate = p_i + list(reversed(p_j[1:]))
                        if len(candidate) == len(set(candidate)):
                            new_paths.append(candidate)
                            used[i] = True
                            used[j] = True
                            changed = True
                            merged = True
                            break
                    
                    # Case 3: p_i start == p_j start
                    if not merged and p_i[0] == p_j[0]:
                        candidate = list(reversed(p_i[1:])) + p_j
                        if len(candidate) == len(set(candidate)):
                            new_paths.append(candidate)
                            used[i] = True
                            used[j] = True
                            changed = True
                            merged = True
                            break
                    
                    # Case 4: p_i start == p_j end
                    if not merged and p_i[0] == p_j[-1]:
                        candidate = p_i[1:] + list(reversed(p_j))
                        if len(candidate) == len(set(candidate)):
                            new_paths.append(candidate)
                            used[i] = True
                            used[j] = True
                            changed = True
                            merged = True
                            break
                
                if not used[i]:
                    new_paths.append(p_i)
            
            paths = new_paths
            
        return paths

    def decompose_graph(edges, n):
        """Decompose graph into minimum number of edge-disjoint simple paths."""
        if not edges:
            return []
            
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_paths = None
        best_count = float('inf')
        
        # Heuristic: Greedy with randomization and beam search for longest path
        # We will try multiple restarts.
        
        # Determine number of restarts based on graph size and time
        # n=20: ~40 edges, n=25: ~60 edges, n=30: ~85 edges
        # Beam search is expensive, so keep restarts low but effective.
        
        max_restarts = 50
        
        for r in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
            if best_count == 1:
                break
                
            curr_edges = set(edges)
            curr_paths = []
            
            rng_local = random.Random(r * 1000 + n)
            
            # Shuffle edge order for randomization? No, we pick start nodes.
            
            while curr_edges:
                if time.time() > deadline - 5.0:
                    # Fallback: just add remaining edges as individual paths
                    for e in curr_edges:
                        curr_paths.append(list(e))
                    break
                
                # Find a good starting node
                # Prefer nodes with high degree in remaining graph
                deg_curr = defaultdict(int)
                for u, v in curr_edges:
                    deg_curr[u] += 1
                    deg_curr[v] += 1
                
                active_nodes = [u for u in deg_curr if deg_curr[u] > 0]
                if not active_nodes:
                    break
                
                # Sort by degree descending, pick top few
                active_nodes.sort(key=lambda x: deg_curr[x], reverse=True)
                
                best_path = None
                best_len = 0
                
                # Try a few starting nodes
                start_candidates = active_nodes[:5]
                # Add some random ones
                if len(active_nodes) > 5:
                    start_candidates += rng_local.sample(active_nodes[5:], min(5, len(active_nodes)-5))
                
                for s in start_candidates:
                    # Beam search for longest path starting at s
                    beam = [([s], {s})]
                    beam_width = 10
                    max_depth = n
                    
                    for d in range(max_depth):
                        new_beam = []
                        for path, visited in beam:
                            u = path[-1]
                            candidates = []
                            for nb in adj[u]:
                                if nb in visited:
                                    continue
                                e = (min(u, nb), max(u, nb))
                                if e in curr_edges:
                                    # Heuristic: prefer neighbors with higher degree in remaining graph
                                    # that are not visited
                                    # Simple score: degree of nb in remaining graph
                                    score = deg_curr.get(nb, 0)
                                    # Add small noise
                                    score += rng_local.random() * 0.1
                                    candidates.append((score, nb))
                            
                            if not candidates:
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_path = path[:]
                                continue
                            
                            candidates.sort(reverse=True)
                            for _, nb in candidates[:beam_width]:
                                new_beam.append((path + [nb], visited | {nb}))
                        
                        if not new_beam:
                            break
                            
                        # Keep top beam_width paths by length
                        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
                        beam = new_beam[:beam_width]
                        
                        if beam and len(beam[0][0]) > best_len:
                            best_len = len(beam[0][0])
                            best_path = beam[0][0][:]
                    
                    if best_path and len(best_path) > 1:
                        # Verify path
                        valid = True
                        for i in range(len(best_path) - 1):
                            u, v = best_path[i], best_path[i+1]
                            e = (min(u, v), max(u, v))
                            if e not in curr_edges:
                                valid = False
                                break
                        if valid:
                            curr_paths.append(best_path)
                            for i in range(len(best_path) - 1):
                                u, v = best_path[i], best_path[i+1]
                                e = (min(u, v), max(u, v))
                                curr_edges.discard(e)
                            break # Found a good path, restart loop
                            
                if not best_path or len(best_path) <= 1:
                    # Fallback: pick any edge
                    e = next(iter(curr_edges))
                    curr_paths.append(list(e))
                    curr_edges.discard(e)
            
            if curr_paths:
                merged_paths = merge_paths(curr_paths)
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
        if best_paths is None:
            # Fallback: each edge is a path
            best_paths = [list(e) for e in edges]
            
        return best_paths

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_graph(edges, n)
        all_results.append(paths)
        
        # Check time
        if time.time() > deadline - 5.0:
            # If we run out of time, we still need to return something valid
            # For remaining graphs, just use trivial decomposition
            for remaining_spec in specs[specs.index(spec)+1:]:
                rn, rp, rseed = remaining_spec
                redges = instance(rn, rp, rseed)
                all_results.append([list(e) for e in redges])
            break

    return all_results
