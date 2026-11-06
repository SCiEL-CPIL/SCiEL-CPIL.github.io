"""Rewritten program using Greedy Path Merging for minimum path edge cover."""
# EVOLVE-BLOCK-START
import numpy as np
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_greedy(n, edge_list, seed):
        if not edge_list:
            return []
        
        rng = Random(seed)
        
        # Initialize paths: each edge is a path
        # Store paths as lists of vertices
        paths = []
        # path_id -> list of vertices
        # Also maintain endpoint mapping: vertex -> set of path_ids
        endpoint_map = {v: set() for v in range(n)}
        
        for u, v in edge_list:
            pid = len(paths)
            paths.append([u, v])
            endpoint_map[u].add(pid)
            endpoint_map[v].add(pid)
        
        active_paths = set(range(len(paths)))
        
        # To avoid infinite loops or getting stuck, we can limit iterations or use a queue
        # We will repeatedly find a vertex with >= 2 endpoints and try to merge
        # We pick vertices in random order to diversify results
        
        # Pre-compute adjacency for quick checks? Not strictly needed for merging, 
        # but we need to check simplicity.
        
        # Helper to check if a path is simple
        def is_simple(path):
            return len(set(path)) == len(path)
            
        # Helper to get the other endpoint of a path
        def get_other_endpoint(path, endpoint):
            if path[0] == endpoint:
                return path[-1]
            elif path[-1] == endpoint:
                return path[0]
            else:
                return None # Should not happen if endpoint_map is correct
                
        # We'll use a list of candidate vertices
        # We can just iterate through all vertices multiple times
        
        max_iter = 10000
        iter_count = 0
        
        while iter_count < max_iter:
            iter_count += 1
            merged_any = False
            
            # Get vertices with at least 2 paths
            candidates = [v for v in range(n) if len(endpoint_map[v]) >= 2]
            
            if not candidates:
                break
                
            # Shuffle candidates to randomize merge order
            rng.shuffle(candidates)
            
            for v in candidates:
                if len(endpoint_map[v]) < 2:
                    continue
                    
                pids = list(endpoint_map[v])
                # Shuffle pids
                rng.shuffle(pids)
                
                # Try to find a pair to merge
                merged = False
                for i in range(len(pids)):
                    for j in range(i + 1, len(pids)):
                        p1_id = pids[i]
                        p2_id = pids[j]
                        
                        if p1_id not in active_paths or p2_id not in active_paths:
                            continue
                            
                        p1 = paths[p1_id]
                        p2 = paths[p2_id]
                        
                        # Determine orientation
                        # p1 must have v as an endpoint
                        if p1[0] == v:
                            p1_start, p1_end = p1[0], p1[-1]
                            p1_oriented = list(reversed(p1)) # Make v the end? 
                            # Actually, let's just build the new path carefully
                            # If p1 starts at v, we want to append p2 to it? 
                            # No, if p1 is (v, ..., a) and p2 is (v, ..., b)
                            # We can merge to (a, ..., v, ..., b) by reversing p1.
                            # Or (b, ..., v, ..., a).
                            # Let's try merging p1 (reversed if needed) with p2 (reversed if needed)
                            # such that v is in the middle.
                            
                            # Case 1: p1 ends at v, p2 starts at v -> p1 + p2[1:]
                            # Case 2: p1 ends at v, p2 ends at v -> p1 + reversed(p2)[1:]
                            # Case 3: p1 starts at v, p2 starts at v -> reversed(p1)[1:] + p2  (wait, reversed(p1) ends at v? No. reversed(p1) starts at a, ends at v. So we want p2_reversed + p1? )
                            # Let's standardize:
                            # We want to form a path that goes through v.
                            # Let p1_rev = reversed(p1) if p1[0] == v else p1. Now p1_rev ends at v.
                            # Let p2_rev = reversed(p2) if p2[0] == v else p2. Now p2_rev ends at v.
                            # We can merge p1_rev and p2_rev by taking p1_rev[:-1] + p2_rev?
                            # No, p1_rev ends at v. p2_rev ends at v.
                            # We want to connect the "tails".
                            # Tail of p1_rev is v. Tail of p2_rev is v.
                            # We can connect the "heads"? No.
                            # We can form: p2_rev_reversed + p1_rev ?
                            # p2_rev_reversed starts at v. p1_rev ends at v.
                            # So p2_rev_reversed + p1_rev[1:]?
                            # Let's just try all 4 combinations of orientations and pick the first simple one.
                            
                            orientations = [
                                (p1, p2),
                                (p1, list(reversed(p2))),
                                (list(reversed(p1)), p2),
                                (list(reversed(p1)), list(reversed(p2)))
                            ]
                            
                            # We need to ensure v is the junction.
                            # In the merged path, v must be internal.
                            # So one path must end at v and the other must start at v.
                            
                            valid_merge = False
                            new_path = None
                            
                            # Check p1 ends at v, p2 starts at v
                            if p1[-1] == v and p2[0] == v:
                                cand = p1 + p2[1:]
                                if is_simple(cand):
                                    new_path = cand
                                    valid_merge = True
                            # Check p1 ends at v, p2 ends at v (reverse p2)
                            if not valid_merge and p1[-1] == v and p2[-1] == v:
                                p2_rev = list(reversed(p2))
                                cand = p1 + p2_rev[1:]
                                if is_simple(cand):
                                    new_path = cand
                                    valid_merge = True
                            # Check p1 starts at v, p2 starts at v (reverse p1)
                            if not valid_merge and p1[0] == v and p2[0] == v:
                                p1_rev = list(reversed(p1))
                                cand = p1_rev + p2[1:]
                                if is_simple(cand):
                                    new_path = cand
                                    valid_merge = True
                            # Check p1 starts at v, p2 ends at v (reverse p1)
                            if not valid_merge and p1[0] == v and p2[-1] == v:
                                p1_rev = list(reversed(p1))
                                cand = p1_rev + p2[1:]
                                if is_simple(cand):
                                    new_path = cand
                                    valid_merge = True
                            
                            if valid_merge:
                                # Perform merge
                                # Remove old paths
                                active_paths.remove(p1_id)
                                active_paths.remove(p2_id)
                                
                                # Determine new endpoints
                                # Old endpoints of p1: p1[0], p1[-1]
                                # Old endpoints of p2: p2[0], p2[-1]
                                # v is removed from endpoint status.
                                # The new endpoints are the "other" ends.
                                
                                # Find the other end of p1
                                if p1[0] == v:
                                    end1 = p1[-1]
                                else:
                                    end1 = p1[0]
                                    
                                if p2[0] == v:
                                    end2 = p2[-1]
                                else:
                                    end2 = p2[0]
                                    
                                # Update endpoint_map
                                # Remove p1_id, p2_id from v
                                endpoint_map[v].remove(p1_id)
                                endpoint_map[v].remove(p2_id)
                                
                                # Remove p1_id from its other endpoint
                                endpoint_map[end1].remove(p1_id)
                                # Remove p2_id from its other endpoint
                                endpoint_map[end2].remove(p2_id)
                                
                                # Add new path
                                new_id = len(paths)
                                paths.append(new_path)
                                active_paths.add(new_id)
                                
                                # Add new_id to its endpoints
                                endpoint_map[end1].add(new_id)
                                endpoint_map[end2].add(new_id)
                                
                                merged_any = True
                                merged = True
                                break
                        if merged:
                            break
                    if merged:
                        break
                
                if merged:
                    break
            
            if not merged_any:
                # No merge found in this pass
                # To avoid infinite loop, we can break or try a different strategy
                # For now, break.
                break
                
        # Extract final paths
        final_paths = [paths[i] for i in active_paths]
        return final_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        # Try multiple random seeds for the greedy process to find a better solution
        best_paths = None
        best_count = float('inf')
        
        for run_seed in range(20):
            paths = solve_greedy(n, edges, seed + run_seed)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                
        results.append(best_paths)
        
    return results
# EVOLVE-BLOCK-END
