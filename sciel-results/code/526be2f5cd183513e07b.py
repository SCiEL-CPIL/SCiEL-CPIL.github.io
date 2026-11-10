import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n, time_limit):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Initial paths: each edge is a path
        # Path is a list of vertices
        # We store paths as (start, end, vertices_tuple) for fast hashing?
        # Or just lists.
        paths = [[u, v] for u, v in edges]
        
        # To speed up merging, we can index paths by their endpoints.
        # But endpoints change.
        
        # Simple greedy merge
        # We try to merge path i and path j if end(i) == start(j) or end(j) == start(i)
        # and vertices are disjoint.
        
        # We will run a loop.
        # To avoid O(N^2) scan every time, we can just scan. N is small.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle paths to avoid bias?
            # random.shuffle(paths)
            
            # Create a list of indices
            indices = list(range(len(paths)))
            random.shuffle(indices)
            
            for i_idx in indices:
                if i_idx >= len(paths):
                    continue
                p1 = paths[i_idx]
                if not p1:
                    continue
                e1 = p1[-1]
                s1 = p1[0]
                v1_set = set(p1)
                
                for j_idx in indices:
                    if j_idx == i_idx or j_idx >= len(paths):
                        continue
                    p2 = paths[j_idx]
                    if not p2:
                        continue
                    s2 = p2[0]
                    e2 = p2[-1]
                    
                    # Check merge conditions
                    # Case 1: e1 == s2
                    if e1 == s2:
                        # Check disjointness
                        if v1_set.isdisjoint(set(p2)):
                            # Merge p1 and p2
                            new_path = p1 + p2[1:]
                            # Remove p2 and replace p1
                            paths[i_idx] = new_path
                            paths.pop(j_idx)
                            improved = True
                            break # Restart inner loop? Or just continue outer?
                            # Breaking inner loop and restarting outer is safer for index consistency
                            # But we can just mark improved and break inner.
                            # The outer loop continues with next i_idx.
                            # But indices are now invalid.
                            # So we should break both.
                    # Case 2: e1 == e2? No, must connect end to start.
                    # Case 3: s1 == s2? No.
                    # Case 4: s1 == e2? Then reverse p2?
                    # Paths are directed in our list?
                    # A path [u, v] is same as [v, u] for edge coverage.
                    # But for merging, we need to align them.
                    # If s1 == e2, we can reverse p2 and then check e1 == s2 (which is now old e2).
                    # Actually, if s1 == e2, then we can merge p2 (reversed) + p1?
                    # Or p1 + p2 (reversed)?
                    # If s1 == e2, then p1 starts at s1, p2 ends at e2.
                    # So they meet at s1/e2.
                    # We can form p2_reversed + p1?
                    # p2_reversed starts at e2 (which is s1) and ends at s2.
                    # So p2_reversed + p1 is valid if p2_reversed end == p1 start?
                    # p2_reversed end is s2. p1 start is s1.
                    # We need s2 == s1? No.
                    # We need the end of the first part to match the start of the second.
                    # If we do p2_reversed + p1, we need end(p2_rev) == start(p1).
                    # end(p2_rev) = s2. start(p1) = s1.
                    # So we need s2 == s1.
                    # But we have s1 == e2.
                    # So this case is s1 == e2.
                    # If we reverse p2, it becomes [e2, ..., s2].
                    # If we append p1 to p2_rev, we need end(p2_rev) == start(p1) => s2 == s1.
                    # This is not generally true.
                    
                    # Let's check all 4 combinations of orientations.
                    # P1: s1 -> e1
                    # P2: s2 -> e2
                    
                    # Combos:
                    # 1. P1 + P2: needs e1 == s2.
                    # 2. P1 + P2_rev: needs e1 == e2. (End of P1 matches End of P2? No, P2_rev starts at e2).
                    #    P2_rev is e2 -> s2.
                    #    So P1 + P2_rev needs e1 == e2.
                    # 3. P1_rev + P2: needs s1 == s2. (P1_rev starts at e1, ends at s1).
                    #    P1_rev + P2 needs s1 == s2.
                    # 4. P1_rev + P2_rev: needs s1 == e2.
                    
                    # So we check:
                    # if e1 == s2: merge p1 + p2
                    # if e1 == e2: merge p1 + p2[::-1]
                    # if s1 == s2: merge p1[::-1] + p2
                    # if s1 == e2: merge p1[::-1] + p2[::-1]
                    
                    merged = False
                    if e1 == s2 and v1_set.isdisjoint(set(p2)):
                        paths[i_idx] = p1 + p2[1:]
                        merged = True
                    elif e1 == e2 and v1_set.isdisjoint(set(p2)):
                        paths[i_idx] = p1 + p2[::-1][1:]
                        merged = True
                    elif s1 == s2 and v1_set.isdisjoint(set(p2)):
                        paths[i_idx] = p1[::-1] + p2[1:]
                        merged = True
                    elif s1 == e2 and v1_set.isdisjoint(set(p2)):
                        paths[i_idx] = p1[::-1] + p2[::-1][1:]
                        merged = True
                    
                    if merged:
                        paths.pop(j_idx)
                        improved = True
                        break # Break inner loop
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Run decomposition with time limit
        # We can run multiple restarts if time permits
        best_paths = None
        best_count = float('inf')
        
        # Since the problem is small, one run should be good.
        # But let's try to be robust.
        
        # Estimate time per graph
        # Total time 140s. 3 graphs. ~46s each.
        
        current_limit = time.time() + 40
        
        # Run greedy
        paths = decompose(edges, n, current_limit)
        
        # Validate and count
        if paths:
             # Check validity
             valid = True
             used_edges = set()
             for path in paths:
                 for u, v in zip(path, path[1:]):
                     edge = tuple(sorted((u, v)))
                     if edge in used_edges:
                         valid = False
                         break
                     used_edges.add(edge)
                 # Check simple
                 if len(set(path)) != len(path):
                     valid = False
                     break
             if valid and used_edges == set(tuple(sorted(e)) for e in edges):
                 if len(paths) < best_count:
                     best_count = len(paths)
                     best_paths = paths
        
        if not best_paths:
             # Fallback: each edge as path
             best_paths = [[u, v] for u, v in edges]
        
        results.append(best_paths)
        
    return results

from random import Random
