import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to trivial solution if time runs out
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        m = len(edges)
        target_k = (m + 1) // 2
        
        best_paths = None
        best_count = float('inf')
        
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            rng = Random(attempt * 1000 + spec[2])
            # Start with each edge as a path
            paths = []
            for u, v in edges:
                paths.append([u, v])
                
            # Greedy merging
            # To make it faster, we can just iterate and merge if possible
            # Since n is small, checking intersection is O(n)
            merged = True
            while merged:
                merged = False
                # Shuffle paths to avoid bias? Or just iterate.
                # Iterating is deterministic for a given start, but we restart with different seeds?
                # The initial set of paths is the same (all edges).
                # The order of merging matters.
                # Let's shuffle the list of paths before each merge pass?
                # Or just pick a random pair?
                # Picking a random pair is better for exploration.
                
                # Optimization: Only try to merge if we can reduce count below target or just keep going
                # We want to minimize count.
                
                # Find all mergeable pairs? Too slow.
                # Pick a random pair of indices i, j
                # If they merge, do it.
                # If not, try another pair?
                # If no pair merges, break.
                
                # Better: Iterate through all pairs?
                # n is ~30, m is ~100-150. Paths ~100. Pairs ~5000.
                # 5000 * 50 attempts * 3 graphs = 750,000 operations. Very fast.
                
                # Let's do a full pass of checking all pairs in random order
                indices = list(range(len(paths)))
                rng.shuffle(indices)
                
                found_merge = False
                for i_idx in indices:
                    if i_idx >= len(paths): continue
                    for j_idx in indices:
                        if j_idx <= i_idx or j_idx >= len(paths): continue
                        
                        p1 = paths[i_idx]
                        p2 = paths[j_idx]
                        
                        # Try 4 orientations
                        merged_path = None
                        
                        # 1. p1 + p2
                        if p1[-1] == p2[0]:
                            new_p = p1 + p2[1:]
                            if len(new_p) == len(set(new_p)):
                                merged_path = new_p
                                
                        # 2. p1 + reversed(p2)
                        if merged_path is None and p1[-1] == p2[-1]:
                            new_p = p1 + p2[::-1][1:]
                            if len(new_p) == len(set(new_p)):
                                merged_path = new_p
                                
                        # 3. p2 + p1
                        if merged_path is None and p2[-1] == p1[0]:
                            new_p = p2 + p1[1:]
                            if len(new_p) == len(set(new_p)):
                                merged_path = new_p
                                
                        # 4. p2 + reversed(p1)
                        if merged_path is None and p2[-1] == p1[-1]:
                            new_p = p2 + p1[::-1][1:]
                            if len(new_p) == len(set(new_p)):
                                merged_path = new_p
                                
                        if merged_path is not None:
                            # Remove p2 and replace p1 with merged
                            # Careful with indices if we remove
                            # Easier: create new list
                            new_paths = []
                            for k in range(len(paths)):
                                if k != i_idx and k != j_idx:
                                    new_paths.append(paths[k])
                            new_paths.append(merged_path)
                            paths = new_paths
                            found_merge = True
                            # Restart the loop to check new possibilities
                            # Or just continue? Restarting is safer for greedy
                            break
                            
                    if found_merge:
                        break
                        
                if not found_merge:
                    break
                    
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
                
            if best_count == target_k:
                break
                
        if best_paths is None:
            # Should not happen, but fallback
            best_paths = [list(e) for e in edges]
            
        results.append(best_paths)
        
    return results
