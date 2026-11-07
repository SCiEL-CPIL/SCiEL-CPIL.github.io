import time
import random

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    trees = instance_trees(20, 611)
    
    # Sort indices for processing: 18 down to 0
    # T20 (idx 19) is fixed
    
    result = [None] * 20
    result[19] = list(range(20))
    
    used_edges = set()
    for u, v in trees[19]:
        if u > v: u, v = v, u
        used_edges.add((u, v))
        
    def try_pack():
        local_used = set(used_edges)
        local_result = [None] * 20
        local_result[19] = list(range(20))
        
        packed_edges = len(used_edges)
        
        # Process T19 down to T1
        for idx in range(18, -1, -1):
            k = idx + 1
            if k == 1:
                local_result[idx] = [0]
                continue
                
            tree = trees[idx]
            found = False
            
            # For k=2, just find any free edge
            if k == 2:
                for u in range(20):
                    for v in range(u+1, 20):
                        if (u, v) not in local_used:
                            local_result[idx] = [u, v]
                            local_used.add((u, v))
                            packed_edges += 1
                            found = True
                            break
                    if found: break
                if not found:
                    continue
                continue
            
            # For k >= 3
            # Try random permutations
            # Limit attempts per tree to keep time down
            max_attempts = 500
            for _ in range(max_attempts):
                perm = list(range(20))
                random.shuffle(perm)
                mapping = perm[:k]
                
                valid = True
                # Check edges
                for u, v in tree:
                    gu = mapping[u]
                    gv = mapping[v]
                    if gu > gv: gu, gv = gv, gu
                    if (gu, gv) in local_used:
                        valid = False
                        break
                if valid:
                    for u, v in tree:
                        gu = mapping[u]
                        gv = mapping[v]
                        if gu > gv: gu, gv = gv, gu
                        local_used.add((gu, gv))
                    local_result[idx] = mapping
                    packed_edges += k - 1
                    found = True
                    break
                    
            # If not found, skip this tree
            
        return local_result, packed_edges

    best_res = result
    best_val = len(used_edges)
    
    # Run multiple attempts
    # Adjust number of attempts based on time
    attempts = 0
    while time.time() < deadline:
        res, val = try_pack()
        if val > best_val:
            best_res = res
            best_val = val
            if best_val == 190:
                return best_res
        attempts += 1
        
    return best_res
