import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute popcount
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(1, MAX_V):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute compatibility: for each vertex v, a Python int bitmask of vertices u where dist(v,u) >= 2
    # We need this for fast "is nb compatible with all used vertices" check
    # compatible_mask[v] has bit u set if dist(v,u) >= 2
    # Then "nb compatible with all used" means: (used_mask ^ (1<<cur)) & ~compatible_mask[nb] == 0
    # i.e., no used vertex (other than cur) is at distance 1 from nb
    # Wait, that's not right. We need dist(nb, u) >= 2 for all used u != cur.
    # So we need: for all u in used \ {cur}, dist(nb, u) >= 2
    # This is equivalent to: (used_mask ^ (1<<cur)) & incompatible_mask[nb] == 0
    # where incompatible_mask[nb] has bit u set if dist(nb, u) <= 1 (i.e., u == nb or dist==1)
    # But nb is not in used (we check that separately), so we just need dist(nb,u) != 1 for all used u != cur
    # Actually dist(nb, u) can be 0 (if u==nb, but nb not in used) or 1 or >=2
    # So we need: no used vertex u (other than cur) is adjacent to nb
    # i.e., (used_mask ^ (1<<cur)) & adj_mask[nb] == 0
    # where adj_mask[nb] has bit u set if dist(nb,u) == 1
    
    # Let's precompute adj_mask: for each v, bitmask of its neighbors
    adj_mask = [0] * MAX_V
    for v in range(MAX_V):
        m = 0
        for nb in neighbors[v]:
            m |= (1 << nb)
        adj_mask[v] = m
    
    # Now the compatibility check is: (used_mask ^ (1<<cur)) & adj_mask[nb] == 0
    # This means: no used vertex (other than cur) is adjacent to nb
    # Combined with: nb not in used (checked separately) and nb adjacent to cur (checked separately)
    # This gives us: nb is adjacent to cur, not used, and no other used vertex is adjacent to nb
    # But we also need: no other used vertex is at distance 1 from nb -- that's exactly adj_mask
    # AND we need non-consecutive vertices differ in at least 2 bits
    # Consecutive vertices differ in exactly 1 bit (guaranteed by adjacency)
    # Non-consecutive: for any i < j with j-i >= 2, dist(path[i], path[j]) >= 2
    # This is equivalent to: for any two non-consecutive used vertices, they are not adjacent
    # When we add nb as the new last vertex, we need: for all used u (except cur), dist(nb, u) >= 2
    # This is: (used_mask ^ (1<<cur)) & adj_mask[nb] == 0
    # Yes! That's correct.
    
    def is_valid_extension(cur, nb, used_mask):
        """Check if nb can be added after cur in the path with current used_mask."""
        if (used_mask >> nb) & 1:
            return False
        # nb must be adjacent to cur
        if ((1 << nb) & adj_mask[cur]) == 0:
            return False
        # No other used vertex should be adjacent to nb
        mask_others = used_mask ^ (1 << cur)
        if (mask_others & adj_mask[nb]) != 0:
            return False
        return True
    
    def count_valid_extensions(cur, used_mask):
        """Count how many valid extensions are available from cur."""
        count = 0
        for nb in neighbors[cur]:
            if is_valid_extension(cur, nb, used_mask):
                count += 1
        return count
    
    def greedy_extend(path, used_mask, randomize=True, seed=None):
        """Greedily extend a path using Warnsdorff-like heuristic."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            
            candidates = []
            for nb in neighbors[cur]:
                if is_valid_extension(cur, nb, used_mask):
                    new_used = used_mask | (1 << nb)
                    deg = count_valid_extensions(nb, new_used)
                    candidates.append((deg, nb))
            
            if not candidates:
                return path, used_mask
            
            candidates.sort(key=lambda x: x[0])
            
            if randomize and len(candidates) > 1:
                k = min(3, len(candidates))
                chosen = rng.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
    
    def backtrack_extend(path, used_mask, best_len_ref):
        """
        Backtracking search with MRV heuristic.
        best_len_ref is a list [best_len, best_path] for nonlocal access.
        """
        if time.time() > deadline:
            return
        
        cur = path[-1]
        
        candidates = []
        for nb in neighbors[cur]:
            if is_valid_extension(cur, nb, used_mask):
                new_used = used_mask | (1 << nb)
                deg = count_valid_extensions(nb, new_used)
                candidates.append((deg, nb))
        
        if not candidates:
            if len(path) > best_len_ref[0]:
                best_len_ref[0] = len(path)
                best_len_ref[1] = path[:]
            return
        
        candidates.sort(key=lambda x: x[0])
        
        for deg, nb in candidates:
            if time.time() > deadline:
                return
            
            path.append(nb)
            used_mask |= (1 << nb)
            
            backtrack_extend(path, used_mask, best_len_ref)
            
            path.pop()
            used_mask ^= (1 << nb)
    
    def try_extend_both_ends(path, used_mask):
        """Try to extend the path at both ends repeatedly."""
        new_path = path[:]
        new_mask = used_mask
        
        # Try prepending
        while True:
            if time.time() > deadline:
                break
            found = False
            first = new_path[0]
            for nb in neighbors[first]:
                if (new_mask >> nb) & 1:
                    continue
                # Check: nb adjacent to first, and no other vertex in path is adjacent to nb
                mask_others = new_mask ^ (1 << first)
                if (mask_others & adj_mask[nb]) == 0:
                    new_path = [nb] + new_path
                    new_mask = new_mask | (1 << nb)
                    found = True
                    break
            if not found:
                break
        
        # Try appending
        while True:
            if time.time() > deadline:
                break
            found = False
            last = new_path[-1]
            for nb in neighbors[last]:
                if (new_mask >> nb) & 1:
                    continue
                mask_others = new_mask ^ (1 << last)
                if (mask_others & adj_mask[nb]) == 0:
                    new_path = new_path + [nb]
                    new_mask = new_mask | (1 << nb)
                    found = True
                    break
            if not found:
                break
        
        return new_path, new_mask
    
    # Main search
    random.seed(12345)
    best_len = 0
    best_path = []
    best_len_ref = [0, []]
    
    # Phase 1: Greedy restarts
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    for start_idx in range(min(200, len(start_vertices))):
        if time.time() > deadline - 30:
            break
        
        start = start_vertices[start_idx]
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = greedy_extend(path, used_mask, randomize=True, seed=start_idx)
        path, used_mask = try_extend_both_ends(path, used_mask)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Backtracking from best start
    if best_path and len(best_path) > 0:
        start = best_path[0]
        path = [start]
        used_mask = 1 << start
        best_len_ref = [best_len, best_path[:]]
        backtrack_extend(path, used_mask, best_len_ref)
        if best_len_ref[0] > best_len:
            best_len = best_len_ref[0]
            best_path = best_len_ref[1][:]
    
    # Phase 3: More backtracking from different starts if time remains
    for start_idx in range(len(start_vertices)):
        if time.time() > deadline - 5:
            break
        
        start = start_vertices[start_idx]
        if best_path and start == best_path[0]:
            continue
        
        path = [start]
        used_mask = 1 << start
        best_len_ref = [best_len, best_path[:]]
        backtrack_extend(path, used_mask, best_len_ref)
        if best_len_ref[0] > best_len:
            best_len = best_len_ref[0]
            best_path = best_len_ref[1][:]
    
    if not best_path:
        return [0]
    
    return best_path
