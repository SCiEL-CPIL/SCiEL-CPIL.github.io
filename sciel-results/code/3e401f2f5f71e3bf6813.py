import time
import random

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks and non-neighbor masks for fast adjacency checks
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(42)
    best_path = None
    best_len = 0

    def validate_and_score(path):
        """Check validity and return edge count (len-1) or -1 if invalid."""
        if not path:
            return 0
        n = len(path)
        seen = set()
        for i, v in enumerate(path):
            if v in seen:
                return -1
            seen.add(v)
            if i >= 1:
                if (v ^ path[i-1]).bit_count() != 1:
                    return -1
                for j in range(i - 1):
                    if (v ^ path[j]).bit_count() == 1:
                        return -1
        return n - 1

    def greedy_extend(path, priority):
        """Extend a partial path greedily using given bit-priority order with 1-step lookahead."""
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        cur = path[-1]
        while True:
            if time.time() > deadline - 2:
                break
            nxt = None
            best_nxt = None
            best_lookahead = -1
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                # Check v not adjacent to any earlier vertex except cur
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                # 1-step lookahead: count how many valid extensions from v
                la = 0
                for b2 in priority:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if used_mask & (1 << v2):
                        continue
                    forb2 = NEIGHBOR_MASK[v2] & used_mask
                    forb2 &= ~(1 << v)
                    if not forb2:
                        la += 1
                if best_nxt is None or la > best_lookahead:
                    best_nxt = v
                    best_lookahead = la
                if la >= 3:
                    break
            if best_nxt is None:
                break
            path.append(best_nxt)
            used_mask |= (1 << best_nxt)
            cur = best_nxt
        return path

    def random_greedy():
        """Generate a random greedy snake with lookahead."""
        start = rnd.randint(0, SIZE - 1)
        priority = list(range(N))
        rnd.shuffle(priority)
        path = [start]
        used_mask = 1 << start
        cur = start
        while True:
            if time.time() > deadline - 2:
                break
            best_nxt = None
            best_la = -1
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                # Lookahead
                la = 0
                for b2 in priority:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if used_mask & (1 << v2):
                        continue
                    forb2 = NEIGHBOR_MASK[v2] & used_mask
                    forb2 &= ~(1 << v)
                    if not forb2:
                        la += 1
                if best_nxt is None or la > best_la:
                    best_nxt = v
                    best_la = la
                if la >= 4:
                    break
            if best_nxt is None:
                break
            path.append(best_nxt)
            used_mask |= (1 << best_nxt)
            cur = best_nxt
        return path

    def backtrack_improve(path, depth_limit=None):
        """
        Try to improve a path using backtracking on the last few positions.
        Replaces suffixes with better alternatives.
        """
        if not path:
            return path
        L = len(path)
        if depth_limit is None:
            depth_limit = min(L, 8)
        
        # Build used_mask for the prefix
        prefix_len = L - depth_limit
        if prefix_len < 0:
            prefix_len = 0
        prefix = path[:prefix_len]
        suffix = path[prefix_len:]
        
        used_mask = 0
        for v in prefix:
            used_mask |= (1 << v)
        
        cur = prefix[-1] if prefix else None
        new_suffix = []
        
        # Try to extend from the end of prefix
        # Use a DFS with limited depth
        def dfs(cur_v, used_m, suffix_list, max_depth):
            if time.time() > deadline - 1:
                return
            if len(suffix_list) >= max_depth:
                return
            best_extend = None
            best_extend_len = 0
            
            # Try all bit flips in random order
            bits = list(range(N))
            rnd.shuffle(bits)
            for b in bits:
                v = cur_v ^ (1 << b) if cur_v is not None else (1 << b)
                if cur_v is None:
                    v = 1 << b
                if used_m & (1 << v):
                    continue
                # Check adjacency constraint
                if cur_v is not None:
                    forbidden = NEIGHBOR_MASK[v] & used_m
                    forbidden &= ~(1 << cur_v)
                    if forbidden:
                        continue
                
                new_list = suffix_list + [v]
                new_used = used_m | (1 << v)
                new_cur = v
                
                # Greedy extend from here
                temp_path = new_list
                temp_used = new_used
                temp_cur = new_cur
                extend_count = 0
                while extend_count < max_depth - len(new_list):
                    found = False
                    for bb in bits:
                        vv = temp_cur ^ (1 << bb)
                        if temp_used & (1 << vv):
                            continue
                        forb = NEIGHBOR_MASK[vv] & temp_used
                        forb &= ~(1 << temp_cur)
                        if forb:
                            continue
                        temp_path.append(vv)
                        temp_used |= (1 << vv)
                        temp_cur = vv
                        extend_count += 1
                        found = True
                        break
                    if not found:
                        break
                
                if len(temp_path) > best_extend_len:
                    best_extend_len = len(temp_path)
                    best_extend = temp_path
                
                if best_extend and len(best_extend) >= max_depth:
                    break
        
        # This simple approach may not be very effective. Let's do a proper local search instead.
        return path

    def local_search_improve(path):
        """
        Try to improve a path by local search:
        - Remove a vertex from the suffix and re-extend greedily.
        - Try multiple random priorities for re-extension.
        """
        if not path:
            return path
        L = len(path)
        best_local = list(path)
        best_local_len = L
        
        # Try removing vertices from various positions and re-extending
        # Focus on later positions first (less disruptive to the prefix)
        positions = list(range(1, len(best_local)))
        rnd.shuffle(positions)
        
        for i in positions:
            if time.time() > deadline - 5:
                break
            # Remove vertex at position i
            candidate = best_local[:i] + best_local[i+1:]
            # Quick validity check: removing a middle vertex from a valid snake
            # is valid if the new consecutive pair (path[i-1], path[i+1]) differ in >= 2 bits
            # OR if they're the same (can't happen in a path) 
            # Actually removing path[i] makes path[i-1] adjacent to path[i+1].
            # They must differ in exactly 1 bit for the path to remain valid.
            if i > 0 and i < len(best_local) - 1:
                if (best_local[i-1] ^ best_local[i+1]).bit_count() != 1:
                    continue
            
            # Try several random priorities for re-extension
            for trial in range(3):
                if time.time() > deadline - 5:
                    break
                pr = list(range(N))
                rnd.shuffle(pr)
                extended = greedy_extend(list(candidate), pr)
                if len(extended) > best_local_len:
                    best_local = extended
                    best_local_len = len(extended)
                    break
        
        # Also try extending the current best by trying different end priorities
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            pr = list(range(N))
            rnd.shuffle(pr)
            extended = greedy_extend(list(best_local), pr)
            if len(extended) > best_local_len:
                best_local = extended
                best_local_len = len(extended)
        
        return best_local

    # Phase 0: Initial greedy with standard priority
    priority = list(range(N))
    path = greedy_extend([0], priority)
    # Paths built by greedy_extend are valid by construction; skip O(n^2) validation
    if len(path) > 1:
        best_len = len(path) - 1
        best_path = path

    # Phase 1: Many random greedy restarts
    attempts = 0
    while time.time() < deadline - 15:
        p = random_greedy()
        # random_greedy maintains validity incrementally; trust the length
        L = len(p) - 1
        if L > best_len:
            best_len = L
            best_path = p
            # Try local improvement
            p2 = local_search_improve(best_path)
            L2 = len(p2) - 1
            if L2 > best_len:
                best_len = L2
                best_path = p2
        attempts += 1

    # Phase 2: Beam search with multiple widths
    def beam_search(width, max_steps, seed_priority):
        """Beam search: maintain up to `width` partial paths. Returns valid path by construction."""
        beams = [([0], 1, 0)]
        best_found = beams[0][0]
        best_found_len = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            new_beams = []
            seen_paths = set()
            
            for path, used_mask, last_v in beams:
                cur = path[-1]
                candidates = []
                for b in seed_priority:
                    v = cur ^ (1 << b)
                    if used_mask & (1 << v):
                        continue
                    forbidden = NEIGHBOR_MASK[v] & used_mask
                    forbidden &= ~(1 << cur)
                    if forbidden:
                        continue
                    candidates.append(v)
                
                if not candidates:
                    L = len(path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = list(path)
                    continue
                
                for v in candidates:
                    new_path = path + [v]
                    key = tuple(new_path[-8:])  # Use suffix as key for diversity
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    new_beams.append((new_path, used_mask | (1 << v), v))
                    
                    L = len(new_path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = list(new_path)
                
                if len(new_beams) > width * 3:
                    break
            
            if not new_beams:
                break
            
            # Prune: keep top width beams by length
            new_beams.sort(key=lambda x: -len(x[0]))
            beams = new_beams[:width]
        
        return best_found, best_found_len

    # Run beam searches
    beam_configs = [(16, 200), (32, 180), (64, 150), (16, 250)]
    for width, max_steps in beam_configs:
        if time.time() > deadline - 10:
            break
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            priority = list(range(N))
            rnd.shuffle(priority)
            p, L = beam_search(width, max_steps, priority)
            if p and len(p) > 1:
                vL = len(p) - 1
                if vL > best_len:
                    best_len = vL
                    best_path = p
                    # Try local improvement
                    p2 = local_search_improve(best_path)
                    L2 = len(p2) - 1
                    if L2 > best_len:
                        best_len = L2
                        best_path = p2

    # Phase 3: Continue with more random greedy and beam searches
    while time.time() < deadline - 5:
        if time.time() < deadline - 25:
            # More random greedy
            p = random_greedy()
            L = len(p) - 1
            if L > best_len:
                best_len = L
                best_path = p
        else:
            # Beam search
            width = rnd.choice([16, 32])
            max_steps = rnd.randint(150, 250)
            priority = list(range(N))
            rnd.shuffle(priority)
            p, L = beam_search(width, max_steps, priority)
            if p and len(p) > 1:
                vL = len(p) - 1
                if vL > best_len:
                    best_len = vL
                    best_path = p

    # Final validation of best path (safety check)
    if best_path is None:
        return [0]
    final_check = validate_and_score(best_path)
    if final_check < 0:
        # Fallback: return a simple valid path
        return [0]
    return best_path
