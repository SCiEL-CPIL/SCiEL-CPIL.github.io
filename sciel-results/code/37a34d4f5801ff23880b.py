import random
import time

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute for each vertex: bitmask of all neighbors
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 110.0
    rnd = random.Random(20260715)
    best_path = None
    best_len = 0

    def validate_and_score(path):
        """Check validity and return edge count."""
        if not path:
            return 0
        n = len(path)
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return -1
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return -1
            if i >= 2:
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return -1
            used_mask |= (1 << v)
        return n - 1

    def greedy_extend(path, used_mask, last_v, priority):
        """Try to greedily extend a partial path."""
        path = list(path)
        used_mask = used_mask
        cur = last_v
        for _ in range(300):
            if time.time() > deadline:
                break
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        return path, used_mask

    def beam_search(width, max_steps, seed_priority=None):
        """Beam search: maintain up to `width` partial paths."""
        if seed_priority is None:
            seed_priority = list(range(N))
        
        beams = [([0], 1, 0)]
        
        best_found = [0]
        best_found_len = 1
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            new_beams = []
            seen_keys = set()
            
            for path, used_mask, last_v in beams:
                cur = path[-1]
                # Track current best
                L = len(path)
                if L > best_found_len:
                    best_found_len = L
                    best_found = list(path)
                
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
                    # Dead end
                    L = len(path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = list(path)
                    continue
                
                for v in candidates:
                    new_path = path + [v]
                    key = (tuple(new_path), used_mask | (1 << v))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    new_beams.append((new_path, used_mask | (1 << v), v))
                    
                    L = len(new_path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = list(new_path)
                
                if len(new_beams) > width * 3:
                    break
            
            if not new_beams:
                break
            
            # Prune: keep top `width` by length
            new_beams.sort(key=lambda x: -len(x[0]))
            beams = new_beams[:width]
        
        # Final check
        for path, used_mask, last_v in beams:
            L = len(path)
            if L > best_found_len:
                best_found_len = L
                best_found = list(path)
        
        return best_found, best_found_len

    def greedy(priority):
        path = [0]
        used = {0}
        used_mask = 1
        for _ in range(300):
            if time.time() > deadline:
                break
            cur = path[-1]
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            used_mask |= (1 << nxt)
        return path

    # Phase 0: Quick greedy baseline
    seed_path = greedy(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # Phase 1: Beam search with various configs
    beam_configs = [
        (16, 200),
        (32, 180),
        (16, 220),
        (8, 250),
        (24, 200),
    ]
    
    for width, max_steps in beam_configs:
        if time.time() > deadline:
            break
        for trial in range(15):
            if time.time() > deadline:
                break
            priority = list(range(N))
            rnd.shuffle(priority)
            p, L = beam_search(width, max_steps, priority)
            if p and L > best_len:
                vL = validate_and_score(p)
                if vL > best_len:
                    best_len = vL
                    best_path = p

    # Phase 2: Greedy restarts
    while time.time() < deadline - 10:
        priority = list(range(N))
        rnd.shuffle(priority)
        p = greedy(priority)
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p

    # Phase 3: Use remaining time for more beam searches
    while time.time() < deadline - 3:
        width = rnd.choice([16, 24, 32])
        max_steps = rnd.randint(180, 250)
        priority = list(range(N))
        rnd.shuffle(priority)
        p, L = beam_search(width, max_steps, priority)
        if p and L > best_len:
            vL = validate_and_score(p)
            if vL > best_len:
                best_len = vL
                best_path = p

    if best_path is None:
        return [0]
    return best_path
