import time
import random
import numpy as np

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor bitmasks
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = m

# Precompute bit indices for fast iteration
def _iter_bits(mask):
    """Yield indices of set bits in mask."""
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb

def _greedy_extend(start, rng):
    """Greedy extension from a start vertex, with random tie-breaking."""
    path = [start]
    used = 1 << start
    forbidden = 0
    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used | forbidden)
        if candidates == 0:
            break
        # Pick a random candidate
        bits = list(_iter_bits(candidates))
        v = rng.choice(bits)
        path.append(v)
        used |= 1 << v
        forbidden |= NEIGHBOR_MASKS[cur]
    return path

def _beam_search(start, width, deadline, rng):
    """
    Beam search using bitmask state.
    Each beam entry: (path, used_mask, forbidden_mask)
    """
    beam = [([start], 1 << start, 0)]
    best_local_path = [start]
    best_local_len = 1
    
    # Track seen suffixes to reduce redundancy
    seen_suffixes = set()
    
    while time.time() < deadline - 3:
        new_beam = []
        seen_suffixes = set()
        
        for path, used_mask, forbidden_mask in beam:
            cur = path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            
            if candidates == 0:
                if len(path) > best_local_len:
                    best_local_len = len(path)
                    best_local_path = list(path)
                continue
            
            # Try each candidate
            for v in _iter_bits(candidates):
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                
                # Deduplicate by last 4 vertices
                key = tuple(new_path[-4:]) if len(new_path) >= 4 else tuple(new_path)
                if key in seen_suffixes:
                    continue
                seen_suffixes.add(key)
                
                new_beam.append((new_path, new_used, new_forbidden))
                
                if len(new_path) > best_local_len:
                    best_local_len = len(new_path)
                    best_local_path = list(new_path)
        
        if not new_beam:
            break
        
        # Sort by length descending, keep top width
        new_beam.sort(key=lambda x: -len(x[0]))
        beam = new_beam[:width]
        
        # Early stop if beam is all dead ends
        if all(NEIGHBOR_MASKS[p[-1]] & ~(u | f) == 0 for p, u, f in beam):
            break
    
    return best_local_path

def _local_improve(path, deadline):
    """
    Try to improve a path by backtracking and trying different branches.
    Uses iterative deepening with time limit.
    """
    if len(path) <= 2:
        return path
    
    best = list(path)
    best_len = len(path)
    
    def build_state(p):
        used = 0
        forbidden = 0
        for i, v in enumerate(p):
            used |= 1 << v
            if i < len(p) - 1:
                forbidden |= NEIGHBOR_MASKS[v]
        return used, forbidden
    
    # Try to extend from the end with backtracking
    def extend_with_backtrack(cur_path, cur_used, cur_forbidden, depth_limit):
        """Try to extend cur_path up to depth_limit more steps."""
        if depth_limit == 0 or time.time() > deadline - 2:
            return cur_path
        
        cur = cur_path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forbidden)
        
        if candidates == 0:
            return cur_path
        
        best_ext = list(cur_path)
        best_len_local = len(cur_path)
        
        # Shuffle candidates for diversity
        cand_list = list(_iter_bits(candidates))
        random.shuffle(cand_list)
        
        for v in cand_list:
            new_path = cur_path + [v]
            new_used = cur_used | (1 << v)
            new_forbidden = cur_forbidden | NEIGHBOR_MASKS[cur]
            
            ext = extend_with_backtrack(new_path, new_used, new_forbidden, depth_limit - 1)
            if len(ext) > best_len_local:
                best_len_local = len(ext)
                best_ext = ext
            
            if time.time() > deadline - 2:
                break
        
        return best_ext
    
    # Try extending from the end with increasing depth
    for depth in [3, 5, 8]:
        if time.time() > deadline - 3:
            break
        used, forbidden = build_state(best)
        ext = extend_with_backtrack(best, used, forbidden, depth)
        if len(ext) > best_len:
            best_len = len(ext)
            best = ext
    
    # Try to insert a vertex in the middle (more complex, skip for now)
    
    return best

def construct():
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    best_path = _greedy_extend(0, rng)
    best_len = len(best_path)
    
    # Phase 1: Multiple greedy starts with randomization
    while time.time() < deadline - 30:
        start = rng.randint(0, SIZE - 1)
        p = _greedy_extend(start, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 2: Beam search from best path's end and various starts
    widths = [30, 50, 100, 200]
    
    while time.time() < deadline - 15:
        width = rng.choice(widths)
        
        # Try beam from the current best path's last vertex
        if len(best_path) > 0:
            end_v = best_path[-1]
            # Start beam from a few random vertices near the end
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                start = end_v ^ (1 << rng.randint(0, N - 1))  # flip one bit
                if start < SIZE:
                    p = _beam_search(start, width, deadline, rng)
                    if len(p) > best_len:
                        best_len = len(p)
                        best_path = p
        
        # Also try random starts
        start = rng.randint(0, SIZE - 1)
        p = _beam_search(start, width, deadline, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 3: Local improvement on the best path
    if time.time() < deadline - 5:
        best_path = _local_improve(best_path, deadline)
        best_len = len(best_path)
    
    # Phase 4: Final greedy extensions from best path ends
    if time.time() < deadline - 2:
        # Try extending from the front
        if len(best_path) > 0:
            front_v = best_path[0]
            # Try to prepend
            used = 0
            for v in best_path:
                used |= 1 << v
            forbidden = NEIGHBOR_MASKS[best_path[0]]
            candidates = NEIGHBOR_MASKS[front_v] & ~(used | forbidden)
            if candidates:
                for v in _iter_bits(candidates):
                    new_path = [v] + best_path
                    # Validate
                    valid = True
                    for i in range(len(new_path)):
                        if i > 0:
                            if bin(new_path[i] ^ new_path[i-1]).count('1') != 1:
                                valid = False
                                break
                        for j in range(i+2, len(new_path)):
                            if bin(new_path[i] ^ new_path[j]).count('1') < 2:
                                valid = False
                                break
                    if valid and len(new_path) > best_len:
                        best_len = len(new_path)
                        best_path = new_path
                        break
    
    return best_path

# Test locally
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Length: {len(res)}, Time: {t1-t0:.1f}s")
    # Validate
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            diff = bin(res[i] ^ res[j]).count('1')
            if abs(i-j) == 1:
                if diff != 1:
                    valid = False
                    print(f"Consecutive {i},{j}: diff={diff}")
            else:
                if diff < 2:
                    valid = False
                    print(f"Non-consecutive {i},{j}: diff={diff}")
    print(f"Valid: {valid}")
