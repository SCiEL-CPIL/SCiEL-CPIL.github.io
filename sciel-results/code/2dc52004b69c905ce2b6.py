"""Snake-in-the-box Q_9: bitmask-accelerated beam search with time-budgeted restarts."""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

# Precompute bit masks for each dimension
BIT_MASKS = [1 << b for b in range(N)]
# Precompute neighbor mask for each vertex: bitmask of all 9 neighbors
NEIGHBOR_MASKS = [0] * (1 << N)
for v in range(1 << N):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m

def _greedy_extend(start, rng):
    """Greedy extension from start vertex. Returns list of vertices."""
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0  # vertices at dist 1 from any non-last path vertex
    
    while True:
        cur = path[-1]
        # Candidate neighbors of cur
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break
        # Pick first available (or random)
        if rng is not None:
            # Pick a random available bit
            bits = []
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                bits.append(lsb.bit_length() - 1)
                tmp ^= lsb
            v = bits[rng.randrange(len(bits))]
        else:
            v = (candidates & (-candidates)).bit_length() - 1
        
        path.append(v)
        used_mask |= (1 << v)
        # cur becomes non-consecutive, so its neighbors are now forbidden
        forbidden_mask |= NEIGHBOR_MASKS[cur]
    
    return path

def _beam_search(start, beam_width, max_time, rng=None):
    """
    Beam search using bitmask state for O(1) safety checks.
    State: (path_list, used_mask, forbidden_mask)
    Returns list of vertices.
    """
    # Each beam entry: (path_list, used_mask, forbidden_mask)
    beam = [([start], 1 << start, 0)]
    best_path = [start]
    best_len = 1
    
    while True:
        if time.time() > max_time:
            break
        
        new_beam = []
        for path, used_mask, forbidden_mask in beam:
            cur = path[-1]
            # Available candidates: neighbors of cur not in used or forbidden
            candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            if candidates == 0:
                continue
            
            # Iterate over set bits in candidates
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                # cur becomes non-consecutive; add its neighbors to forbidden
                new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                new_beam.append((new_path, new_used, new_forbidden))
        
        if not new_beam:
            break
        
        # Sort by length descending, keep top beam_width
        # All paths in a single step have same length, so sorting is stable/diverse
        # Use a diversity-aware selection: keep first beam_width unique paths
        if len(new_beam) > beam_width:
            # Sort by path content for diversity (lexicographic)
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]
        else:
            beam = new_beam
        
        # Track best
        for p, _, _ in beam:
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
        
        # Early exit if no beam can extend (checked at top of next iteration)
    
    return best_path

def _beam_search_random(start, beam_width, max_time, rng):
    """Beam search with random candidate ordering for diversity."""
    beam = [([start], 1 << start, 0)]
    best_path = [start]
    best_len = 1
    
    while True:
        if time.time() > max_time:
            break
        
        new_beam = []
        for path, used_mask, forbidden_mask in beam:
            cur = path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            if candidates == 0:
                continue
            
            # Get list of candidate vertices and shuffle
            bits = []
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                bits.append(lsb.bit_length() - 1)
                tmp ^= lsb
            
            rng.shuffle(bits)
            
            for v in bits:
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                new_beam.append((new_path, new_used, new_forbidden))
        
        if not new_beam:
            break
        
        if len(new_beam) > beam_width:
            # Random selection for diversity
            rng.shuffle(new_beam)
            beam = new_beam[:beam_width]
        else:
            beam = new_beam
        
        for p, _, _ in beam:
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_path = None
    best_len = 0
    rng = random.Random(42)
    
    # Greedy seed
    path = _greedy_extend(0, None)
    best_path = path
    best_len = len(path)
    
    # Beam search with increasing widths from start=0
    # With bitmask ops, each step is ~9 candidates * O(1) checks
    # So we can afford much larger beams
    beam_widths = [50, 100, 200, 400, 800]
    
    for bw in beam_widths:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        t0 = time.time()
        # Allocate proportional time: larger beams need more time
        alloc = min(25.0, remaining * 0.25)
        result = _beam_search(0, bw, t0 + alloc, None)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result
    
    # Random-start beam searches
    if time.time() < deadline:
        starts = list(range(1 << N))
        rng.shuffle(starts)
        for start in starts[:30]:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 8:
                break
            t0 = time.time()
            alloc = min(8.0, remaining - 2.0)
            if alloc < 2:
                break
            # Alternate between deterministic and random beam
            if rng.random() < 0.5:
                result = _beam_search(start, 100, t0 + alloc, None)
            else:
                result = _beam_search_random(start, 100, t0 + alloc, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result
    
    # Final: try largest beam with best remaining time
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 10:
            t0 = time.time()
            result = _beam_search(0, 1000, t0 + remaining - 1.0, None)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result
    
    return best_path
