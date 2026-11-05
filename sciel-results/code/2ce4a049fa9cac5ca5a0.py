import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

# Precompute bit counts for all possible XOR results (0 to 2^9 - 1)
_BIT_COUNT = [i.bit_count() for i in range(1 << N)]
# Precompute neighbors for all 512 vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]
# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * (1 << N)
for v in range(1 << N):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask

def _greedy_search(seed_offset=0, depth_limit=1000):
    """
    Optimized Greedy search using Forbidden Mask (bitmask).
    """
    rng = random.Random(seed_offset)
    path = [0]
    forbidden = 1  # vertex 0 is forbidden
    
    current = 0
    
    while len(path) < depth_limit:
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many valid neighbors it has
        scored_neighbors = []
        for v in valid_neighbors:
            forbidden_new = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~forbidden_new
            count = valid_mask.bit_count()
            scored_neighbors.append((count, v))
            
        if not scored_neighbors:
            break

        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << next_v)
        
        current = next_v
        
    return path

def _warnsdorff_search(seed_offset=0, max_backtracks=50000, deadline=None):
    """
    Warnsdorff-style backtracking search for induced paths.
    Uses the "fewest remaining moves" heuristic with limited backtracking.
    """
    rng = random.Random(seed_offset)
    path = [0]
    forbidden = 1  # vertex 0 forbidden
    current = 0
    
    best_path = list(path)
    best_len = 1
    backtrack_count = 0
    aborted = False
    
    def extend(current, forbidden, path, depth):
        nonlocal best_path, best_len, backtrack_count, aborted
        
        if aborted:
            return False
            
        if deadline is not None and time.time() > deadline:
            aborted = True
            return False
            
        if backtrack_count > max_backtracks:
            return False
            
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Get valid neighbors
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            return False
            
        # Warnsdorff: order by fewest valid extensions
        scored = []
        for v in valid_neighbors:
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~new_forbidden
            count = valid_mask.bit_count()
            scored.append((count, v))
            
        scored.sort(key=lambda x: x[0])
        
        for count, v in scored:
            if aborted:
                return False
                
            if time.time() > deadline:
                aborted = True
                return False
                
            if backtrack_count >= max_backtracks:
                return False
                
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            path.append(v)
            backtrack_count += 1
            
            if extend(v, new_forbidden, path, depth + 1):
                path.pop()
                return True
                
            path.pop()
            
        return False
    
    extend(current, forbidden, path, 0)
    
    return best_path

def _beam_search(seed_offset=0, beam_width=30, deadline=None):
    """
    Beam search that maintains multiple partial paths simultaneously.
    """
    rng = random.Random(seed_offset)
    # Each beam entry: (path, forbidden_mask, current_vertex)
    beams = [([0], 1, 0)]
    best_path = [0]
    best_len = 1
    
    while beams:
        if deadline is not None and time.time() > deadline:
            break
            
        new_beams = []
        for path, forbidden, current in beams:
            valid_neighbors = []
            for v in _NEIGHBORS[current]:
                if not (forbidden >> v) & 1:
                    valid_neighbors.append(v)
                    
            if not valid_neighbors:
                continue
                
            # Score each neighbor
            scored = []
            for v in valid_neighbors:
                new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
                valid_mask = _NEIGHBOR_MASK[v] & ~new_forbidden
                count = valid_mask.bit_count()
                scored.append((count, v))
                
            # Take top candidates from this beam
            scored.sort(key=lambda x: -x[0])
            top_k = min(beam_width // max(1, len(beams)), len(scored))
            for count, v in scored[:top_k]:
                new_path = path + [v]
                new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
                new_beams.append((new_path, new_forbidden, v))
                if len(new_path) > best_len:
                    best_path = list(new_path)
                    best_len = len(new_path)
                    
        if not new_beams:
            break
            
        # Keep only top beam_width beams
        new_beams.sort(key=lambda x: -len(x[0]))
        # Within same length, prefer those with more future options
        def beam_score(entry):
            path, forbidden, current = entry
            valid_count = 0
            for v in _NEIGHBORS[current]:
                if not (forbidden >> v) & 1:
                    valid_count += 1
            return (len(path), valid_count)
            
        new_beams.sort(key=beam_score, reverse=True)
        beams = new_beams[:beam_width]
        
    return best_path

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy
    path = [0]
    forbidden = 1
    current = 0
    while True:
        nxt = None
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << nxt)
        current = nxt
        
    best_path = path
    best_len = len(best_path)
    
    # Alternating search strategies with time budget
    i = 0
    deadline = start_time + MAX_TIME
    
    while time.time() < deadline:
        # Check if we should stop
        if time.time() > deadline:
            break
            
        # Try Warnsdorff backtracking with increasing allowance
        if i % 2 == 0:
            bt_limit = min(30000 + i * 5000, 100000)
            p = _warnsdorff_search(i, max_backtracks=bt_limit, deadline=deadline)
            if len(p) > best_len:
                best_path = p
                best_len = len(p)
        else:
            # Try beam search
            p = _beam_search(i, beam_width=25, deadline=deadline)
            if len(p) > best_len:
                best_path = p
                best_len = len(p)
                
        i += 1
        
        # Occasionally try greedy for diversity
        if i % 5 == 0:
            gp = _greedy_search(i)
            if len(gp) > best_len:
                best_path = gp
                best_len = len(gp)
                
    return best_path
