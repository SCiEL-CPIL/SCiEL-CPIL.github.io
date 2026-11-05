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
    # Forbidden mask as a bitmask: bit i is set if vertex i is forbidden
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
            # If we pick v, the new path is path + [v].
            # The new forbidden set for the NEXT step will be:
            # Forbidden_new = Forbidden_old | _NEIGHBOR_MASK[current] | (1 << v)
            
            # We want to count u in Neighbors(v) such that u is NOT in Forbidden_new.
            # Valid candidates: bits in _NEIGHBOR_MASK[v] that are NOT in Forbidden_new
            forbidden_new = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~forbidden_new
            count = valid_mask.bit_count()
            scored_neighbors.append((count, v))
            
        if not scored_neighbors:
            break

        # Sort by count descending, pick randomly from top ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        
        # Update Forbidden Mask
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << next_v)
        
        current = next_v
        
    return path

def _warnsdorff_search(seed_offset=0, max_backtracks=50000, max_depth=200):
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
    
    # Iterative deepening with backtracking
    # We'll use a stack-based approach to limit memory
    # State: (current, forbidden, path_so_far, depth)
    
    # Use a simple backtracking with Warnsdorff ordering
    # At each step, order neighbors by fewest valid extensions (Warnsdorff)
    # If we hit a dead end, backtrack
    
    def extend(current, forbidden, path, depth):
        nonlocal best_path, best_len, backtrack_count
        
        if depth > 0 and backtrack_count > max_backtracks:
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
            
        # Warnsdorff: order by fewest valid extensions (to avoid dead ends)
        scored = []
        for v in valid_neighbors:
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~new_forbidden
            count = valid_mask.bit_count()
            scored.append((count, v))
            
        # Sort by count ascending (Warnsdorff: fewest first)
        scored.sort(key=lambda x: x[0])
        
        # Try each neighbor in order
        for count, v in scored:
            if depth > 0 and backtrack_count > max_backtracks:
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

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy
    path = [0]
    forbidden = 1  # vertex 0 forbidden
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
        # Update forbidden
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << nxt)
        current = nxt
        
    best_path = path
    best_len = len(best_path)
    
    # Try Warnsdorff backtracking searches with different seeds
    i = 0
    while time.time() - start_time < MAX_TIME:
        # Warnsdorff search with increasing backtracking allowance
        bt_limit = min(50000 + i * 10000, 200000)
        p = _warnsdorff_search(i, max_backtracks=bt_limit)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        i += 1
        
        # Also try greedy to maintain diversity
        if i % 3 == 0:
            gp = _greedy_search(i)
            if len(gp) > best_len:
                best_path = gp
                best_len = len(gp)
        
    return best_path
