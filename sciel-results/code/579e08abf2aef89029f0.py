"""Snake-in-the-box in Q_9: backtracking CSP with bitmask forward checking."""
import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
TOTAL_VERTICES = 1 << N

def _build_neighbor_bitmasks():
    """Precompute for each vertex, the bitmask of its N neighbors."""
    neighbors = [0] * TOTAL_VERTICES
    for v in range(TOTAL_VERTICES):
        mask = 0
        for b in range(N):
            mask |= 1 << (v ^ (1 << b))
        neighbors[v] = mask
    return neighbors

def _precompute_adjacency():
    """Build adjacency as list of sets for fast membership testing."""
    adj = [set() for _ in range(TOTAL_VERTICES)]
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            adj[v].add(v ^ (1 << b))
    return adj

def _solve_backtrack(start, deadline, random_seed=None):
    """Backtracking search to build longest induced path starting from start.
    
    Uses bitmask-based forward checking:
    - 'used' bitmask: vertices already in path
    - For the current vertex, candidate neighbors are those not used and not adjacent to path[-2]
    - MRV: at each step, the choice is forced (only 1 candidate) or we branch
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    path = [start]
    used = 1 << start
    
    # Precompute neighbor lists
    neighbors = _build_neighbor_bitmasks()
    
    best_len = len(path)
    best_path = path[:]
    
    # We'll use iterative deepening / randomized backtracking
    # Key insight: at each step, from current vertex v, the next must be:
    #   - a neighbor of v
    #   - not in used
    #   - not a neighbor of path[-2] (the one before current)
    # So valid candidates = neighbors(v) & ~used & ~neighbors(path[-2])
    
    def backtrack(path, used, depth):
        nonlocal best_len, best_path
        
        if time.time() > deadline:
            return False
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        
        # Get candidate neighbors
        if len(path) >= 2:
            prev = path[-2]
            candidates_mask = neighbors[cur] & ~used & ~neighbors[prev]
        else:
            candidates_mask = neighbors[cur] & ~used
        
        if candidates_mask == 0:
            return False
        
        # Count candidates
        cand_count = candidates_mask.bit_count()
        
        if cand_count == 1:
            # Forced move
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            path.append(v)
            used |= 1 << v
            if backtrack(path, used, depth + 1):
                pass
            path.pop()
            used ^= 1 << v
            return False  # Continue exploring other branches if possible
        else:
            # Multiple candidates: try them in random order
            # Extract candidate vertices
            cands = []
            mask = candidates_mask
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                cands.append(v)
                mask ^= lsb
            
            if random_seed is not None:
                random.shuffle(cands)
            
            # Try each candidate
            found = False
            for v in cands:
                if time.time() > deadline:
                    return False
                path.append(v)
                used |= 1 << v
                if backtrack(path, used, depth + 1):
                    found = True
                    # Don't return immediately - keep searching for longer paths
                path.pop()
                used ^= 1 << v
                
                if time.time() > deadline:
                    return False
            
            return found
    
    backtrack(path, used, 0)
    return best_path

def _solve_with_restarts(deadline, n_restarts=200):
    """Try multiple random starting vertices with backtracking."""
    best_path = None
    best_len = 0
    
    # First try start=0
    path = _solve_backtrack(0, deadline, random_seed=None)
    if path and len(path) > best_len:
        best_len = len(path)
        best_path = path[:]
    
    # Try random starts
    remaining_time = deadline - time.time()
    per_restart = remaining_time / max(1, n_restarts)
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        start = random.randint(0, TOTAL_VERTICES - 1)
        # Give each restart a portion of remaining time
        sub_deadline = min(deadline, time.time() + max(0.5, per_restart))
        
        path = _solve_backtrack(start, sub_deadline, random_seed=i + 42)
        if path and len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    return best_path

def construct():
    """Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: 20s before 150s limit
    
    # Precompute neighbor bitmasks (fast, done once)
    # This is cached inside _solve_backtrack but let's do it once globally
    
    # Use randomized backtracking with restarts
    result = _solve_with_restarts(deadline, n_restarts=100)
    
    if result is None:
        # Fallback: simple greedy
        result = _greedy_fallback()
    
    return result

def _greedy_fallback():
    """Simple greedy fallback if backtracking fails completely."""
    N = 9
    path = [0]
    used = {0}
    
    while True:
        cur = path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check distance >= 2 from all path[0:-1]
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used.add(v)
                extended = True
                break
        if not extended:
            break
    
    return path

# Precompute neighbor bitmasks at module level for speed
_NEIGHBOR_BITMASKS = _build_neighbor_bitmasks()

def _solve_backtrack_fast(start, deadline, random_seed=None):
    """Faster backtracking using precomputed neighbor bitmasks."""
    if random_seed is not None:
        random.seed(random_seed)
    
    path = [start]
    used = 1 << start
    neighbors = _NEIGHBOR_BITMASKS
    
    best_len = len(path)
    best_path = path[:]
    
    def backtrack(path, used):
        nonlocal best_len, best_path
        
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        
        if len(path) >= 2:
            prev = path[-2]
            candidates_mask = neighbors[cur] & ~used & ~neighbors[prev]
        else:
            candidates_mask = neighbors[cur] & ~used
        
        if candidates_mask == 0:
            return
        
        cand_count = candidates_mask.bit_count()
        
        if cand_count == 1:
            # Forced move
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            path.append(v)
            used |= 1 << v
            backtrack(path, used)
            path.pop()
            used ^= 1 << v
        else:
            # Multiple candidates: extract and shuffle
            cands = []
            mask = candidates_mask
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                cands.append(v)
                mask ^= lsb
            
            random.shuffle(cands)
            
            for v in cands:
                if time.time() > deadline:
                    return
                path.append(v)
                used |= 1 << v
                backtrack(path, used)
                path.pop()
                used ^= 1 << v
    
    backtrack(path, used)
    return best_path

def _solve_with_restarts_fast(deadline, n_restarts=50):
    """Try multiple random starting vertices with fast backtracking."""
    best_path = None
    best_len = 0
    
    # First try start=0 with full time allocation
    time_left = deadline - time.time()
    
    # Allocate time: first 30% to start=0, rest split among random starts
    first_alloc = min(time_left * 0.3, deadline - time.time())
    first_deadline = time.time() + max(1.0, first_alloc)
    
    path = _solve_backtrack_fast(0, first_deadline, random_seed=0)
    if path and len(path) > best_len:
        best_len = len(path)
        best_path = path[:]
    
    # Try random starts
    remaining_time = deadline - time.time()
    if remaining_time < 1.0:
        return best_path
    
    per_restart = remaining_time / max(1, n_restarts)
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        start = random.randint(0, TOTAL_VERTICES - 1)
        sub_deadline = min(deadline, time.time() + max(0.3, per_restart))
        
        path = _solve_backtrack_fast(start, sub_deadline, random_seed=i + 100)
        if path and len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    return best_path

def construct():
    """Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: 20s before 150s limit
    
    # Use fast backtracking with restarts
    result = _solve_with_restarts_fast(deadline, n_restarts=50)
    
    if result is None or len(result) == 0:
        # Fallback: simple greedy
        result = _greedy_fallback()
    
    return result
