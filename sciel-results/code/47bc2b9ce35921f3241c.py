"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N
DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_path = None
    best_len = 0
    
    # Precompute neighbor masks
    bit_masks = [1 << b for b in range(N)]
    
    def is_valid_extension(v, path, used):
        """Check if v can be appended to path."""
        if v in used:
            return False
        # v must differ from all non-last vertices by >= 2 bits
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used, bit_order):
        """Extend path greedily using the given bit order."""
        nonlocal deadline
        while True:
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ bit_masks[b]
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path, used
            path.append(nxt)
            used.add(nxt)
            # Check deadline
            if time.time() > deadline:
                return path, used
    
    # Strategy 1: Original greedy with bit order 0,1,...,8
    path, used = [0], {0}
    path, used = greedy_extend(path, used, list(range(N)))
    best_path = path[:]
    best_len = len(path) - 1
    
    if time.time() > deadline:
        return best_path
    
    # Strategy 2: Try different starting vertices and bit orders
    # Use randomized restarts with different bit permutations
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random starting vertex
        start_v = rng.randint(0, TOTAL_VERTICES - 1)
        
        # Random bit order
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        
        path, used = [start_v], {start_v}
        path, used = greedy_extend(path, used, bit_order)
        
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path[:]
    
    # Strategy 3: Local search / backtracking to extend stuck paths
    # Try to extend the best path found so far by backtracking
    if best_path and time.time() < deadline:
        improved = try_backtrack_extend(best_path, deadline)
        if improved and len(improved) - 1 > best_len:
            best_len = len(improved) - 1
            best_path = improved
    
    return best_path

def try_backtrack_extend(path, deadline):
    """Try to extend path using limited backtracking."""
    path = path[:]
    used = set(path)
    cur_path_len = len(path)
    
    def backtrack(depth):
        nonlocal deadline, cur_path_len
        if depth == 0 or time.time() > deadline:
            return False
        
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v conflicts with any non-last vertex
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            
            path.append(v)
            used.add(v)
            cur_path_len += 1
            
            if backtrack(depth - 1):
                return True
            
            path.pop()
            used.remove(v)
            cur_path_len -= 1
        
        return False
    
    # Try limited backtracking from the current end
    success = backtrack(5)
    if success:
        return path
    return None
# EVOLVE-BLOCK-END
