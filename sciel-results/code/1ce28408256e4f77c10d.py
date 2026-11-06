"""Snake-in-the-box in Q_9: iterative deepening DFS with bitmask pruning."""
import time
import random
import sys

N = 9
MASK = (1 << N) - 1
TOTAL_VERTS = 1 << N  # 512
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute neighbors for fast lookup
_NEIGHBORS = []
for v in range(TOTAL_VERTS):
    nbrs = []
    for b in range(N):
        nbrs.append(v ^ (1 << b))
    _NEIGHBORS.append(nbrs)


def _is_valid_snake_full(path):
    """Full validation: consecutive differ by 1, non-consecutive differ by >= 2."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_extend_deterministic(path, used_mask):
    """Deterministic greedy: lowest bit flip first."""
    while True:
        cur = path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if (used_mask >> v) & 1:
                continue
            # Check induced constraint: v must not be adjacent to any used vertex except cur
            valid = True
            for w in _NEIGHBORS[v]:
                if w == cur:
                    continue
                if (used_mask >> w) & 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used_mask |= (1 << v)
                extended = True
                break
        if not extended:
            break
    return path, used_mask


def _dfs_extend(path, used_mask, target_len, deadline, rng, bit_order_cache):
    """
    Depth-first search trying to extend path to reach target_len.
    Returns True if a path of length >= target_len was found, updating path and used_mask in place.
    Uses iterative deepening style: we just try to find ANY extension to target_len.
    """
    if len(path) >= target_len:
        return True
    
    if time.time() > deadline:
        return False
    
    cur = path[-1]
    
    # Get neighbor ordering
    if cur in bit_order_cache:
        bits = bit_order_cache[cur]
    else:
        bits = list(range(N))
        rng.shuffle(bits)
        bit_order_cache[cur] = bits
    
    for b in bits:
        v = cur ^ (1 << b)
        
        # Check if v is already used
        if (used_mask >> v) & 1:
            continue
        
        # Check induced constraint: v must not be adjacent to any used vertex except cur
        valid = True
        for w in _NEIGHBORS[v]:
            if w == cur:
                continue
            if (used_mask >> w) & 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Extend
        path.append(v)
        used_mask |= (1 << v)
        
        if _dfs_extend(path, used_mask, target_len, deadline, rng, bit_order_cache):
            return True
        
        # Backtrack
        path.pop()
        used_mask ^= (1 << v)
        
        if time.time() > deadline:
            return False
    
    return False


def _iterative_deepening_search(start_v, max_depth, deadline, rng):
    """
    Iterative deepening: try to find a snake of length max_depth starting from start_v.
    Returns the path if found, else None.
    """
    path = [start_v]
    used_mask = 1 << start_v
    bit_order_cache = {}
    
    if _dfs_extend(path, used_mask, max_depth, deadline, rng, bit_order_cache):
        return path
    
    return None


def _quick_greedy(start_v, rng, max_steps=500):
    """Quick randomized greedy to get a baseline path."""
    path = [start_v]
    used_mask = 1 << start_v
    
    for _ in range(max_steps):
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        extended = False
        for b in bits:
            v = cur ^ (1 << b)
            if (used_mask >> v) & 1:
                continue
            valid = True
            for w in _NEIGHBORS[v]:
                if w == cur:
                    continue
                if (used_mask >> w) & 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used_mask |= (1 << v)
                extended = True
                break
        if not extended:
            break
    
    return path


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Get a good baseline via multiple greedy starts
    for seed in range(20):
        if time.time() > DEADLINE:
            break
        rng = random.Random(seed)
        start_v = rng.randint(0, TOTAL_VERTS - 1)
        path = _quick_greedy(start_v, rng, max_steps=1000)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 2: Iterative deepening DFS from best start points
    # Try to extend beyond current best
    current_best_start = best_path[0] if best_path else 0
    
    # Try increasing target lengths
    target = best_len + 1
    max_target = min(best_len + 50, TOTAL_VERTS)
    
    # Use multiple starting vertices for diversity
    start_vertices = [current_best_start]
    # Add some random starts
    for _ in range(10):
        start_vertices.append(rng.randint(0, TOTAL_VERTS - 1))
    
    for sv in start_vertices:
        if time.time() > DEADLINE - 5:
            break
        
        # Try to find a path longer than current best
        target = best_len + 1
        while target <= max_target and time.time() < DEADLINE - 5:
            path = _iterative_deepening_search(sv, target, DEADLINE - 5, rng)
            if path is not None:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                    # Continue trying to extend further
                    target += 1
                else:
                    break
            else:
                # Couldn't reach this target, try next start vertex
                break
        
        # If we found a good path, try to extend it further with greedy
        if best_path and len(best_path) >= best_len:
            # Try greedy extension from the end
            path_copy = list(best_path)
            used_mask = 0
            for v in path_copy:
                used_mask |= (1 << v)
            path_copy, used_mask = _greedy_extend_deterministic(path_copy, used_mask)
            if len(path_copy) > best_len:
                best_len = len(path_copy)
                best_path = list(path_copy)
    
    # Phase 3: More random starts with iterative deepening
    if time.time() < DEADLINE - 10:
        for seed in range(100, 200):
            if time.time() > DEADLINE - 5:
                break
            rng = random.Random(seed)
            start_v = rng.randint(0, TOTAL_VERTS - 1)
            
            # First get a quick greedy baseline
            greedy_path = _quick_greedy(start_v, rng, max_steps=500)
            
            # Try to extend beyond greedy baseline using DFS
            target = len(greedy_path) + 1
            while target <= len(greedy_path) + 30 and time.time() < DEADLINE - 5:
                path = _iterative_deepening_search(start_v, target, DEADLINE - 5, rng)
                if path is not None:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = list(path)
                    target += 1
                else:
                    break
    
    # Final validation
    if best_path and not _is_valid_snake_full(best_path):
        # Fallback to deterministic greedy
        path = [0]
        used_mask = 1 << 0
        path, used_mask = _greedy_extend_deterministic(path, used_mask)
        best_path = path
        best_len = len(path)
    
    # Ensure we have at least the seed length
    if len(best_path) < 104:
        path = [0]
        used_mask = 1 << 0
        path, used_mask = _greedy_extend_deterministic(path, used_mask)
        best_path = path
        best_len = len(path)
    
    return best_path


if __name__ == "__main__":
    result = construct()
    print(f"Snake length: {len(result)}")
