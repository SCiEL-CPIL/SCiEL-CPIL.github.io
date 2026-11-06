"""Snake-in-the-box in Q_9: build a long induced path (snake)."""
import time
import random
import itertools

N = 9
FULL_MASK = (1 << N) - 1

def _valid_extension(v, path, used):
    """Check if v can be appended to path to keep it an induced path.
    v must not be in used, and must not be at distance 1 from any vertex
    in path[:-1] (i.e., path[0]..path[-2]).
    """
    if v in used:
        return False
    # Check against all previous vertices except the last one (which is path[-1])
    # Actually we need to check against path[0:-1] (all except the current last)
    # Because consecutive vertices (path[-2], path[-1]) are allowed to be distance 1.
    # But v is being appended after path[-1], so v and path[-2] must be distance >= 2.
    # And v and any other path[i] for i < len(path)-1 must be distance >= 2.
    for i in range(len(path) - 1):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, used, bit_order=None):
    """Try to extend the path greedily. Returns True if extended."""
    cur = path[-1]
    if bit_order is None:
        candidates = list(range(N))
    else:
        candidates = list(bit_order)
    
    for b in candidates:
        v = cur ^ (1 << b)
        if _valid_extension(v, path, used):
            path.append(v)
            used.add(v)
            return True
    return False

def _construct_greedy(seed=0, bit_order=None):
    """Construct a greedy snake starting from seed."""
    path = [seed]
    used = {seed}
    while _greedy_extend(path, used, bit_order):
        pass
    return path

def _try_backtrack(path, used, max_depth=500, deadline=None):
    """Try to extend the path using limited backtracking.
    Returns True if the path was extended (possibly via backtracking).
    """
    if deadline is not None and time.time() > deadline:
        return False
    
    # Try greedy first
    if _greedy_extend(path, used):
        return True
    
    # If greedy fails, try backtracking: remove last vertex, try different choices
    # This is a simplified approach: we'll try to find an alternative path suffix
    # For now, just return False to indicate no extension possible
    return False

def _shuffle_bits():
    """Return a shuffled list of bit indices."""
    bits = list(range(N))
    random.shuffle(bits)
    return bits

def _construct_with_randomization(iterations=1000, deadline=None):
    """Try multiple randomized greedy constructions and return the longest."""
    best_path = None
    best_len = 0
    
    for i in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random starting vertex
        seed = random.randint(0, (1 << N) - 1)
        
        # Random bit order
        bit_order = _shuffle_bits()
        
        path = _construct_greedy(seed, bit_order)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    if best_path is None:
        # Fallback to default greedy
        best_path = _construct_greedy(0)
    
    return best_path

def _local_search(path, deadline=None):
    """Attempt to improve the path via local search operations.
    This is a placeholder for more sophisticated search.
    """
    if deadline is not None and time.time() > deadline:
        return path
    
    # Try to extend from the beginning of the path
    # Check if we can prepend a vertex
    first = path[0]
    used = set(path)
    
    for b in range(N):
        v = first ^ (1 << b)
        if v in used:
            continue
        # Check if v is at distance >= 2 from all vertices in path[1:]
        # (v will be adjacent to path[0], which is allowed)
        valid = True
        for i in range(1, len(path)):
            if (v ^ path[i]).bit_count() == 1:
                valid = False
                break
        if valid:
            path.insert(0, v)
            used.add(v)
            # Try to extend further from the new beginning
            # ... (simplified: just one step)
            break
    
    return path

def construct():
    """Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 10s before the 120s limit
    
    # Start with a good greedy seed
    best_path = _construct_greedy(0)
    best_len = len(best_path)
    
    # Try multiple randomized greedy constructions
    iterations = 5000
    for i in range(iterations):
        if time.time() > deadline:
            break
        
        seed = random.randint(0, (1 << N) - 1)
        bit_order = _shuffle_bits()
        path = _construct_greedy(seed, bit_order)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Try local search on the best path found
    if time.time() < deadline - 5.0:
        best_path = _local_search(best_path, deadline)
        best_len = len(best_path)
    
    return best_path
