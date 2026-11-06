"""VNS-based solver for snake-in-the-box in Q_9."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# Precompute neighbors
_NEIGHBORS = [None] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    __NEIGHBORS[_v] = [_v ^ (1 << b) for b in range(N)]

# Precompute adjacency masks
_ADJ_MASK = [0] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _m = 0
    for _nb in _NEIGHBORS[_v]:
        _m |= (1 << _nb)
    _ADJ_MASK[_v] = _m


def _is_valid_path(path):
    """Check if path is a valid induced path."""
    if not path:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    
    # Check consecutive adjacency
    for i in range(len(path) - 1):
        v1, v2 = path[i], path[i + 1]
        if (v1 ^ v2).bit_count() != 1:
            return False
    
    # Check non-consecutive non-adjacency
    # For each vertex, it cannot be adjacent to any other non-consecutive vertex in the path
    # We can do this by checking for each vertex v in path, if any other vertex u in path (not prev/next) is a neighbor of v
    # Optimization: For each v, check neighbors. If a neighbor u is in path, and u is not path[i-1] or path[i+1], invalid.
    for i, v in enumerate(path):
        for nb in _NEIGHBORS[v]:
            if nb in used:
                # Check if nb is consecutive
                if i > 0 and path[i - 1] == nb:
                    continue
                if i < len(path) - 1 and path[i + 1] == nb:
                    continue
                return False
    return True


def _greedy_extend(path, used_set, used_mask):
    """Greedily extend the path as far as possible."""
    cur = path[-1]
    while True:
        found = False
        for nb in _NEIGHBORS[cur]:
            if nb in used_set:
                continue
            # Check conflict with all other vertices
            conflict = False
            for u in path:
                if u == cur:
                    continue
                if (u ^ nb).bit_count() == 1:
                    conflict = True
                    break
            if not conflict:
                path.append(nb)
                used_set.add(nb)
                used_mask |= (1 << nb)
                cur = nb
                found = True
                break
        if not found:
            break
    return path


def _greedy_seed(bit_order=None):
    """Generate a greedy snake."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [0]
    used_set = {0}
    used_mask = 1
    _greedy_extend(path, used_set, used_mask)
    return path


def _vns_search(initial_path, deadline, max_iter=1000):
    """
    Variable Neighborhood Search.
    Neighborhoods:
    1. Swap two vertices.
    2. Reverse a sub-segment.
    3. Insert a new vertex at a random position (if valid).
    4. Remove a vertex (to allow re-insertion elsewhere later? No, we want to grow. But maybe removing a "bad" vertex allows extension).
    Actually, simpler:
    - Try to insert a new vertex at a random position.
    - Try to swap two vertices.
    - Try to reverse a segment.
    After each perturbation, greedily extend. If longer, accept.
    """
    best_path = list(initial_path)
    best_len = len(best_path)
    
    # Precompute used set for initial
    used_set = set(best_path)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Randomly choose a neighborhood structure
        # k = neighborhood size (distance from current solution)
        # For simplicity, we just pick random operations.
        
        candidate = list(best_path)
        cand_len = len(candidate)
        
        op = random.randint(0, 3)
        
        if op == 0 and cand_len > 1:
            # Swap two vertices
            i1 = random.randint(0, cand_len - 1)
            i2 = random.randint(0, cand_len - 1)
            if i1 != i2:
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                
        elif op == 1 and cand_len > 2:
            # Reverse a sub-segment
            i1 = random.randint(0, cand_len - 2)
            i2 = random.randint(i1 + 1, cand_len - 1)
            candidate[i1:i2+1] = reversed(candidate[i1:i2+1])
            
        elif op == 2:
            # Insert a new vertex
            # Pick a random unused vertex
            # To make it efficient, pick a neighbor of an existing vertex?
            # Or just pick random unused.
            # Let's pick a random unused vertex from the neighbors of the current path to increase chance of validity.
            # Actually, picking any unused vertex is a big jump.
            # Let's pick a random position to insert and a random unused vertex.
            # To be smarter, pick a vertex that is a neighbor of some path vertex.
            # For now, random unused.
            # Optimization: Only try if we have time.
            # Generate a random unused vertex.
            # 2^9 = 512 vertices.
            # We can just pick a random integer and check if unused.
            while True:
                v = random.randint(0, TOTAL_VERTICES - 1)
                if v not in used_set:
                    break
            
            pos = random.randint(0, cand_len)
            candidate.insert(pos, v)
            
        elif op == 3 and cand_len > 1:
            # Remove a vertex (to potentially allow a different extension)
            # Only remove if we can afford to lose length (we will try to extend back)
            # This is risky but might escape deep local optima.
            # Only do this if we are stuck for a while?
            # For now, just remove a random middle vertex.
            pos = random.randint(1, cand_len - 2)
            removed_v = candidate.pop(pos)
            # Note: used_set still has removed_v, but we will re-check validity from scratch in _is_valid_path
            # Wait, _is_valid_path uses the list.
            # But _greedy_extend uses used_set. We need to update used_set if we accept.
            # For the candidate, we just check validity of the list.
            pass # No-op, just structural change
        
        # Validate candidate
        if not _is_valid_path(candidate):
            continue
        
        # If valid, try to extend it
        # We need a fresh used_set for the candidate
        cand_used_set = set(candidate)
        cand_used_mask = 0
        for v in candidate:
            cand_used_mask |= (1 << v)
            
        _greedy_extend(candidate, cand_used_set, cand_used_mask)
        
        if len(candidate) > best_len:
            best_path = candidate
            best_len = len(candidate)
            used_set = set(best_path) # Update for next iteration's insertion logic if needed
            
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Phase 1: Generate multiple greedy seeds to find a good starting point
    best_path = None
    best_len = 0
    
    # Try natural order and a few shuffles
    for i in range(10):
        if time.time() > deadline - 60.0:
            break
        if i == 0:
            path = _greedy_seed()
        else:
            bo = list(range(N))
            random.shuffle(bo)
            path = _greedy_seed(bo)
        
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
            
    if best_path is None:
        best_path = _greedy_seed()
        best_len = len(best_path)
        
    # Phase 2: VNS Search
    # Allocate most of the remaining time to VNS
    vns_deadline = deadline - 5.0
    best_path = _vns_search(best_path, vns_deadline, max_iter=5000)
    
    # Phase 3: Final greedy extension just in case
    used_set = set(best_path)
    used_mask = 0
    for v in best_path:
        used_mask |= (1 << v)
    _greedy_extend(best_path, used_set, used_mask)
    
    return best_path
