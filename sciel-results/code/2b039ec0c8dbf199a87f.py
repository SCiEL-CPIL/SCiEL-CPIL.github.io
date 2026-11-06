"""
Snake-in-the-Box in Q9 using Iterated Local Search with Truncation and Randomized Greedy.
"""
import time
import random
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _is_valid_extension(v, path, used):
    """
    Check if vertex v can be appended to path.
    v must be adjacent to path[-1] and not adjacent to any path[i] for i < len(path)-1.
    """
    if v in used:
        return False
    # v is adjacent to path[-1] by construction of the caller.
    # Check non-consecutive adjacency: v must not differ by exactly 1 bit from any path[i] where i <= len(path)-2
    limit = len(path) - 1
    for i in range(limit):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, used, rng):
    """
    Greedily extend the path from the end using random bit ordering.
    Returns the extended path and the updated used set.
    """
    while True:
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        found = False
        for b in bits:
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(v, path, used):
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path, used

def _try_extend_ends(path, used):
    """
    Try to extend the path from both ends.
    Returns True if any extension was made.
    """
    extended = False
    # Try extending from the end
    cur = path[-1]
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used:
            continue
        if _is_valid_extension(v, path, used):
            path.append(v)
            used.add(v)
            extended = True
            break
    
    # Try extending from the beginning
    if not extended:
        cur = path[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v can be prepended: v adjacent to path[0], not adjacent to path[2:]
            ok = True
            # v is adjacent to path[0].
            # v must not be adjacent to path[1]? No, path[0] and path[1] are consecutive.
            # v and path[1] are non-consecutive (distance 2). So v must not be adjacent to path[1].
            # v and path[i] for i >= 2 must not be adjacent.
            for i in range(1, len(path)):
                if (v ^ path[i]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.insert(0, v)
                used.add(v)
                extended = True
                break
                
    return extended

def _try_insert(path, used, v):
    """
    Try to insert v at some position in the path. Returns new path or None.
    """
    L = len(path)
    for i in range(L - 1):
        a, b = path[i], path[i+1]
        if (v ^ a).bit_count() != 1 or (v ^ b).bit_count() != 1:
            continue
        # Check v against all other path vertices
        ok = True
        for j in range(L):
            if j == i or j == i+1:
                continue
            if (v ^ path[j]).bit_count() == 1:
                ok = False
                break
        if not ok:
            continue
        new_path = path[:i+1] + [v] + path[i+1:]
        return new_path
    return None

def construct():
    """
    Construct the longest induced path in Q_9.
    Strategy:
    1. Initialize with a greedy snake.
    2. Iterated Local Search:
       - Try to extend from ends.
       - Try to insert vertices.
       - If stuck, truncate the tail and re-extend greedily (randomized).
    3. Keep track of the best path found.
    """
    deadline = time.time() + 115  # Safety margin
    rng = random.Random(42)
    
    # Initial greedy snake
    best_path = [0]
    best_used = {0}
    best_path, best_used = _greedy_extend(best_path, best_used, rng)
    best_len = len(best_path)
    
    # Try a few random greedy starts to find a better initial solution
    for s in range(1, 50):
        if time.time() > deadline - 10:
            break
        p = [0]
        u = {0}
        p, u = _greedy_extend(p, u, rng)
        if len(p) > best_len:
            best_path = p
            best_used = u
            best_len = len(p)
            
    # Main ILS Loop
    current_path = list(best_path)
    current_used = set(best_used)
    
    while time.time() < deadline - 2:
        # 1. Try to extend from ends
        if _try_extend_ends(current_path, current_used):
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_used = set(current_used)
                best_len = len(current_path)
            continue # Restart the loop to try extending again
        
        # 2. Try to insert a vertex
        # Find candidates: vertices adjacent to some edge in the path
        candidates = set()
        for i in range(len(current_path) - 1):
            a, b = current_path[i], current_path[i+1]
            diff_bit = (a ^ b).bit_length() - 1
            for l in range(N):
                if l == diff_bit:
                    continue
                v = b ^ (1 << l)
                if v not in current_used:
                    candidates.add(v)
        
        inserted = False
        for v in list(candidates):
            if time.time() > deadline - 2:
                break
            new_path = _try_insert(current_path, current_used, v)
            if new_path is not None:
                current_path = new_path
                current_used = set(current_path)
                inserted = True
                if len(current_path) > best_len:
                    best_path = list(current_path)
                    best_used = set(current_used)
                    best_len = len(current_path)
                break # Restart loop after one insertion
        
        if inserted:
            continue
            
        # 3. Truncate and Re-extend (Perturbation)
        # Randomly truncate 1 to 5 vertices from the end
        trunc_amount = rng.randint(1, 5)
        if len(current_path) > trunc_amount + 5: # Don't truncate too much if path is short
            current_path = current_path[:-trunc_amount]
            # Rebuild used set
            current_used = set(current_path)
            
            # Re-extend greedily
            current_path, current_used = _greedy_extend(current_path, current_used, rng)
            
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_used = set(current_used)
                best_len = len(current_path)
                
            # If the re-extended path is not better, we might want to keep the truncated version 
            # to try different extensions next time? 
            # For now, we just continue. The next iteration will try to extend this new state.
            # If it's worse than best, we might want to revert to best? 
            # ILS usually accepts worse solutions to escape local minima.
            # Let's accept the new state regardless, as it's a perturbation.
            
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
