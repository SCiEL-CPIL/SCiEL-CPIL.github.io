import time
import numpy as np
from collections import defaultdict

N = 9
TOTAL = 1 << N
MASK = TOTAL - 1

# Precompute neighbor lists for fast lookup
_NEIGHBORS = np.array([[v ^ (1 << b) for b in range(N)] for v in range(TOTAL)], dtype=np.uint32)

def _elapsed(deadline):
    return time.time() >= deadline

def _is_valid_insertion(path, pos, v, used_set):
    """Check if vertex v can be inserted at position pos in path."""
    if v in used_set:
        return False
    
    # Check v against all path vertices except neighbors at pos-1 and pos
    for i, p in enumerate(path):
        if i == pos - 1 or i == pos:
            continue
        if (v ^ p).bit_count() == 1:
            return False
    
    # Check if v is adjacent to the vertices it would connect to
    if pos > 0 and (v ^ path[pos-1]).bit_count() != 1:
        return False
    if pos < len(path) and (v ^ path[pos]).bit_count() != 1:
        return False
    
    return True

def _local_search(path, deadline, rng):
    """Perform local search with simulated annealing to improve path length."""
    current = list(path)
    best = list(current)
    best_len = len(current)
    
    used_set = set(current)
    
    # Build a set of all unused vertices that could potentially be added
    all_vertices = set(range(TOTAL))
    unused = all_vertices - used_set
    
    # Temperature schedule
    T0 = 1.0
    T_end = 0.01
    cooling = 0.999
    
    T = T0
    
    # Precompute for speed: for each vertex, which path vertices are at distance 1
    # This helps quickly reject invalid insertions
    
    while not _elapsed(deadline) and T > T_end:
        # Try random insertion of an unused vertex
        if unused and rng.random() < 0.7:
            # Pick a random unused vertex
            v = int(rng.choice(list(unused), size=1)[0])
            
            # Try all possible positions
            best_pos = None
            for pos in range(len(current) + 1):
                if _is_valid_insertion(current, pos, v, used_set):
                    best_pos = pos
                    break
            
            if best_pos is not None:
                new_path = current[:best_pos] + [v] + current[best_pos:]
                new_len = len(new_path)
                delta = new_len - len(current)
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    # Accept move
                    current = new_path
                    used_set.add(v)
                    unused.discard(v)
                    
                    if len(current) > best_len:
                        best_len = len(current)
                        best = list(current)
        else:
            # Try swap: remove a vertex and reinsert elsewhere
            if len(current) > 2:
                # Pick a random vertex to remove
                idx = int(rng.integers(1, len(current) - 1))  # Don't remove endpoints
                v = current[idx]
                
                new_path = current[:idx] + current[idx+1:]
                # The path is now broken: new_path[idx-1] and new_path[idx] need to be adjacent
                # Actually, removing a middle vertex breaks the path unless its neighbors are adjacent
                
                # Check if removing v keeps a valid path
                if idx > 0 and idx < len(current) - 1:
                    left = current[idx-1]
                    right = current[idx+1]
                    if (left ^ right).bit_count() == 1:
                        # Valid removal
                        new_used = used_set - {v}
                        
                        # Try to reinsert v at a different position
                        best_pos = None
                        for pos in range(len(new_path) + 1):
                            if _is_valid_insertion(new_path, pos, v, new_used):
                                best_pos = pos
                                break
                        
                        if best_pos is not None:
                            final_path = new_path[:best_pos] + [v] + new_path[best_pos:]
                            delta = len(final_path) - len(current)
                            
                            if delta >= 0 or rng.random() < np.exp(delta / T):
                                current = final_path
                                # used_set unchanged since we removed and reinserted v
                                
                                if len(current) > best_len:
                                    best_len = len(current)
                                    best = list(current)
        
        T *= cooling
    
    return best

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    rng = np.random.default_rng(seed=42)
    
    # Start with a greedy path to get a decent initial solution
    best_path = None
    best_len = 0
    
    # Generate multiple initial greedy paths
    for restart in range(50):
        if _elapsed(deadline):
            break
        
        start_v = int(rng.integers(TOTAL))
        
        # Forward greedy
        path = [start_v]
        used = {start_v}
        
        while not _elapsed(deadline):
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Also try backward greedy from start_v
        path2 = [start_v]
        used2 = {start_v}
        
        while not _elapsed(deadline):
            cur = path2[0]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used2:
                    continue
                ok = True
                for p in path2[1:]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            path2.insert(0, nxt)
            used2.add(nxt)
        
        if len(path2) > best_len:
            best_len = len(path2)
            best_path = list(path2)
    
    if best_path is None:
        return [0]
    
    # Now do local search on the best path found
    # Allocate remaining time for local search
    elapsed = time.time() - (deadline - 110)
    remaining = 110 - elapsed
    
    if remaining > 10:
        # Do local search with a portion of the time
        ls_deadline = time.time() + min(remaining * 0.8, 80)
        improved = _local_search(best_path, ls_deadline, rng)
        
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
    
    return best_path
