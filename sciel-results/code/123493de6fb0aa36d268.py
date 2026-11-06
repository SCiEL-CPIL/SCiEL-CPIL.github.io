"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Precompute neighbors for all vertices
    def get_neighbors(v):
        return [v ^ (1 << b) for b in range(N)]
    
    def is_valid_extension(v, path, used, forbidden_mask):
        """Check if v can be appended to path.
        forbidden_mask: bitmask of vertices that would be at Hamming distance 1 from v
        and are in the path (excluding the last vertex which is allowed).
        """
        if v in used:
            return False
        # Check against all path vertices except the last one
        # The last vertex is the current end, so distance 1 is allowed
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used, bit_order):
        """Extend path greedily using given bit order."""
        while time.time() < deadline:
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(v, path, used, None):
                    nxt = v
                    break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
        return path
    
    def backtrack_extend(path, used, max_depth=50):
        """Try to extend path with limited backtracking."""
        if time.time() > deadline:
            return path
        
        cur = path[-1]
        neighbors = [cur ^ (1 << b) for b in range(N)]
        # Shuffle for variety
        random.shuffle(neighbors)
        
        for v in neighbors:
            if v in used:
                continue
            # Check validity
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Try this extension
            path.append(v)
            used.add(v)
            
            # Recurse with limited depth
            if max_depth > 1:
                result = backtrack_extend(path, used, max_depth - 1)
                if len(result) > len(path):
                    return result
            else:
                # Continue greedy from here
                result = greedy_extend(path, used, list(range(N)))
                return result
            
            # Backtrack
            path.pop()
            used.remove(v)
        
        return path
    
    best_path = []
    
    # Strategy 1: Try different starting vertices with greedy extension
    starting_vertices = [0]
    # Add some random starting vertices
    for _ in range(20):
        starting_vertices.append(random.randint(0, 2**N - 1))
    
    for start_v in starting_vertices:
        if time.time() > deadline:
            break
        
        path = [start_v]
        used = {start_v}
        bit_order = list(range(N))
        random.shuffle(bit_order)
        path = greedy_extend(path, used, bit_order)
        
        if len(path) > len(best_path):
            best_path = path[:].copy()
    
    # Strategy 2: Try backtracking extension from best path
    if time.time() < deadline - 10:
        # Try to improve best_path with backtracking
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            path = best_path[:].copy()
            used = set(path)
            path = backtrack_extend(path, used, max_depth=30)
            if len(path) > len(best_path):
                best_path = path[:].copy()
    
    # Strategy 3: Try different bit orderings more aggressively
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        start_v = random.randint(0, 2**N - 1)
        path = [start_v]
        used = {start_v}
        bit_order = list(range(N))
        random.shuffle(bit_order)
        path = greedy_extend(path, used, bit_order)
        
        if len(path) > len(best_path):
            best_path = path[:].copy()
    
    return best_path
# EVOLVE-BLOCK-END
