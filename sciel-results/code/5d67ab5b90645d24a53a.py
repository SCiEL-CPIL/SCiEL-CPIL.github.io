"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 9

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    
    def validate_path(path):
        """Check if path is a valid snake (induced path in hypercube)."""
        if len(path) < 2:
            return True
        used = set(path)
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                if abs(i - j) == 1:
                    if (path[i] ^ path[j]).bit_count() != 1:
                        return False
                else:
                    if (path[i] ^ path[j]).bit_count() == 1:
                        return False
        return True

    def greedy_extend(path, used, bit_order):
        """Extend path greedily using given bit order."""
        cur = path[-1]
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v conflicts with any non-adjacent vertex
            conflict = False
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    conflict = True
                    break
            if not conflict:
                path.append(v)
                used.add(v)
                return True
        return False

    def backtrack_search(path, used, depth_limit, bit_order):
        """Backtrack search with limited depth."""
        if time.time() > deadline:
            return False
        if len(path) >= depth_limit:
            return True
        
        cur = path[-1]
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            conflict = False
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    conflict = True
                    break
            if not conflict:
                candidates.append((b, v))
        
        if not candidates:
            return False
        
        # Try candidates in order
        for b, v in candidates:
            path.append(v)
            used.add(v)
            if backtrack_search(path, used, depth_limit, bit_order):
                return True
            path.pop()
            used.remove(v)
            if time.time() > deadline:
                return False
        return False

    # Start with greedy snake
    path = [0]
    used = {0}
    bit_order = list(range(N))
    
    # Initial greedy extension
    while greedy_extend(path, used, bit_order):
        pass
    
    best_path = list(path)
    best_len = len(path)
    
    # Try different bit orderings to escape local optima
    import random
    rng = random.Random(42)
    
    # Generate multiple bit orderings
    orderings = [list(range(N))]
    for _ in range(50):
        orderings.append(rng.sample(range(N), N))
    
    for bit_ord in orderings:
        if time.time() > deadline:
            break
        
        # Reset and try with this ordering
        path = [0]
        used = {0}
        
        # Greedy extend
        while greedy_extend(path, used, bit_ord):
            pass
        
        # If we got longer, try to extend further with backtracking
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Try backtracking to extend further
        depth_limit = best_len + 10
        if backtrack_search(path, used, depth_limit, bit_ord):
            if len(path) > best_len:
                best_path = list(path)
                best_len = len(path)
    
    # Final greedy extension on best path
    path = list(best_path)
    used = set(path)
    for bit_ord in [list(range(N))] + orderings[:10]:
        if time.time() > deadline:
            break
        while greedy_extend(path, used, bit_ord):
            pass
    
    if len(path) > best_len:
        best_path = path
        best_len = len(path)
    
    return best_path
# EVOLVE-BLOCK-END
