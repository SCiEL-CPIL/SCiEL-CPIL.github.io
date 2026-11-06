"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    
    def is_valid_path(path):
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

    def get_candidates(cur, path, used):
        """Get all valid next vertices from cur given current path."""
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v conflicts with any non-adjacent vertex (path[:-1] excludes the immediate predecessor)
            conflict = False
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    conflict = True
                    break
            if not conflict:
                candidates.append((b, v))
        return candidates

    def greedy_extend(path, used, bit_order, deadline):
        """Extend path greedily using given bit order."""
        while True:
            cur = path[-1]
            extended = False
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
                    path.append(v)
                    used.add(v)
                    extended = True
                    break
            if not extended or time.time() > deadline:
                break

    def backtrack_search(path, used, depth_limit, bit_order, deadline):
        """
        Backtrack search with forward checking and fail-first heuristic.
        Returns True if a path of length >= depth_limit is found.
        """
        if time.time() > deadline:
            return False
        if len(path) >= depth_limit:
            return True
        
        cur = path[-1]
        
        # Get candidates and sort by "fail-first" heuristic:
        # Prefer candidates that leave fewer options for the next step (constraining first)
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
                # Count how many options v would leave for the next step
                # This is a heuristic to find dead ends earlier
                future_options = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 in used or v2 == cur:
                        continue
                    # Quick check: does v2 conflict with path[:-1]?
                    ok = True
                    for p in path[:-1]:
                        if (v2 ^ p).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        future_options += 1
                candidates.append((future_options, b, v))
        
        if not candidates:
            return False
        
        # Sort by ascending future_options (fail-first)
        candidates.sort(key=lambda x: x[0])
        
        for _, b, v in candidates:
            path.append(v)
            used.add(v)
            if backtrack_search(path, used, depth_limit, bit_order, deadline):
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
    greedy_extend(path, used, bit_order, deadline)
    
    best_path = list(path)
    best_len = len(path)
    
    # Try different bit orderings to escape local optima
    rng = random.Random(42)
    
    # Generate multiple bit orderings
    orderings = [list(range(N))]
    for _ in range(100):
        orderings.append(rng.sample(range(N), N))
    
    for bit_ord in orderings:
        if time.time() > deadline:
            break
        
        # Reset and try with this ordering
        path = [0]
        used = {0}
        
        # Greedy extend
        greedy_extend(path, used, bit_ord, deadline)
        
        # If we got longer, try to extend further with backtracking
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Try backtracking to extend further
        # Aim for a significant improvement over current best
        target_len = best_len + 5
        if backtrack_search(path, used, target_len, bit_ord, deadline):
            if len(path) > best_len:
                best_path = list(path)
                best_len = len(path)
    
    # Final greedy extension on best path
    path = list(best_path)
    used = set(path)
    for bit_ord in [list(range(N))] + orderings[:10]:
        if time.time() > deadline:
            break
        greedy_extend(path, used, bit_ord, deadline)
    
    if len(path) > best_len:
        best_path = path
        best_len = len(path)
    
    return best_path
# EVOLVE-BLOCK-END
