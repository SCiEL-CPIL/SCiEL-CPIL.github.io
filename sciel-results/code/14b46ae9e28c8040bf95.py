import random
import time
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute for each vertex: bitmask of all neighbors
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(20260715)
    best_path = None
    best_len = 0

    def validate_and_score(path):
        """Check validity and return edge count."""
        if not path:
            return 0
        n = len(path)
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return -1
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return -1
            if i >= 2:
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return -1
            used_mask |= (1 << v)
        return n - 1

    def greedy(priority):
        path = [0]
        used = {0}
        used_mask = 1
        for _ in range(300):
            if time.time() > deadline:
                break
            cur = path[-1]
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            used_mask |= (1 << nxt)
        return path

    # Initial seed
    seed_path = greedy(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # VNS Strategy
    # 1. Start with a valid path
    # 2. Try to extend it using greedy
    # 3. Apply local search moves (2-opt, reversal, insertion) to improve
    # 4. If stuck, perturb and restart local search
    
    def extend_greedy(path, priority):
        if not path:
            return path
        used = set(path)
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        cur = path[-1]
        new_path = list(path)
        for _ in range(300):
            if time.time() > deadline:
                break
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            new_path.append(nxt)
            used.add(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        return new_path

    def local_search(path):
        """
        Perform local search on a valid path.
        Moves:
        1. 2-opt: Reverse a segment.
        2. Single vertex re-insertion: Remove a vertex and re-insert it elsewhere.
        """
        if not path:
            return path
        
        improved = True
        current_path = list(path)
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try 2-opt moves
            # A 2-opt move in a path is equivalent to reversing a segment.
            # But wait, in a snake, reversing a segment [i...j] changes the connectivity at i-1 and j.
            # Original: ... A - u_i - ... - u_j - B ...
            # New: ... A - u_j - ... - u_i - B ...
            # This is valid if:
            # 1. A is adjacent to u_j (was adjacent to u_i)
            # 2. B is adjacent to u_i (was adjacent to u_j)
            # 3. No new conflicts are created inside the segment (reversal preserves internal adjacency)
            # 4. No new conflicts are created with the rest of the path? 
            #    Actually, since the set of vertices is the same, and the internal structure is just reversed,
            #    the only new adjacencies are at the boundaries.
            
            n = len(current_path)
            found_2opt = False
            for i in range(1, n - 1):
                if time.time() > deadline - 5:
                    return current_path
                for j in range(i + 1, n):
                    # Check if reversing segment [i:j] is valid
                    # Boundaries: current_path[i-1] and current_path[j] (if j < n-1)
                    # Wait, if we reverse [i:j], the new neighbors are:
                    # Left: current_path[i-1] connects to current_path[j]
                    # Right: current_path[j+1] (if exists) connects to current_path[i]
                    
                    u_left = current_path[i-1]
                    u_right = current_path[j+1] if j < n-1 else None
                    
                    # Check left connection
                    if (u_left ^ current_path[j]).bit_count() != 1:
                        continue
                    
                    # Check right connection
                    if u_right is not None:
                        if (u_right ^ current_path[i]).bit_count() != 1:
                            continue
                    else:
                        # If j is the last element, we just need to ensure the new last element (current_path[i]) 
                        # is not adjacent to any other vertex in the path except its new neighbor (current_path[i+1] in reversed order, which is current_path[i-1]? No.)
                        # Let's just do a full validation for simplicity if n is small, or careful checks.
                        # Actually, if j == n-1, the new end is current_path[i].
                        # It was previously connected to current_path[i+1] (if i < n-1) or was an end.
                        # In the reversed segment, current_path[i] is connected to current_path[i+1] (which is now its predecessor in the segment? No, successor in original, predecessor in reversed).
                        # Let's just validate the whole path. It's O(N^2) which is 512^2 ~ 260k ops. Might be too slow if done for every pair.
                        # But we only do it if boundary checks pass.
                        pass
                    
                    # If boundary checks pass, validate the whole path
                    new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
                    if validate_and_score(new_path) > 0:
                        current_path = new_path
                        improved = True
                        found_2opt = True
                        break
                if found_2opt:
                    break
            
            if improved:
                continue
                
            # Try single vertex re-insertion
            # Remove vertex at index k, and insert it at index m.
            # This is more complex to check incrementally.
            # Given time constraints, we might skip this or do it sparsely.
            
            # If no 2-opt found, try a random perturbation?
            # For now, just break out of local search if no 2-opt improvement.
            break
            
        return current_path

    # Main VNS Loop
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
            
        # Start from best found so far
        current_path = list(best_path)
        
        # 1. Extend greedily
        # Use a random priority order for extension
        priority = list(range(N))
        rnd.shuffle(priority)
        extended_path = extend_greedy(current_path, priority)
        
        # 2. Local Search
        improved_path = local_search(extended_path)
        
        # 3. Update best
        L = validate_and_score(improved_path)
        if L > best_len:
            best_len = L
            best_path = improved_path
            
        # 4. Perturbation: If we didn't improve, try to break out of local optimum
        # by removing a random internal vertex and re-optimizing?
        # Or just restart with a different greedy priority.
        # The current loop already uses random priority for extension.
        
        # Additional Strategy: Try to insert "missing" vertices?
        # This is hard.
        
        # Let's add a simple "kick" if no improvement in last few iterations.
        # For now, rely on randomization of greedy priority.

    return best_path
