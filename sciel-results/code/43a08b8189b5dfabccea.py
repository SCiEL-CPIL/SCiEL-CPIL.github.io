import random
import time

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

    def greedy_extend(path, used_mask, last_v, priority):
        """Try to greedily extend a partial path."""
        path = list(path)
        used_mask = used_mask
        cur = last_v
        for _ in range(300):
            if time.time() > deadline:
                break
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
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
            used_mask |= (1 << nxt)
            cur = nxt
        return path, used_mask

    def local_search(path, used_mask, last_v, priority, max_iterations=1000):
        """
        Perform local search by removing and re-inserting vertices.
        This helps escape local optima by allowing the path to 'breathe' and reconfigure.
        """
        path = list(path)
        used_mask = used_mask
        cur = last_v
        
        # We'll try to improve the path by removing a vertex and re-inserting it.
        # To keep it simple and fast, we'll focus on extending the path.
        # If we can't extend, we try to remove a vertex that is "blocking" an extension.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to extend
            new_path, new_used = greedy_extend(path, used_mask, cur, priority)
            if len(new_path) > len(path):
                path = new_path
                used_mask = new_used
                cur = path[-1]
                improved = True
                continue
            
            # 2. If stuck, try to remove a vertex and re-insert
            # We look for a vertex that, if removed, allows a new extension.
            # To be efficient, we only try removing vertices near the end or randomly.
            # Let's try removing a random vertex (not the last one) and see if we can extend.
            
            if len(path) < 2:
                break
                
            # Pick a random index to remove (avoid the last one as it's the current end)
            # Actually, removing the last one just shortens it, which is not helpful for extension.
            # We want to remove an internal vertex to "open up" space.
            # However, re-inserting is complex. 
            # A simpler heuristic: "Backtrack and Retry"
            # If we are stuck, go back 1 step and try a different branch.
            
            # Let's implement a simple "kick":
            # Remove the last vertex, and try to extend from the new end.
            # If that fails, remove the second-to-last, etc.
            
            # But this is just backtracking.
            # The "2-opt" idea is more powerful.
            # Let's try: Remove vertex at index i, then try to re-insert it at the end? No, it must be connected.
            
            # Alternative: "Cycle Move"
            # Find a vertex v in the path such that there is an unused vertex u adjacent to v,
            # and u is not adjacent to any other vertex in the path.
            # If we can find such a pair, we might be able to "pivot".
            
            # Let's stick to a simpler but effective strategy:
            # Randomly remove a vertex from the middle, then greedily extend.
            # If the extension is longer than the original, accept it.
            
            if len(path) > 10:
                # Pick a random index to remove
                # We must ensure the path remains valid after removal.
                # Removing path[i] requires path[i-1] and path[i+1] to be non-adjacent.
                # In a hypercube, path[i-1] and path[i+1] differ by 2 bits (since path[i] differs from both by 1 bit).
                # So they are NEVER adjacent. Thus, removing any internal vertex always yields a valid path segment.
                
                idx = rnd.randint(1, len(path) - 2)
                removed_v = path[idx]
                
                # Create new path
                new_path = path[:idx] + path[idx+1:]
                new_used_mask = used_mask & ~(1 << removed_v)
                
                # The new end is still path[-1] (unless we removed the last one, which we don't)
                # But wait, the path is a sequence. Removing an internal vertex doesn't change the end.
                # So we can just try to extend from the current end.
                
                extended_path, extended_used = greedy_extend(new_path, new_used_mask, new_path[-1], priority)
                
                if len(extended_path) > len(path):
                    path = extended_path
                    used_mask = extended_used
                    cur = path[-1]
                    improved = True
                    continue

            # If no improvement found in this iteration, break to avoid infinite loop
            # Or try a few more random removals
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                if len(path) > 10:
                    idx = rnd.randint(1, len(path) - 2)
                    removed_v = path[idx]
                    new_path = path[:idx] + path[idx+1:]
                    new_used_mask = used_mask & ~(1 << removed_v)
                    extended_path, extended_used = greedy_extend(new_path, new_used_mask, new_path[-1], priority)
                    if len(extended_path) > len(path):
                        path = extended_path
                        used_mask = extended_used
                        cur = path[-1]
                        improved = True
                        break

        return path, used_mask

    # Phase 0: Quick greedy baseline
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

    seed_path = greedy(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # Phase 1: Local Search with Randomized Greedy
    # We run multiple instances of the greedy heuristic, and for each, we apply local search.
    
    for i in range(100):
        if time.time() > deadline - 10.0:
            break
            
        # Randomize the priority order for greedy
        priority = list(range(N))
        rnd.shuffle(priority)
        
        # Get a greedy path
        p, used_mask = greedy_extend([0], 1, 0, priority)
        
        # Apply local search
        p, used_mask = local_search(p, used_mask, p[-1], priority)
        
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p

    if best_path is None:
        return [0]
    return best_path
