"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    def is_valid_extension(v, path, used):
        if v in used:
            return False
        # Check against all non-consecutive vertices in path
        # Consecutive in the new context would be path[i-1] and path[i+1] if inserted,
        # but here we are extending, so we check against all existing path vertices
        # except the immediate predecessor (which is path[-1])
        # Wait, the check is: non-consecutive vertices differ in at least two bits.
        # If we add v to end of path, v is consecutive to path[-1].
        # So v must differ from path[i] by >= 2 bits for all i < len(path)-1.
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True

    def greedy_extend(path, used, bit_order):
        """Greedy extension with random bit order."""
        while True:
            if time.time() > deadline:
                return path
            cur = path[-1]
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(v, path, used):
                    candidates.append(v)
            if not candidates:
                return path
            best_v = random.choice(candidates)
            path.append(best_v)
            used.add(best_v)

    def find_detour(start_v, end_v, forbidden, max_len=8):
        """
        Find a path from start_v to end_v of length <= max_len
        that does not use any vertex in 'forbidden'.
        The path vertices (excluding start_v and end_v) must be valid extensions
        relative to the global context? No, this is for local re-routing.
        
        For a valid re-routing of a segment in an induced path:
        The new internal vertices must not be adjacent to any vertex in the 
        rest of the path (outside the segment) except the immediate neighbors 
        in the new segment.
        
        This function just finds a simple path in the hypercube avoiding forbidden vertices.
        The caller will verify global validity.
        """
        # BFS for shortest path avoiding forbidden
        from collections import deque
        queue = deque([(start_v, [start_v])])
        visited = {start_v}
        
        while queue:
            cur, path = queue.popleft()
            if len(path) > max_len:
                continue
            if cur == end_v:
                return path
            for b in range(N):
                nxt = cur ^ (1 << b)
                if nxt in visited or nxt in forbidden:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    def local_search(path, max_iterations=2000):
        """
        Improve path by trying to replace segments with longer detours.
        """
        best_path = list(path)
        best_len = len(best_path)
        used = set(best_path)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if len(best_path) < 2:
                break
                
            # Pick a random segment to try to extend
            # We want to replace a sub-path P[i:j] with a longer path P'[i:j]
            # such that P'[i] == P[i] and P'[j] == P[j]
            # The internal vertices of P' must be disjoint from the rest of the path.
            
            # Strategy: Pick two vertices that are close in the path but far in Hamming distance?
            # Or just pick a random short segment and try to detour.
            
            # Let's try to extend a single edge (i, i+1) by replacing it with a path of length 2 or 3.
            # Or replace a segment of length 1 (edge) with a path of length 3 (2 edges).
            
            if len(best_path) < 3:
                break
                
            # Randomly select an index i where we can try to insert a vertex between i and i+1
            # Or replace segment i..j with a longer path.
            
            # Simple move: Try to replace edge (i, i+1) with a path of length 2 or 3.
            # i.e., find v such that i - v - i+1 is a valid sub-path.
            
            # Let's try to replace a segment of length L with a segment of length L+1 or L+2.
            # Start with small segments.
            
            segment_len = random.randint(1, min(3, len(best_path)-1))
            i = random.randint(0, len(best_path) - segment_len - 1)
            j = i + segment_len
            
            start_v = best_path[i]
            end_v = best_path[j]
            
            # Forbidden vertices: all vertices in best_path except the internal ones of the segment
            # Actually, the new internal vertices must not be in the rest of the path.
            # The rest of the path is best_path[:i] and best_path[j:]
            # The internal vertices of the old segment are best_path[i+1:j]
            # We are replacing the WHOLE segment i..j? No, we keep start_v and end_v.
            # So forbidden = set(best_path) - {start_v, end_v}
            # Wait, the new path must not intersect the rest of the path.
            # So forbidden = set(best_path[:i]) | set(best_path[j:])
            
            forbidden = set(best_path[:i]) | set(best_path[j:])
            
            # We want a path from start_v to end_v with length > segment_len
            # Let's try to find a path of length segment_len + 1 or + 2
            # Using BFS to find a simple path avoiding forbidden
            
            # Since N=9, the hypercube is small.
            # We can just do a DFS/BFS for a path of specific length.
            
            # Let's try to find ANY path from start_v to end_v avoiding forbidden
            # that is longer than the direct segment.
            
            # Optimization: If start_v and end_v are adjacent (segment_len=1), 
            # we look for a path of length 2 or 3.
            
            # BFS with path tracking
            from collections import deque
            # State: (current_vertex, path_so_far)
            # We want path length > segment_len
            # Max length to search: segment_len + 3 (small detours first)
            
            found = None
            # Use BFS to find shortest path first, then check if we can find a longer one?
            # Actually, we just want ANY valid extension.
            # If we find a path of length L > segment_len, we can replace.
            
            # To ensure we find a longer path, we can just do BFS and record paths.
            # But BFS finds shortest. If shortest is segment_len, we need to explore further.
            # Let's do DFS with depth limit = segment_len + 3.
            
            def dfs_detour(cur, path_so_far, depth_limit):
                if len(path_so_far) > segment_len:
                    return path_so_far
                if len(path_so_far) >= depth_limit:
                    return None
                if cur == end_v:
                    return None # We need strictly longer, so if we hit end_v early, it's not a detour
                    # Wait, if we hit end_v, the path is valid but length might be <= segment_len.
                    # We require length > segment_len.
                
                for b in range(N):
                    nxt = cur ^ (1 << b)
                    if nxt in forbidden:
                        continue
                    if nxt in path_so_far:
                        continue
                    # Check global validity for new vertex nxt?
                    # The new vertex nxt is only connected to prev and next in the new segment.
                    # It must not be adjacent to any vertex in the rest of the path.
                    # The rest of the path is forbidden.
                    # So we just need to ensure nxt is not adjacent to any vertex in forbidden?
                    # Yes, because in the final induced path, nxt will be non-consecutive to all vertices in forbidden.
                    # So (nxt ^ v).bit_count() must be != 1 for all v in forbidden.
                    
                    is_adj_to_forbidden = False
                    for v in forbidden:
                        if (nxt ^ v).bit_count() == 1:
                            is_adj_to_forbidden = True
                            break
                    if is_adj_to_forbidden:
                        continue
                    
                    res = dfs_detour(nxt, path_so_far + [nxt], depth_limit)
                    if res:
                        return res
                return None

            depth_limit = segment_len + 3
            found_path = dfs_detour(start_v, [start_v], depth_limit)
            
            if found_path and len(found_path) > segment_len + 1: # +1 because found_path includes start_v, so edges = len-1
                # Replace segment
                new_path = best_path[:i] + found_path + best_path[j:]
                # Verify validity of the whole new path?
                # The local check ensured no adjacency to forbidden.
                # The internal vertices of found_path are simple.
                # The connections at i and j are preserved.
                # So it should be valid.
                
                if len(new_path) > best_len:
                    best_path = new_path
                    best_len = len(new_path)
                    used = set(best_path) # Update used set? Not strictly needed if we recompute forbidden each time, but good for consistency if we used it.
                    # We recompute forbidden from best_path each time, so used set is not strictly necessary for the search, 
                    # but we keep it for potential other uses or if we switch to a set-based check.
                    # Actually, is_valid_extension uses 'used'. Here we use 'forbidden' derived from best_path.
                    # So we don't need to update 'used' explicitly unless we use it elsewhere.
        
        return best_path

    best_path = []
    
    # Phase 1: Generate initial greedy snakes
    for _ in range(100):
        if time.time() > deadline:
            break
        random_bit_order = list(range(N))
        random.shuffle(random_bit_order)
        path = [0]
        used = {0}
        path = greedy_extend(path, used, random_bit_order)
        if len(path) > len(best_path):
            best_path = path

    # Phase 2: Local Search Improvement
    if time.time() < deadline:
        improved_path = local_search(best_path)
        if len(improved_path) > len(best_path):
            best_path = improved_path
            
    # Phase 3: More Local Search from different starts if time permits
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            random_bit_order = list(range(N))
            random.shuffle(random_bit_order)
            path = [0]
            used = {0}
            path = greedy_extend(path, used, random_bit_order)
            
            # Quick local search
            # Use a smaller max_iterations for these
            improved_path = local_search(path, max_iterations=200)
            
            if len(improved_path) > len(best_path):
                best_path = improved_path

    return best_path
# EVOLVE-BLOCK-END
