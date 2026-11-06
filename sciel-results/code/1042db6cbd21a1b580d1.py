"""Snake-in-the-box in the 9-dimensional hypercube - Backtracking with bitmask state."""
N = 9
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Precompute neighbors for each vertex
    # neighbors[v] = list of vertices at Hamming distance 1 from v
    # forbidden_mask tracks which vertices are "blocked" (cannot be next in path)
    # A vertex is blocked if it's at distance 1 from any non-last vertex in path
    
    # For efficient state management, we track:
    # - path: list of vertices in order
    # - used: set of vertices already in path  
    # - blocked: set of vertices that cannot be added (dist 1 from path[:-1])
    # When we add vertex v to end of path:
    #   - v must not be in used or blocked
    #   - new blocked = blocked ∪ {neighbors of path[-2]} (since path[-2] becomes non-consecutive)
    #   - but we need to be careful: only vertices at dist 1 from path[-2] that aren't in path
    #   - Actually: when we extend path by v, the vertex path[-2] (second to last) 
    #     becomes non-consecutive to future vertices, so its neighbors become blocked
    #   - But some neighbors might already be in path, that's fine
    
    # Better approach: maintain a "forbidden" bitmask where bit i is set if 
    # vertex i cannot be the next vertex in the path.
    # forbidden includes: used vertices + vertices at dist 1 from path[:-1]
    # When extending by v: new_forbidden = forbidden | {all neighbors of path[-2]}
    # But we also need v not to be in forbidden
    
    # Let's use a different representation:
    # - path: list
    # - forbidden: int bitmask (2^9 bits), bit i set means vertex i is forbidden
    # Initially: forbidden has bit 0 set (vertex 0 is used)
    # When we add v to path:
    #   - Check: bit v of forbidden must be 0
    #   - new_forbidden = forbidden | neighbor_mask[path[-2]]
    #   where neighbor_mask[x] is a bitmask of all vertices at Hamming distance 1 from x
    #   - But we also need to mark v as used, so: new_forbidden |= (1 << v)
    #   - Wait, that's not quite right. Let me think again.
    #
    # State: path = [v0, v1, ..., vk]
    # Forbidden for next vertex: 
    #   - All vi for i <= k (used)
    #   - All vertices at dist 1 from v0, v1, ..., v_{k-1} (non-consecutive to next)
    #   - Note: vertices at dist 1 from vk are NOT forbidden (they're consecutive candidates)
    #
    # So forbidden_mask = used_mask | union_{i=0}^{k-1} neighbor_mask[vi]
    #
    # When we extend by v_{k+1}:
    #   - v_{k+1} must not be in forbidden_mask
    #   - new_forbidden = used_mask | (1 << v_{k+1}) | union_{i=0}^{k} neighbor_mask[vi]
    #   - = old_forbidden | (1 << v_{k+1}) | neighbor_mask[vk]
    #   - Because old_forbidden already has union_{i=0}^{k-1} neighbor_mask[vi] and used_mask
    #   - We add (1 << v_{k+1}) for the new used vertex
    #   - We add neighbor_mask[vk] because vk becomes non-consecutive
    #
    # This is O(1) per step!
    
    # Precompute neighbor masks
    neighbor_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            nv = v ^ (1 << b)
            mask |= (1 << nv)
        neighbor_mask[v] = mask
    
    bit_masks = [1 << b for b in range(N)]
    
    def _backtrack_search(start_vertex, deadline, max_depth=None, rng=None):
        """Backtracking search with Warnsdorff heuristic using bitmask state."""
        if rng is None:
            rng = random.Random(42)
        
        best_path = [start_vertex]
        best_len = 1
        
        # State: path, forbidden_mask
        # forbidden_mask bit i set = vertex i cannot be next
        # Initially: forbidden = (1 << start_vertex) | neighbor_mask[start_vertex]
        # Wait no. Initially path = [start_vertex].
        # Forbidden for next: used = {start_vertex}, non-consecutive neighbors = {} (none yet)
        # So forbidden = (1 << start_vertex)
        # When we add v1: forbidden becomes (1 << start_vertex) | (1 << v1) | neighbor_mask[start_vertex]
        
        def _search(path, forbidden_mask, depth):
            nonlocal best_path, best_len
            
            if time.time() > deadline:
                return
            
            if depth > best_len:
                best_len = depth
                best_path = path[:]
            
            if max_depth is not None and depth >= max_depth:
                return
            
            cur = path[-1]
            
            # Find candidate next vertices
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                # v is a valid candidate
                # Compute onward moves for Warnsdorff
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, b, v))
            
            if not candidates:
                return
            
            # Sort by onward moves (Warnsdorff: fewest onward first)
            candidates.sort(key=lambda x: x[0])
            
            # Try each candidate
            for onward, b, v in candidates:
                if time.time() > deadline:
                    return
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                path.append(v)
                _search(path, new_forbidden, depth + 1)
                path.pop()
        
        _search([start_vertex], (1 << start_vertex), 1)
        return best_path
    
    def _greedy_extend(path, deadline, rng=None):
        """Greedily extend a path using Warnsdorff with bitmask state."""
        if rng is None:
            rng = random.Random(42)
        
        if not path:
            return path
        
        # Reconstruct forbidden mask from path
        forbidden_mask = 0
        for v in path:
            forbidden_mask |= (1 << v)
        for v in path[:-1]:
            forbidden_mask |= neighbor_mask[v]
        
        cur = path[-1]
        while time.time() < deadline:
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, b, v))
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x[0])
            _, b, v = candidates[0]
            
            path.append(v)
            forbidden_mask |= (1 << v) | neighbor_mask[cur]
            cur = v
        
        return path
    
    # Phase 1: Backtracking search from multiple starts
    rng = random.Random(12345)
    
    starts = [0, 1, (1 << N) - 1, 0x155, 0x1AA, 0x0FF, 0x1F0, 0x123, 0x1AB]
    for _ in range(20):
        starts.append(rng.randint(0, (1 << N) - 1))
    
    best_path = None
    best_len = 0
    
    for sv in starts:
        if time.time() > deadline - 30:
            break
        # Give each backtracking search a time budget
        search_deadline = min(deadline - 5, time.time() + 2.0)
        path = _backtrack_search(sv, search_deadline, max_depth=120, rng=rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 2: Greedy extension from best path and random starts
    if best_path:
        extended = _greedy_extend(best_path[:], deadline - 10, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], deadline - 5, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 3: More backtracking with higher depth limits from promising starts
    if best_path:
        # Try to extend from both ends
        for end in [0, 1]:
            if time.time() > deadline - 5:
                break
            if end == 0:
                # Reverse path and extend from new end
                rev_path = best_path[::-1]
                extended = _greedy_extend(rev_path, deadline - 5, rng)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = extended[::-1]
            else:
                extended = _greedy_extend(best_path[:], deadline - 5, rng)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = extended
    
    # Phase 4: Final backtracking attempts with more time
    if time.time() < deadline - 3:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            sv = rng.randint(0, (1 << N) - 1)
            search_deadline = min(deadline - 1, time.time() + 1.0)
            path = _backtrack_search(sv, search_deadline, max_depth=130, rng=rng)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    # Final: ensure we have at least a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2
    
    return best_path
