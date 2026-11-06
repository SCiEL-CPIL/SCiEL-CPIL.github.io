import time
import random
import sys

N = 9
FULL_MASK = (1 << 512) - 1

# Precompute neighbor masks for all 512 vertices
_NEIGHBOR_MASKS = [0] * (1 << N)
for _v in range(1 << N):
    _m = 0
    for _b in range(N):
        _m |= 1 << (_v ^ (1 << _b))
    _NEIGHBOR_MASKS[_v] = _m

def _neighbors_mask(v):
    """Return a 512-bit mask with bits set for all N neighbors of v."""
    return _NEIGHBOR_MASKS[v]

def _try_extend(path, used_mask, blocked_mask, deadline, direction=1):
    """
    Try to greedily extend the path from one end.
    direction=1: extend forward (add to end)
    direction=-1: extend backward (add to front)
    Returns the extended path.
    """
    extended = list(path)
    ext_used = used_mask
    ext_blocked = blocked_mask
    
    if direction == 1:
        # Extend forward
        while time.time() < deadline:
            cur = extended[-1]
            valid_mask = _neighbors_mask(cur) & ~ext_used & ~ext_blocked
            
            if not valid_mask:
                break
            
            # Pick the vertex with fewest exits (Warnsdorff)
            best_v = -1
            best_score = 100
            vm = valid_mask
            while vm:
                lsb = vm & -vm
                v = lsb.bit_length() - 1
                vm ^= lsb
                
                new_used = ext_used | (1 << v)
                new_blocked = ext_blocked | _neighbors_mask(cur)
                exits = (_neighbors_mask(v) & ~new_used & ~new_blocked).bit_count()
                
                if exits < best_score:
                    best_score = exits
                    best_v = v
            
            if best_v < 0:
                break
            
            extended.append(best_v)
            ext_used |= 1 << best_v
            ext_blocked |= _neighbors_mask(cur)
    else:
        # Extend backward
        while time.time() < deadline:
            cur = extended[0]
            valid_mask = _neighbors_mask(cur) & ~ext_used & ~ext_blocked
            
            if not valid_mask:
                break
            
            # Pick the vertex with fewest exits (Warnsdorff)
            best_v = -1
            best_score = 100
            vm = valid_mask
            while vm:
                lsb = vm & -vm
                v = lsb.bit_length() - 1
                vm ^= lsb
                
                new_used = ext_used | (1 << v)
                new_blocked = ext_blocked | _neighbors_mask(cur)
                exits = (_neighbors_mask(v) & ~new_used & ~new_blocked).bit_count()
                
                if exits < best_score:
                    best_score = exits
                    best_v = v
            
            if best_v < 0:
                break
            
            extended.insert(0, best_v)
            ext_used |= 1 << best_v
            ext_blocked |= _neighbors_mask(cur)
    
    return extended, ext_used, ext_blocked

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    best_path = [0]
    best_len = 1
    sys.setrecursionlimit(10000)
    
    def count_exits(v, used_mask, blocked_mask, cur):
        """
        Calculate the number of valid exits from v, assuming we move from cur to v.
        """
        new_used = used_mask | (1 << v)
        new_blocked = blocked_mask | _neighbors_mask(cur)
        valid_exits = _neighbors_mask(v) & ~new_used & ~new_blocked
        return valid_exits.bit_count()

    def dfs(path, used_mask, blocked_mask, depth):
        nonlocal best_path, best_len, deadline
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
        cur = path[-1]
        
        # Generate candidates
        candidates = []
        # Valid next vertices are neighbors of cur that are not used and not blocked
        valid_mask = _neighbors_mask(cur) & ~used_mask & ~blocked_mask
        
        # Iterate through set bits in valid_mask
        vm = valid_mask
        while vm:
            # Get least significant set bit
            lsb = vm & -vm
            v = lsb.bit_length() - 1
            vm ^= lsb
            
            # Calculate heuristic: number of exits from v
            score = count_exits(v, used_mask, blocked_mask, cur)
            candidates.append((score, v))
            
        if not candidates:
            return
            
        # Sort by score (Warnsdorff: fewest exits first)
        candidates.sort(key=lambda x: x[0])
        
        # Add random noise to break ties and allow diversity
        noise_scale = 0.05 if depth < 50 else 0.15
        for i in range(len(candidates)):
            candidates[i] = (candidates[i][0] + random.random() * noise_scale, candidates[i][1])
        candidates.sort(key=lambda x: x[0])
        
        for score, v in candidates:
            if time.time() > deadline:
                return
            
            # Move to v
            path.append(v)
            new_used_mask = used_mask | (1 << v)
            new_blocked_mask = blocked_mask | _neighbors_mask(cur)
            
            dfs(path, new_used_mask, new_blocked_mask, depth + 1)
            
            path.pop()

    # Main loop with multiple starting vertices
    start_vertices = list(range(1 << N))
    random.shuffle(start_vertices)
    
    restart_count = 0
    while time.time() < deadline:
        # Pick a starting vertex
        if restart_count < len(start_vertices):
            start_v = start_vertices[restart_count]
        else:
            start_v = random.randint(0, (1 << N) - 1)
        
        restart_count += 1
        
        # Run DFS from this starting vertex
        dfs([start_v], 1 << start_v, 0, 0)
        
        # Try to extend the best path found so far from both ends
        if best_len > 1:
            # Reconstruct used_mask and blocked_mask for best_path
            ext_used = 0
            for v in best_path:
                ext_used |= 1 << v
            
            # blocked_mask: neighbors of all internal vertices
            ext_blocked = 0
            for i in range(1, len(best_path) - 1):
                ext_blocked |= _neighbors_mask(best_path[i])
            
            # Try extending forward
            extended_fwd, _, _ = _try_extend(best_path, ext_used, ext_blocked, deadline, direction=1)
            if len(extended_fwd) > best_len:
                best_path = extended_fwd
                best_len = len(best_path)
            
            # Try extending backward
            extended_bwd, _, _ = _try_extend(best_path, ext_used, ext_blocked, deadline, direction=-1)
            if len(extended_bwd) > best_len:
                best_path = extended_bwd
                best_len = len(best_path)
        
        # Check time and break if we're close to deadline
        if time.time() > deadline - 5.0:
            break

    return best_path
