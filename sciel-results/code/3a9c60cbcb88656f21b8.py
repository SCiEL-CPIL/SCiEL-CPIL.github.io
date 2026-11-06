import time
import random
import sys

N = 9
FULL_MASK = (1 << 512) - 1

def _neighbors_mask(v):
    """Return a 512-bit mask with bits set for all N neighbors of v."""
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    best_path = [0]
    best_len = 1
    sys.setrecursionlimit(10000)
    
    def count_exits(v, used_mask, blocked_mask, cur):
        """
        Calculate the number of valid exits from v, assuming we move from cur to v.
        new_used = used_mask | (1 << v)
        new_blocked = blocked_mask | _neighbors_mask(cur)
        Exits are neighbors of v that are not in new_used and not in new_blocked.
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
            # We pass cur because when we move to v, cur becomes blocked
            score = count_exits(v, used_mask, blocked_mask, cur)
            candidates.append((score, v))
            
        if not candidates:
            return
            
        # Sort by score (Warnsdorff: fewest exits first)
        candidates.sort(key=lambda x: x[0])
        
        # Add random noise to break ties and allow diversity
        for i in range(len(candidates)):
            candidates[i] = (candidates[i][0] + random.random() * 0.1, candidates[i][1])
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

    # Main loop
    while time.time() < deadline:
        # Reset for each restart
        dfs([0], 1, 0, 0)

    return best_path
