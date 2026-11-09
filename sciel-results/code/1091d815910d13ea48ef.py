import time
import sys

def _precompute_divisibility():
    """
    Precompute divisibility masks.
    P = lcm(1..12) = 27720.
    div_mask[r][j] is a 12-bit integer where bit k is set if (k+1) divides (r + 1 + j).
    j ranges from 0 to 143 (max L is 144).
    """
    P = 27720
    MAX_L = 144
    div_mask = [[0] * MAX_L for _ in range(P)]
    
    for r in range(P):
        base = r + 1
        for j in range(MAX_L):
            val = base + j
            mask = 0
            mask |= 1
            if val % 2 == 0: mask |= (1 << 1)
            if val % 3 == 0: mask |= (1 << 2)
            if val % 4 == 0: mask |= (1 << 3)
            if val % 5 == 0: mask |= (1 << 4)
            if val % 6 == 0: mask |= (1 << 5)
            if val % 7 == 0: mask |= (1 << 6)
            if val % 8 == 0: mask |= (1 << 7)
            if val % 9 == 0: mask |= (1 << 8)
            if val % 10 == 0: mask |= (1 << 9)
            if val % 11 == 0: mask |= (1 << 10)
            if val % 12 == 0: mask |= (1 << 11)
            div_mask[r][j] = mask
    return div_mask

def _has_matching(mask_row, L):
    """
    Check if there is a matching of size 12 in the window of length L.
    Uses bitmask-based adjacency for faster iteration.
    """
    if L < 12:
        return False
    
    # Build adjacency: for each left node i (0..11), a bitmask of positions j (0..L-1)
    # where i+1 divides position j+1
    adj_mask = [0] * 12
    for j in range(L):
        mask = mask_row[j]
        for i in range(12):
            if mask & (1 << i):
                adj_mask[i] |= (1 << j)
    
    # Check: each left node must have at least one option
    for i in range(12):
        if adj_mask[i] == 0:
            return False
    
    # match_pos[j] = left node matched to position j, or -1
    match_pos = [-1] * L
    
    def dfs(i, visited):
        avail = adj_mask[i]
        while avail:
            bit = avail & (-avail)
            avail ^= bit
            j = bit.bit_length() - 1
            if not visited[j]:
                visited[j] = True
                if match_pos[j] == -1 or dfs(match_pos[j], visited):
                    match_pos[j] = i
                    return True
        return False
    
    # Process left nodes in order of fewest options first for better pruning
    order = sorted(range(12), key=lambda i: bin(adj_mask[i]).count('1'))
    
    for i in order:
        visited = [False] * L
        if not dfs(i, visited):
            return False
    
    return True

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    div_mask = _precompute_divisibility()
    
    best_m = 0
    best_f = 12
    
    # Precompute for each r, the minimal L using a smarter approach
    # First, check if L=12 works (fast path)
    for r in range(27720):
        if time.time() - start_time > DEADLINE:
            break
        
        # Quick check: does L=12 work?
        if _has_matching(div_mask[r], 12):
            continue  # f=12, not better than current best if best_f >= 12
        
        # Binary search for minimal L > 12
        lo, hi = 13, 144
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if _has_matching(div_mask[r], mid):
                best_f_candidate = mid
                found = True
                hi = mid - 1
            else:
                lo = mid + 1
                
        if found:
            if best_f_candidate > best_f:
                best_f = best_f_candidate
                best_m = r
                
    return int(best_m)

if __name__ == "__main__":
    m = construct()
    print(f"Best m: {m}, f: {best_f}")
