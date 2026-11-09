import numpy as np

def construct():
    N = 256
    
    # Precompute 4-AP conflicts
    # conflict_mask[x] is a bitset where bit y is set if x and y are in a 4-AP
    # We use 0-indexed bits: element i (1-indexed) corresponds to bit (i-1)
    conflict_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            for pos in [1, 2, 3]:
                y = x + pos * d
                if y <= N:
                    mask |= (1 << (y - 1))
        # x is 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x, (N - x) // 2) + 1):
            for pos in [-1, 1, 2]:
                y = x + pos * d
                if 1 <= y <= N:
                    mask |= (1 << (y - 1))
        # x is 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            for pos in [-2, -1, 1]:
                y = x + pos * d
                if 1 <= y <= N:
                    mask |= (1 << (y - 1))
        # x is 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            for pos in [-3, -2, -1]:
                y = x + pos * d
                if 1 <= y <= N:
                    mask |= (1 << (y - 1))
        conflict_mask[x] = mask

    # Precompute 3-AP contributions
    # contrib[x] is a list of masks for the other two elements in 3-APs containing x
    contrib = [[] for _ in range(N + 1)]
    
    # Generate all 3-APs
    for a in range(1, N - 1):
        for b in range(a + 1, N):
            c = 2 * b - a
            if c > N:
                break
            # 3-AP (a, b, c)
            # For a: need b and c
            contrib[a].append((1 << (b - 1)) | (1 << (c - 1)))
            # For b: need a and c
            contrib[b].append((1 << (a - 1)) | (1 << (c - 1)))
            # For c: need a and b
            contrib[c].append((1 << (a - 1)) | (1 << (b - 1)))

    # Greedy Selection
    S_mask = 0
    available = (1 << N) - 1  # All elements available initially
    
    selected = []
    
    while available:
        best_x = -1
        best_gain = -1
        
        # Iterate through available elements
        for x in range(1, N + 1):
            if not (available & (1 << (x - 1))):
                continue
            
            # Calculate gain
            gain = 0
            for mask in contrib[x]:
                if (S_mask & mask) == mask:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
            elif gain == best_gain:
                # Tie-breaker: prefer smaller x
                if best_x == -1 or x < best_x:
                    best_x = x
        
        if best_x == -1:
            break
        
        selected.append(best_x)
        S_mask |= (1 << (best_x - 1))
        
        # Remove best_x and its conflicts from available
        available &= ~(1 << (best_x - 1))
        available &= ~conflict_mask[best_x]

    return sorted(selected)
