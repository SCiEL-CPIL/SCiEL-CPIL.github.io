import time
import itertools

N = 16
TOTAL_POINTS = N * N

# Precompute isosceles masks
# IsoMask[i][j] is a bitmask of points k such that i, j, k form an isosceles triangle
# i, j, k must be distinct.

def get_coords(idx):
    return (idx // N, idx % N)

def dist2(i, j):
    xi, yi = get_coords(i)
    xj, yj = get_coords(j)
    return (xi - xj) ** 2 + (yi - yj) ** 2

# Precompute all pairwise distances squared
# D[i][j]
D = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        if i != j:
            D[i][j] = dist2(i, j)

# Precompute IsoMask
# IsoMask[i][j] : bitmask of k
IsoMask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        if i == j:
            continue
        mask = 0
        d_ij = D[i][j]
        for k in range(TOTAL_POINTS):
            if k == i or k == j:
                continue
            d_ik = D[i][k]
            d_jk = D[j][k]
            # Isosceles if any two distances are equal
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        IsoMask[i][j] = mask

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # U: bitmask of unblocked points
    # Initially all points are unblocked
    U = (1 << TOTAL_POINTS) - 1
    
    S = []
    
    while U != 0:
        if time.time() > deadline:
            break
            
        best_x = -1
        max_blocked = -1
        
        # Iterate through all candidates in U
        # Extract indices from bitmask
        u = U
        while u:
            # Get lowest set bit
            lb = u & -u
            x = lb.bit_length() - 1
            u -= lb
            
            # Calculate blocked points if we add x
            # blocked by x and any a in S
            blocked_mask = 0
            for a in S:
                blocked_mask |= IsoMask[x][a]
            
            # Only care about points in U
            new_blocked = blocked_mask & U
            
            # Count bits
            # x itself is in U, but adding x removes it.
            # The "gain" is the number of points in new_blocked.
            # Note: x is in U, but x is not in new_blocked because IsoMask[x][a] doesn't include x (k != i).
            # So count is correct.
            
            count = new_blocked.bit_count()
            
            if count > max_blocked:
                max_blocked = count
                best_x = x
            elif count == max_blocked and best_x != -1:
                # Tie-break: pick smallest index? Or random?
                # Smallest index is deterministic
                if x < best_x:
                    best_x = x
            
            # Check time in inner loop? Might be too slow.
            # The outer loop runs |S| times. |S| is small.
            # Inner loop runs |U| times. |U| <= 256.
            # 256 * |S| iterations. If |S| < 100, 25600 iterations.
            # Each iteration does |S| bitwise ORs.
            # Total ops ~ |S|^2 * 256? No.
            # For each x, we do |S| ORs.
            # Total ORs per step: |U| * |S|.
            # Total ORs total: sum(|U|_t * |S|_t).
            # This is small.
            
        if best_x == -1:
            # Should not happen if U != 0
            break
            
        # Add best_x to S
        S.append(best_x)
        
        # Update U
        # Remove best_x
        U &= ~(1 << best_x)
        
        # Remove points blocked by best_x and existing S
        # The new blocking effect is from pairs (best_x, a) for a in S_old
        # S_old is S without best_x
        # We just added best_x to S, so S[-1] is best_x.
        # We need to OR IsoMask[best_x][a] for all a in S[:-1]
        
        blocked_mask = 0
        for a in S[:-1]:
            blocked_mask |= IsoMask[best_x][a]
            
        U &= ~blocked_mask
        
    # Convert indices back to (x, y) tuples
    result = [get_coords(i) for i in S]
    return result
