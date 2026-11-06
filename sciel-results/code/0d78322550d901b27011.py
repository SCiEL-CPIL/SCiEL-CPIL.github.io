import time
import random
import numpy as np

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]

def precompute():
    dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = POINTS[i]
        for j in range(NUM_POINTS):
            xj, yj = POINTS[j]
            dist2[i][j] = (xi - xj) ** 2 + (yi - yj) ** 2

    # block_mask[i][j] is a bitmask of points x such that {i, j, x} is isosceles
    # x != i and x != j
    block_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = dist2[i][j]
            for x in range(NUM_POINTS):
                if x == i or x == j:
                    continue
                d_ix = dist2[i][x]
                d_jx = dist2[j][x]
                if d_ix == d_jx or d_ix == d_ij or d_jx == d_ij:
                    mask |= (1 << x)
            block_mask[i][j] = mask
            block_mask[j][i] = mask
    return dist2, block_mask

DIST2, BLOCK_MASK = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit (evaluator kills at 330s)
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check if a set (as a bitmask) is a valid no-isosceles set
    # S_mask is an integer bitmask
    def is_valid(S_mask):
        # For all pairs (a, b) in S, check if (BLOCK_MASK[a][b] & S_mask) == 0
        # Note: BLOCK_MASK[a][b] does not contain a or b.
        # So if there is any x in S (x != a, b) such that x is in BLOCK_MASK[a][b], it's invalid.
        # This is equivalent to (BLOCK_MASK[a][b] & S_mask) != 0.
        
        # Iterate through set bits in S_mask
        s_list = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_list.append(idx)
            m ^= lsb
        
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if BLOCK_MASK[a][b] & S_mask:
                    return False
        return True

    # Helper to check if a set is maximal
    # S_mask is an integer bitmask
    def is_maximal(S_mask):
        # For every x not in S, there must be a pair (a, b) in S such that x is in BLOCK_MASK[a][b]
        s_list = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_list.append(idx)
            m ^= lsb
        
        for x in range(NUM_POINTS):
            if (S_mask >> x) & 1:
                continue
            blocked = False
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    if (BLOCK_MASK[a][b] >> x) & 1:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return False
        return True

    # Heuristic: Greedy Construction with Coverage + Reduction
    def run_heuristic(seed_val):
        random.seed(seed_val)
        
        # Step 1: Greedy construction of a maximal independent set
        # We maintain counts[x] = number of pairs in S that block x.
        # If counts[x] > 0, x cannot be added.
        # We want to add points that are "useful" for blocking others.
        
        S_mask = 0
        S_list = []
        counts = [0] * NUM_POINTS
        
        # Order of insertion:
        # Random order, but we can bias towards points that block many others?
        # Actually, for independent set, we just need to avoid conflicts.
        # Let's try random order first.
        
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        for p_idx in indices:
            if counts[p_idx] == 0:
                # Add p_idx
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)
                # Update counts
                for q_idx in S_list:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] += 1
                        m ^= lsb
        
        # Step 2: Reduction
        # Try to remove points while maintaining maximality.
        improved = True
        while improved:
            improved = False
            # Shuffle S_list to try different removal orders
            s_list_copy = S_list[:]
            random.shuffle(s_list_copy)
            
            for p_idx in s_list_copy:
                if not (S_mask >> p_idx) & 1:
                    continue
                
                # Simulate removal
                # Decrement counts for all x in BLOCK_MASK[p_idx][q] for q in S \ {p_idx}
                changes = []
                for q_idx in S_list:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] -= 1
                        changes.append(x_idx)
                        m ^= lsb
                
                # Check if p_idx is still blocked
                if counts[p_idx] == 0:
                    # Revert
                    for x_idx in changes:
                        counts[x_idx] += 1
                    continue
                
                # Check if all x not in S \ {p_idx} are blocked
                # S \ {p_idx} is S_mask with p_idx bit cleared
                new_S_mask = S_mask & ~(1 << p_idx)
                valid = True
                for x in range(NUM_POINTS):
                    if (new_S_mask >> x) & 1:
                        continue
                    if counts[x] == 0:
                        valid = False
                        break
                
                if not valid:
                    for x_idx in changes:
                        counts[x_idx] += 1
                    continue
                
                # Removal is valid
                S_mask = new_S_mask
                S_list.remove(p_idx)
                improved = True
                break # Restart loop
        
        return len(S_list), S_mask

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        size, s_mask = run_heuristic(restarts)
        if size < best_size:
            best_size = size
            best_set = s_mask
        restarts += 1
        
        # Optional: Local Search on the best set found so far
        # If we have time, try to improve the best set further.
        if best_set is not None and restarts % 10 == 0:
            # Try to remove more points from best_set?
            # The reduction phase should have done this.
            # Maybe try adding and removing?
            pass

    if best_set is None:
        # Fallback: Simple greedy
        S_mask = 0
        S_list = []
        counts = [0] * NUM_POINTS
        for p_idx in range(NUM_POINTS):
            if counts[p_idx] == 0:
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)
                for q_idx in S_list:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] += 1
                        m ^= lsb
        best_set = S_mask

    # Convert bitmask to list of points
    result = []
    m = best_set
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(POINTS[idx])
        m ^= lsb
    
    return result

if __name__ == "__main__":
    s = construct()
    print(len(s))
