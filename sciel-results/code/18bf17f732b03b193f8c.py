import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses using Fermat's little theorem
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # For each y, the pair (y, inv[y]) forms a 2-cycle (or 1-cycle if y == inv[y]).
    # An interval [L, R] is valid iff for all y in [L, R], inv[y] is NOT in [L, R].
    # 
    # Key insight: For a fixed L, the maximum valid R is determined by the
    # minimum "high endpoint" among all y >= L where low[y] >= L.
    # low[y] = min(y, inv[y]), high[y] = max(y, inv[y])
    #
    # We sweep L from N down to 1, maintaining a heap of high[y] values
    # for all y with low[y] >= L. The minimum of this heap gives R_conflict(L),
    # so max valid R = R_conflict(L) - 1.
    
    low = [0] * (N + 1)
    high = [0] * (N + 1)
    for y in range(1, N + 1):
        iy = inv[y]
        low[y] = min(y, iy)
        high[y] = max(y, iy)
    
    # Bucket by low[y] value for efficient sweep
    bucket = [[] for _ in range(N + 2)]
    for y in range(1, N + 1):
        bucket[low[y]].append(y)
    
    # Sweep L from N down to 1
    heap = []
    best_start = 2
    best_length = 0
    best_end = 2
    
    for L in range(N, 0, -1):
        if time.time() > deadline:
            break
        
        # Add y's that become active at this L (low[y] == L)
        for y in bucket[L]:
            heapq.heappush(heap, high[y])
        
        if heap:
            r_conflict = heap[0]
            max_R = min(r_conflict - 1, N)
        else:
            max_R = N
        
        if max_R >= L:
            length = max_R - L + 1
            if length > best_length:
                best_length = length
                best_start = L
                best_end = max_R
    
    if best_length == 0:
        # Fallback: return a trivial valid interval
        # Self-inverse residues mod 1009 are 1 and 1008
        # So any x not in {1, 1008} is valid as a singleton
        best_start = 2
        best_length = 1
        best_end = 2
    
    # Verification pass: confirm the found interval is actually valid
    # This is a final polish to ensure correctness
    if time.time() < deadline - 5.0:
        valid = True
        for x in range(best_start, best_start + best_length):
            if x * x % MOD == 1:
                valid = False
                break
            for y in range(x + 1, best_start + best_length):
                if x * y % MOD == 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # This shouldn't happen, but if it does, fall back to a safe answer
            # Find the longest valid interval by brute force (small enough)
            best_start = 2
            best_length = 0
            for L in range(2, N):
                if time.time() > deadline:
                    break
                R = L
                while R < N:
                    y = R + 1
                    if L <= inv[y] <= y:
                        break
                    R = y
                length = R - L + 1
                if length > best_length:
                    best_length = length
                    best_start = L
    
    return [best_start, best_length]
