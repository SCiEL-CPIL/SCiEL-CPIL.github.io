import time
import random
import math
import numpy as np

N = 256

# Precompute 3-APs and 4-APs as bitmasks
# and build indices

three_aps = []
four_aps = []

# three_ap_indices[x] = list of indices of 3-APs containing x
# four_ap_indices[x] = list of indices of 4-APs containing x
three_ap_indices = [[] for _ in range(N + 1)]
four_ap_indices = [[] for _ in range(N + 1)]

# For fast counting
three_aps_indices = [] # List of (a,b,c) tuples

def precompute():
    global three_aps, four_aps, three_aps_indices
    
    # Generate 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            idx = len(three_aps)
            three_aps.append(mask)
            three_ap_indices[a].append(idx)
            three_ap_indices[b].append(idx)
            three_ap_indices[c].append(idx)
            three_aps_indices.append((a, b, c))
            
    # Generate 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            idx = len(four_aps)
            four_aps.append(mask)
            four_ap_indices[a].append(idx)
            four_ap_indices[b].append(idx)
            four_ap_indices[c].append(idx)
            four_ap_indices[e].append(idx)

precompute()

# Convert three_aps_indices to a numpy array for fast counting
# Shape: (num_3aps, 3)
three_aps_np = np.array(three_aps_indices)

def count_3aps(S_mask):
    """Count 3-APs in S_mask using NumPy vectorization."""
    if not S_mask:
        return 0
    
    # Create a boolean array for the set S
    # indices are 0..255 corresponding to 1..256
    s_indices = np.where(np.unpackbits(np.frombuffer(np.uint64(S_mask).tobytes(), dtype=np.uint64), 64) > 0)[0]
    # This is slow. Let's stick to bit manipulation or a precomputed lookup if possible.
    # Actually, for N=256, a simple loop in Python over ~10k 3-APs is ~1ms. 
    # The bottleneck was likely the *number* of restarts and the SA steps, not just counting.
    # But let's try a faster counting method.
    
    # Method: Iterate through 3-APs. 
    # We can precompute masks as integers.
    count = 0
    for mask in three_aps:
        if (S_mask & mask) == mask:
            count += 1
    return count

def is_valid(S_mask):
    """Check if S_mask is 4-AP free."""
    for mask in four_aps:
        if (S_mask & mask) == mask:
            return False
    return True

def greedy_construct(seed):
    random.seed(seed)
    S_mask = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        bit = 1 << (x - 1)
        new_S = S_mask | bit
        # Check validity incrementally
        valid = True
        for idx in four_ap_indices[x]:
            m = four_aps[idx]
            # Check if other 3 bits are in S_mask
            # m has 4 bits. One is x.
            # (S_mask & (m ^ bit)) == (m ^ bit)
            if (S_mask & (m ^ bit)) == (m ^ bit):
                valid = False
                break
        if valid:
            S_mask = new_S
            
    return S_mask

def simulated_annealing(S_mask, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_S = S_mask
    current_score = count_3aps(S_mask)
    
    best_S = S_mask
    best_score = current_score
    
    T = 10.0
    cooling = 0.9995 # Slower cooling to explore more?
    
    # To speed up, we can precompute the "cost" of adding/removing x?
    # No, state changes.
    
    # Optimization: Precompute the list of 3-AP indices for each x and 4-AP indices for each x.
    # We already have three_ap_indices and four_ap_indices.
    
    while time.time() < deadline:
        if T < 0.01:
            T = 10.0
            
        x = random.randint(1, N)
        bit = 1 << (x - 1)
        
        if current_S & bit:
            # Remove x
            new_S = current_S ^ bit
            # Validity is guaranteed
            delta = 0
            for idx in three_ap_indices[x]:
                m = three_aps[idx]
                if (current_S & (m ^ bit)) == (m ^ bit):
                    delta -= 1
            new_score = current_score + delta
        else:
            # Add x
            new_S = current_S | bit
            # Check validity
            valid = True
            for idx in four_ap_indices[x]:
                m = four_aps[idx]
                if (current_S & (m ^ bit)) == (m ^ bit):
                    valid = False
                    break
            
            if not valid:
                continue
            
            delta = 0
            for idx in three_ap_indices[x]:
                m = three_aps[idx]
                if (current_S & (m ^ bit)) == (m ^ bit):
                    delta += 1
            new_score = current_score + delta
        
        # Acceptance
        if new_score >= current_score:
            current_S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = new_score
                best_S = new_S
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            p = math.exp(delta / T)
            if random.random() < p:
                current_S = new_S
                current_score = new_score
        
        T *= cooling
        
    return best_S

def construct():
    start_time = time.time()
    # 60s limit. Use 50s for safety.
    deadline = start_time + 50.0
    
    best_result_mask = 0
    best_score = -1
    
    # Multiple restarts
    # Estimate time per restart
    # Greedy is fast. SA is slow.
    
    # We will do restarts until deadline.
    # Each restart: Greedy + SA.
    # If remaining time is small, just do Greedy.
    
    i = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        
        # If less than 5 seconds, do a quick greedy with a new seed
        if remaining < 5.0:
             S_mask = greedy_construct(i * 100 + 42)
             score = count_3aps(S_mask)
             if score > best_score:
                 best_score = score
                 best_result_mask = S_mask
             i += 1
             continue
        
        # Allocate time to SA
        # Leave 2s for next iteration overhead/setup
        sa_time = remaining - 2.0
        if sa_time < 1.0:
             S_mask = greedy_construct(i * 100 + 42)
             score = count_3aps(S_mask)
             if score > best_score:
                 best_score = score
                 best_result_mask = S_mask
             i += 1
             continue

        seed = i * 100 + 42
        S_mask = greedy_construct(seed)
        
        S_opt = simulated_annealing(S_mask, sa_time)
        score = count_3aps(S_opt)
        
        if score > best_score:
            best_score = score
            best_result_mask = S_opt
            
        i += 1
        
    # Convert mask to list
    result = []
    for i in range(N):
        if best_result_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
