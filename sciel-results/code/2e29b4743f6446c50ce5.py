import random
import time
import math

N = 512

# Precompute all 4-term arithmetic progressions
# APs[i] is a tuple (a, a+d, a+2d, a+3d)
# aps_containing[x] is a list of indices of APs that contain x
APs = []
aps_containing = [[] for _ in range(N + 1)]

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        idx = len(APs)
        APs.append(ap)
        for x in ap:
            aps_containing[x].append(idx)

def greedy_init():
    """Create an initial valid set using randomized greedy."""
    S = [0] * (N + 1)
    ap_count = [0] * len(APs)
    
    # We need to check validity quickly.
    # For greedy, we can just use the can_add logic from seed but optimized.
    # Or simpler: just iterate and check.
    # Given N=512, a simple check is fine for init.
    
    S_set = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    # To speed up greedy, we can precompute conflicts or just check.
    # Let's use the ap_count structure for efficiency even in greedy.
    # Actually, for greedy, we just add if valid.
    # Checking validity of adding x:
    # Count how many other elements of APs containing x are already in S.
    # If any AP has 3 others, we cannot add x.
    
    for x in candidates:
        valid = True
        for idx in aps_containing[x]:
            ap = APs[idx]
            count = 0
            for y in ap:
                if y != x and S[y]:
                    count += 1
            if count == 3:
                valid = False
                break
        if valid:
            S[x] = 1
            # Update ap_count for efficiency in later stages? 
            # For greedy, we don't strictly need ap_count array if we check on the fly,
            # but let's maintain it for consistency with SA.
            for idx in aps_containing[x]:
                ap_count[idx] += 1
                
    return S, ap_count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Initial state
    S, ap_count = greedy_init()
    
    # Calculate initial stats
    size = sum(S)
    num_aps = sum(1 for c in ap_count if c == 4)
    
    best_size = size
    best_S = S[:]
    
    # Simulated Annealing Parameters
    # Cost = num_aps * PENALTY - size
    # We want to minimize Cost.
    # If num_aps > 0, cost is large positive.
    # If num_aps == 0, cost is -size (negative).
    # We want to reach num_aps=0 and maximize size.
    
    PENALTY = 1000.0
    T = 10.0  # Initial temperature
    alpha = 0.9995  # Cooling rate
    T_min = 0.01
    
    # We will run SA until time runs out.
    # To allow multiple restarts if we get stuck in a bad local minimum (high num_aps),
    # we can restart if num_aps doesn't decrease for a long time.
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with a new greedy init if we have time
            if time.time() < deadline - 5.0:
                S, ap_count = greedy_init()
                size = sum(S)
                num_aps = sum(1 for c in ap_count if c == 4)
                if size > best_size and num_aps == 0:
                    best_size = size
                    best_S = S[:]
                T = 10.0
            else:
                break
        
        # Perform a move
        x = random.randint(1, N)
        
        # Calculate delta
        # If x is in S, we are removing it.
        # If x is not in S, we are adding it.
        
        delta_size = -1 if S[x] else 1
        
        delta_aps = 0
        if S[x]:
            # Removing x
            for idx in aps_containing[x]:
                if ap_count[idx] == 4:
                    # This AP was bad, removing x makes it 3 (good)
                    delta_aps -= 1
                # Note: removing x from an AP with count < 4 doesn't change bad status
                # (it goes from k to k-1, neither is 4)
                # Removing x from an AP with count > 4 is impossible (max 4)
                ap_count[idx] -= 1
        else:
            # Adding x
            for idx in aps_containing[x]:
                if ap_count[idx] == 3:
                    # This AP was good (3 elements), adding x makes it 4 (bad)
                    delta_aps += 1
                # Adding x to AP with count < 3: count becomes < 4, still good
                # Adding x to AP with count == 4: impossible (x not in S, so x not in AP count)
                # Wait, if ap_count[idx] == 4, that means all 4 elements are in S.
                # If x is not in S, x is not one of those 4. So x is not in this AP?
                # No, x IS in this AP (by definition of aps_containing).
                # If x is not in S, it cannot be part of the 4 elements in S.
                # So ap_count[idx] can never be 4 if x is not in S?
                # Yes, because ap_count counts elements IN S.
                # So if x is not in S, max count is 3.
                ap_count[idx] += 1
        
        # Calculate Cost Delta
        # Cost = num_aps * PENALTY - size
        # New Cost = (num_aps + delta_aps) * PENALTY - (size + delta_size)
        # Delta Cost = delta_aps * PENALTY - delta_size
        
        delta_cost = delta_aps * PENALTY - delta_size
        
        # Acceptance Criterion
        accepted = False
        if delta_cost <= 0:
            accepted = True
        else:
            # Prob = exp(-delta_cost / T)
            # To avoid underflow/overflow, use log
            if T > 0:
                # delta_cost is positive
                # If delta_cost/T is very large, exp(-large) is 0
                # We can just compute exp if T is reasonable
                try:
                    prob = math.exp(-delta_cost / T)
                except OverflowError:
                    prob = 0.0
                if random.random() < prob:
                    accepted = True
        
        if accepted:
            # Apply move
            if S[x]:
                S[x] = 0
            else:
                S[x] = 1
            
            size += delta_size
            num_aps += delta_aps
            
            # Check for improvement
            if num_aps == 0 and size > best_size:
                best_size = size
                best_S = S[:]
        
        # Cool down
        T *= alpha
        
    return sorted([i for i in range(1, N + 1) if best_S[i]])

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S_set = set(res)
    valid = True
    for ap in APs:
        if all(x in S_set for x in ap):
            valid = False
            break
    print(f"Valid: {valid}")
