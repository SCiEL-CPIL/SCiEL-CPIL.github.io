"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
# EVOLVE-BLOCK-START
import random
import time

N = 512

def _precompute_conflicts():
    """
    Precompute for each x in [1, N] a list of bitmasks.
    Each bitmask represents a set of 3 other elements {a, b, c} such that 
    {a, b, c, x} forms a 4-term arithmetic progression.
    """
    conflicts = [[] for _ in range(N + 1)]
    
    # Generate all 4-term APs in [1, N]
    # AP: p, p+d, p+2d, p+3d
    for p in range(1, N + 1):
        max_d = (N - p) // 3
        for d in range(1, max_d + 1):
            t1 = p
            t2 = p + d
            t3 = p + 2 * d
            t4 = p + 3 * d
            
            # For each element in this AP, the other 3 form a "forbidden triplet"
            # If we add t1, we must not have {t2, t3, t4} all in S.
            # Mask for {t2, t3, t4}
            mask = (1 << t2) | (1 << t3) | (1 << t4)
            conflicts[t1].append(mask)
            
            mask = (1 << t1) | (1 << t3) | (1 << t4)
            conflicts[t2].append(mask)
            
            mask = (1 << t1) | (1 << t2) | (1 << t4)
            conflicts[t3].append(mask)
            
            mask = (1 << t1) | (1 << t2) | (1 << t3)
            conflicts[t4].append(mask)
            
    return conflicts

def construct():
    random.seed(42)
    conflicts = _precompute_conflicts()
    
    # S is a bitmask of the current set
    S = 0
    # List of elements in S for fast random access
    S_list = []
    
    def can_add(x, S_mask):
        """Check if x can be added to S_mask without forming a 4-AP."""
        for mask in conflicts[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list_copy):
        """Try to add any possible elements to S_mask in a random order."""
        remaining = [i for i in range(1, N + 1) if not (S_mask & (1 << i))]
        random.shuffle(remaining)
        for x in remaining:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
                S_list_copy.append(x)
        return S_mask, S_list_copy

    # 1. Greedy Initialization
    # Try multiple random orders to find a good starting point
    best_S = 0
    best_size = 0
    best_list = []
    
    for _ in range(20):
        current_S = 0
        current_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x, current_S):
                current_S |= (1 << x)
                current_list.append(x)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_S
            best_list = current_list[:]
            
    S = best_S
    S_list = best_list
    current_size = best_size

    # 2. Simulated Annealing
    # We use a time-based loop to maximize iterations within the limit.
    time_limit = 55.0 # Seconds
    start_time = time.time()
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, N + 1))
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Move Type 1: Add a random element (if possible)
        # This is always an improvement if possible.
        if random.random() < 0.5:
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add(x, S):
                    S |= (1 << x)
                    S_list.append(x)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S
                        best_list = S_list[:]
        
        # Move Type 2: Swap (Remove x, Add y)
        # This keeps size constant but changes the set.
        # We accept valid swaps to explore the state space.
        else:
            if S_list:
                x = random.choice(S_list)
                y = random.choice(all_elements)
                if x != y and not (S & (1 << y)):
                    # Check if swap is valid
                    S_without_x = S & ~(1 << x)
                    if can_add(y, S_without_x):
                        # Perform swap
                        S = S_without_x | (1 << y)
                        # Update list: replace x with y
                        idx = S_list.index(x)
                        S_list[idx] = y
                        
        # Periodic Greedy Fill
        # After some iterations, try to fill any gaps created by swaps or just random luck
        if iterations % 1000 == 0:
            S, S_list = greedy_fill(S, S_list)
            current_size = len(S_list)
            if current_size > best_size:
                best_size = current_size
                best_S = S
                best_list = S_list[:]

    return sorted(best_list)

# EVOLVE-BLOCK-END
