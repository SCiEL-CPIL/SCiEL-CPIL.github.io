"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
# EVOLVE-BLOCK-START
import random
import time
import math

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
    # Dictionary to map element to its index in S_list for O(1) removal
    S_pos = {}
    
    def can_add(x, S_mask):
        """Check if x can be added to S_mask without forming a 4-AP."""
        for mask in conflicts[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list_copy, S_pos_copy):
        """Try to add any possible elements to S_mask in a random order."""
        remaining = [i for i in range(1, N + 1) if not (S_mask & (1 << i))]
        random.shuffle(remaining)
        for x in remaining:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
                S_list_copy.append(x)
                S_pos_copy[x] = len(S_list_copy) - 1
        return S_mask, S_list_copy, S_pos_copy

    # 1. Greedy Initialization
    # Try multiple random orders to find a good starting point
    best_S = 0
    best_size = 0
    best_list = []
    best_pos = {}
    
    for _ in range(100): # Increase initial attempts for better start
        current_S = 0
        current_list = []
        current_pos = {}
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x, current_S):
                current_S |= (1 << x)
                current_list.append(x)
                current_pos[x] = len(current_list) - 1
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_S
            best_list = current_list[:]
            best_pos = current_pos.copy()
            
    S = best_S
    S_list = best_list
    S_pos = best_pos
    current_size = best_size

    # 2. Simulated Annealing
    # We use a time-based loop to maximize iterations within the limit.
    time_limit = 55.0 # Seconds
    start_time = time.time()
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, N + 1))
    
    # SA Parameters
    T = 5.0  # Initial temperature
    alpha = 0.99995 # Cooling rate
    min_T = 0.001
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        
        r = random.random()
        
        if r < 0.4:
            # Try to add
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add(x, S):
                    # Always accept improvement
                    S |= (1 << x)
                    S_list.append(x)
                    S_pos[x] = len(S_list) - 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S
                        best_list = S_list[:]
                        best_pos = S_pos.copy()
        
        elif r < 0.7:
            # Try to remove
            if S_list:
                x = random.choice(S_list)
                # Calculate energy change: removing 1 element increases energy by 1 (worsens)
                # Delta E = 1
                if random.random() < math.exp(-1.0 / T):
                    # Accept
                    S &= ~(1 << x)
                    # Efficient removal from list
                    idx = S_pos[x]
                    last_elem = S_list[-1]
                    S_list[idx] = last_elem
                    S_pos[last_elem] = idx
                    S_list.pop()
                    del S_pos[x]
                    current_size -= 1
        
        else:
            # Try to swap
            if S_list:
                x = random.choice(S_list)
                y = random.choice(all_elements)
                if x != y and not (S & (1 << y)):
                    # Check if swap is valid
                    S_without_x = S & ~(1 << x)
                    if can_add(y, S_without_x):
                        # Always accept neutral move (size unchanged)
                        S = S_without_x | (1 << y)
                        # Update list
                        idx = S_pos[x]
                        S_list[idx] = y
                        S_pos[y] = idx
                        del S_pos[x]
                    # If invalid, do nothing

        # Cooling
        if T > min_T:
            T *= alpha

    return sorted(best_list)

# EVOLVE-BLOCK-END
