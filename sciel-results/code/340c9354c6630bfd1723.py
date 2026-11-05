import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 512
    # Precompute all 4-APs
    # AP: a, a+d, a+2d, a+3d
    # a >= 1, a+3d <= N
    # d >= 1
    aps = []
    element_to_aps = [[] for _ in range(N + 1)]
    
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for x in ap:
                element_to_aps[x].append(idx)
                
    total_aps = len(aps)
    
    best_set = set()
    best_size = 0
    
    # Run multiple iterations with different random seeds for tie-breaking
    # Each iteration is fast, so we can run many.
    seed_counter = 0
    
    while time.time() < deadline:
        seed_counter += 1
        random.seed(seed_counter)
        
        # State
        S = set(range(1, N + 1))
        # active_aps: set of indices of APs where all 4 elements are in S
        active_aps = set(range(total_aps))
        # deg[x]: number of active APs containing x
        deg = [0] * (N + 1)
        
        for idx in range(total_aps):
            for x in aps[idx]:
                deg[x] += 1
                
        # Greedy Removal
        while active_aps:
            # Find element in S with max deg
            # To allow randomness in ties, we can collect candidates
            # Or just pick max, and if tie, random.
            # Efficient way: iterate S, track max.
            max_deg = -1
            candidates = []
            for x in S:
                if deg[x] > max_deg:
                    max_deg = deg[x]
                    candidates = [x]
                elif deg[x] == max_deg:
                    candidates.append(x)
            
            if not candidates:
                break
                
            # Pick random from candidates
            x_to_remove = random.choice(candidates)
            
            # Remove x_to_remove from S
            S.remove(x_to_remove)
            
            # Update active_aps and deg
            for idx in element_to_aps[x_to_remove]:
                if idx in active_aps:
                    active_aps.remove(idx)
                    for y in aps[idx]:
                        deg[y] -= 1
                        
        # Add-Back Phase
        removed_elements = list(set(range(1, N + 1)) - S)
        random.shuffle(removed_elements)
        
        for x in removed_elements:
            # Check if adding x creates any active AP
            can_add = True
            for idx in element_to_aps[x]:
                # Check if other 3 elements are in S
                ap = aps[idx]
                # The other 3 are ap minus x
                others = [y for y in ap if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                # Update active_aps and deg
                for idx in element_to_aps[x]:
                    ap = aps[idx]
                    others = [y for y in ap if y != x]
                    if all(y in S for y in others):
                        active_aps.add(idx)
                        for y in ap:
                            deg[y] += 1
                            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)
