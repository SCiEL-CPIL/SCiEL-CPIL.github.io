import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    # aps[x] contains a list of tuples (a, b, c) such that {x, a, b, c} is a 4-AP
    aps = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps[a].append((b, c, e))
            aps[b].append((a, c, e))
            aps[c].append((a, b, e))
            aps[e].append((a, b, c))

    best_set = set()
    best_size = 0
    
    # Helper: check if adding x to S (with conflict counts) is valid
    def can_add_fast(x, S, conflict):
        # conflict[x] == 0 means no 4-AP is formed by adding x
        return conflict[x] == 0

    # Helper: update conflict counts when adding x to S
    def add_to_set(x, S, conflict):
        S.add(x)
        for a, b, c in aps[x]:
            if a in S and b in S:
                conflict[c] += 1
            if a in S and c in S:
                conflict[b] += 1
            if b in S and c in S:
                conflict[a] += 1

    # Helper: update conflict counts when removing x from S
    def remove_from_set(x, S, conflict):
        S.discard(x)
        for a, b, c in aps[x]:
            if a in S and b in S:
                conflict[c] -= 1
            if a in S and c in S:
                conflict[b] -= 1
            if b in S and c in S:
                conflict[a] -= 1

    # Helper: greedy initialization
    def greedy_init(order, seed=None):
        rng = random.Random(seed)
        S = set()
        conflict = [0] * (n + 1)
        candidates = list(order)
        rng.shuffle(candidates)
        
        for x in candidates:
            if conflict[x] == 0:
                add_to_set(x, S, conflict)
        return S, conflict

    # Helper: local search
    def local_search(S, conflict, max_steps=100000, rng=None):
        if rng is None:
            rng = random.Random()
        
        current_size = len(S)
        steps_since_improvement = 0
        
        while steps_since_improvement < max_steps and time.time() < deadline:
            # Find all valid elements to add
            valids = []
            for x in range(1, n + 1):
                if x not in S and conflict[x] == 0:
                    valids.append(x)
            
            if valids:
                # Add a random valid element
                x = rng.choice(valids)
                add_to_set(x, S, conflict)
                
                if len(S) > current_size:
                    current_size = len(S)
                    steps_since_improvement = 0
            else:
                # Stuck: no element can be added
                if S:
                    # Try swaps: remove x, add y
                    removable = list(S)
                    rng.shuffle(removable)
                    found_swap = False
                    
                    for x_remove in removable[:30]:  # Limit swap search
                        if time.time() > deadline:
                            break
                        
                        # Temporarily remove x_remove
                        remove_from_set(x_remove, S, conflict)
                        
                        # Find valid elements to add
                        valids_after_remove = []
                        for x_add in range(1, n + 1):
                            if x_add not in S and conflict[x_add] == 0:
                                valids_after_remove.append(x_add)
                        
                        if valids_after_remove:
                            # Add a random valid element
                            x_add = rng.choice(valids_after_remove)
                            add_to_set(x_add, S, conflict)
                            found_swap = True
                            
                            if len(S) > current_size:
                                current_size = len(S)
                                steps_since_improvement = 0
                        else:
                            # Restore
                            add_to_set(x_remove, S, conflict)
                        
                        if found_swap:
                            break
                    
                    if not found_swap:
                        # No swap found, just remove a random element to perturb
                        x_remove = rng.choice(removable)
                        remove_from_set(x_remove, S, conflict)
                    
                    steps_since_improvement += 1
                else:
                    break
        
        return S, conflict

    # Strategy 1: Base-3 digit restriction (Salem-Spencer style)
    # Select integers in [1, 512] that have no digit '2' in their ternary representation
    S3 = set()
    for i in range(1, n + 1):
        valid = True
        temp = i
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            S3.add(i)
    
    # Try to extend S3
    conflict3 = [0] * (n + 1)
    S3_copy = set()
    for x in S3:
        add_to_set(x, S3_copy, conflict3)
    
    # Greedily extend S3
    for x in range(1, n + 1):
        if x not in S3_copy and conflict3[x] == 0:
            add_to_set(x, S3_copy, conflict3)
    
    if len(S3_copy) > best_size:
        best_size = len(S3_copy)
        best_set = S3_copy.copy()

    # Strategy 2: Multiple greedy initializations with different orders
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)),
        list(range(2, n + 1, 2)),
    ]
    
    for i, order in enumerate(orders):
        if time.time() > deadline - 10:
            break
        S, conflict = greedy_init(order, seed=42 + i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Strategy 3: Local search from multiple starting points
    rng = random.Random(123)
    
    # Start from best set found so far
    S_ls = set(best_set)
    conflict_ls = [0] * (n + 1)
    for x in S_ls:
        add_to_set(x, S_ls, conflict_ls)
    
    S_ls, conflict_ls = local_search(S_ls, conflict_ls, max_steps=50000, rng=rng)
    
    if len(S_ls) > best_size:
        best_size = len(S_ls)
        best_set = S_ls.copy()

    # Strategy 4: More aggressive local search with perturbation
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        current_set = set(best_set)
        
        # Perturb: remove 1 to 3 random elements
        num_remove = rng.randint(1, 3)
        to_remove = rng.sample(list(current_set), min(num_remove, len(current_set)))
        for x in to_remove:
            current_set.discard(x)
        
        # Rebuild conflict
        conflict_pert = [0] * (n + 1)
        for x in current_set:
            add_to_set(x, current_set, conflict_pert)
        
        # Refill greedily with random order
        refill_order = [y for y in range(1, n + 1) if y not in current_set]
        rng.shuffle(refill_order)
        
        for x in refill_order:
            if conflict_pert[x] == 0:
                add_to_set(x, current_set, conflict_pert)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    # Final Polish: Try to add any remaining elements
    conflict_final = [0] * (n + 1)
    S_final = set()
    for x in best_set:
        add_to_set(x, S_final, conflict_final)
    
    for x in range(1, n + 1):
        if x not in S_final and conflict_final[x] == 0:
            add_to_set(x, S_final, conflict_final)
    
    if len(S_final) > len(best_set):
        best_set = S_final

    return sorted(best_set)
