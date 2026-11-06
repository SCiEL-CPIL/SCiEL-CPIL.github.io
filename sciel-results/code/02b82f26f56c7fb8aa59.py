import time
import random

def construct():
    start_time = time.time()
    time_limit = 45.0 # Leave 15s margin for evaluation overhead
    
    N = 256
    
    # Helper: Check if a set is 4-AP free
    def is_4ap_free(A):
        A_set = set(A)
        for a in A_set:
            for b in A_set:
                if b <= a:
                    continue
                d = b - a
                # Check if a, b, b+d, b+2d are in A
                if (b + d) in A_set and (b + 2*d) in A_set:
                    return False
        return True

    # Helper: Count 3-APs in A
    def count_3aps(A):
        A_set = set(A)
        count = 0
        for a in A_set:
            for b in A_set:
                if b <= a:
                    continue
                c = 2*b - a
                if c in A_set:
                    count += 1
        return count

    # Helper: Calculate gain in 3-APs when adding x to A
    def add_gain(x, A):
        A_set = set(A)
        gain = 0
        # x is first: (x, y, z) => z = 2y - x
        for y in A_set:
            z = 2*y - x
            if z in A_set and z > y:
                gain += 1
        # x is middle: (y, x, z) => y + z = 2x
        for y in A_set:
            z = 2*x - y
            if z in A_set and z > x:
                gain += 1
        # x is last: (y, z, x) => y = 2z - x
        for z in A_set:
            y = 2*z - x
            if y in A_set and y < z:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs when removing x from A
    def remove_loss(x, A):
        # Same as add_gain, because 3-AP structure is symmetric in count
        return add_gain(x, A)

    # Helper: Check if adding x to A keeps it 4-AP free
    def can_add(x, A):
        if x in A:
            return False
        A_set = set(A)
        # Check 4-APs involving x
        # x can be 1st, 2nd, 3rd, or 4th term
        # 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in A_set and (x + 2*d) in A_set and (x + 3*d) in A_set:
                return False
        # 2nd: x-d, x, x+d, x+2d
        for d in range(1, x):
            if (x - d) in A_set and (x + d) in A_set and (x + 2*d) in A_set:
                return False
        # 3rd: x-2d, x-d, x, x+d
        for d in range(1, x // 2 + 1):
            if (x - 2*d) in A_set and (x - d) in A_set and (x + d) in A_set:
                return False
        # 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (x - 3*d) in A_set and (x - 2*d) in A_set and (x - d) in A_set:
                return False
        return True

    # Helper: Check if swapping u (in A) with v (not in A) keeps it 4-AP free
    def can_swap(u, v, A):
        if u not in A or v in A:
            return False
        A_new = set(A)
        A_new.remove(u)
        A_new.add(v)
        return is_4ap_free(A_new)

    # Initial construction: Greedy 4-AP-free set
    # We iterate 1..N and add if possible.
    A = set()
    for x in range(1, N + 1):
        if can_add(x, A):
            A.add(x)
            
    current_score = count_3aps(A)
    
    # Local Search: Hill Climbing
    # We try to improve by adding, removing, or swapping.
    # Since we start with a maximal 4-AP-free set (in terms of inclusion), we can't add anything.
    # So we focus on Swaps.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_gain = 0
        best_move = None # (u, v) for swap, or None for add (not possible here)
        
        A_list = list(A)
        # Try all swaps
        for u in A_list:
            # Try all v not in A
            for v in range(1, N + 1):
                if v in A:
                    continue
                if not can_swap(u, v, A):
                    continue
                
                # Calculate gain
                # Remove u, add v
                # Gain = add_gain(v, A - {u}) - remove_loss(u, A)
                # Note: add_gain(v, A-{u}) considers interactions with A-{u}
                # remove_loss(u, A) considers interactions with A
                
                A_temp = set(A)
                A_temp.remove(u)
                gain_v = add_gain(v, A_temp)
                loss_u = remove_loss(u, A)
                
                net_gain = gain_v - loss_u
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_move = (u, v)
        
        if best_move is not None:
            u, v = best_move
            A.remove(u)
            A.add(v)
            current_score += best_gain
            improved = True

    return sorted(A)
