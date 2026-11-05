import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflicts using bitsets for speed
    # conflict_mask[v] has bit j set if (v*j) % (v+j) == 0 and j != v
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))  # bit (j-1) represents vertex j
        conflict_mask[i] = mask

    all_mask = (1 << N) - 1  # bits 0..255 set

    # Greedy initialization with min-degree heuristic
    def greedy_min_degree(rem):
        solution = []
        while rem:
            # Find vertex with minimum degree
            min_v = None
            min_deg = float('inf')
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                deg = (conflict_mask[v] & rem).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                temp ^= lsb
            solution.append(min_v)
            rem &= ~(1 << (min_v - 1))
            rem &= ~conflict_mask[min_v]
        return solution

    # Initial greedy solution
    best_sol = greedy_min_degree(all_mask)
    best_size = len(best_sol)

    # Branch and bound exact solver
    # State: (remaining_mask, current_solution_size)
    # Upper bound: current_size + max_independent_set(remaining_mask)
    # We use a simple upper bound: number of vertices in remaining_mask
    # But we can do better with greedy coloring or similar
    
    # For better pruning, we'll use a greedy upper bound
    def greedy_upper_bound(rem):
        """Greedy upper bound on independent set size in rem."""
        count = 0
        while rem:
            # Pick vertex with min degree
            min_v = None
            min_deg = float('inf')
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                deg = (conflict_mask[v] & rem).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                temp ^= lsb
            count += 1
            rem &= ~(1 << (min_v - 1))
            rem &= ~conflict_mask[min_v]
        return count

    # Branch and bound
    def branch_and_bound(rem, current_size, best_size_ref):
        """
        Recursive B&B. Returns best size found.
        rem: bitmask of remaining vertices
        current_size: size of solution so far
        best_size_ref: list containing [best_size] for mutable reference
        """
        if not rem:
            return current_size
        
        # Upper bound: if we can't beat best, prune
        ub = current_size + greedy_upper_bound(rem)
        if ub <= best_size_ref[0]:
            return best_size_ref[0]
        
        # Pick a vertex to branch on
        # Choose vertex with max degree for better pruning
        max_v = None
        max_deg = -1
        temp = rem
        while temp:
            lsb = temp & (-temp)
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            deg = (conflict_mask[v] & rem).bit_count()
            if deg > max_deg:
                max_deg = deg
                max_v = v
            temp ^= lsb
        
        # Branch 1: include max_v
        new_rem = rem & ~(1 << (max_v - 1))
        new_rem &= ~conflict_mask[max_v]
        best_size_ref[0] = branch_and_bound(new_rem, current_size + 1, best_size_ref)
        
        # Branch 2: exclude max_v
        new_rem2 = rem & ~(1 << (max_v - 1))
        best_size_ref[0] = branch_and_bound(new_rem2, current_size, best_size_ref)
        
        return best_size_ref[0]

    # Run B&B with time limit
    best_size_ref = [best_size]
    
    # Use iterative deepening or just run B&B with deadline check
    # For safety, we'll limit the recursion depth implicitly by time
    
    # First, try to improve with randomized restarts
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        rem = all_mask
        sol = []
        
        while rem:
            temp = rem
            candidates = []
            deg_list = []
            
            while temp:
                lsb = temp & (-temp)
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                deg = (conflict_mask[v] & rem).bit_count()
                candidates.append(v)
                deg_list.append(deg)
                temp ^= lsb
            
            sorted_pairs = sorted(zip(deg_list, candidates))
            k = min(3, len(sorted_pairs))
            if k == 0:
                break
            chosen = sorted_pairs[random.randrange(k)][1]
            
            sol.append(chosen)
            rem &= ~(1 << (chosen - 1))
            rem &= ~conflict_mask[chosen]
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_size_ref[0] = best_size
    
    # Now run B&B to try to find exact solution
    # Use a modified B&B that checks deadline
    def bb_with_deadline(rem, current_size, best_size_ref, depth=0):
        if time.time() > deadline:
            return best_size_ref[0]
        
        if not rem:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
            return best_size_ref[0]
        
        # Upper bound
        ub = current_size + greedy_upper_bound(rem)
        if ub <= best_size_ref[0]:
            return best_size_ref[0]
        
        # Pick vertex with max degree
        max_v = None
        max_deg = -1
        temp = rem
        while temp:
            lsb = temp & (-temp)
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            deg = (conflict_mask[v] & rem).bit_count()
            if deg > max_deg:
                max_deg = deg
                max_v = v
            temp ^= lsb
        
        # Branch: include
        new_rem = rem & ~(1 << (max_v - 1))
        new_rem &= ~conflict_mask[max_v]
        bb_with_deadline(new_rem, current_size + 1, best_size_ref, depth + 1)
        
        # Branch: exclude
        new_rem2 = rem & ~(1 << (max_v - 1))
        bb_with_deadline(new_rem2, current_size, best_size_ref, depth + 1)
        
        return best_size_ref[0]

    # Run B&B starting from empty solution
    # This is expensive, so we'll limit it
    if time.time() < deadline - 5:  # Leave 5s for return
        # Try B&B on a reduced problem
        # Start with vertices that are not in current best solution
        # Or just run on full problem with time limit
        bb_with_deadline(all_mask, 0, best_size_ref)
    
    # Final local search to polish
    best_sol_set = set(best_sol)
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        for v in vertices:
            if v in best_sol_set:
                continue
            # Check if v can be added
            can_add = True
            for u in best_sol:
                if conflict_mask[v] & (1 << (u - 1)):
                    can_add = False
                    break
            if can_add:
                best_sol.append(v)
                best_sol_set.add(v)
                improved = True
                break
        if not improved:
            break
    
    return best_sol
