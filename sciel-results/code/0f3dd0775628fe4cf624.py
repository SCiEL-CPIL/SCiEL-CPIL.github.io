import random
import time
import sys

N = 9

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Precompute bit masks for efficiency
    # Not strictly necessary for N=9, but good practice
    
    best_path = None
    best_len = 0
    
    def is_valid_snake(path):
        """Check if the path is a valid induced path."""
        if not path:
            return True
        if len(set(path)) != len(path):
            return False
        used = set(path)
        for i in range(len(path) - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        # Check non-consecutive distances
        # Optimization: Only check against vertices not adjacent in the path
        # For N=9, full check is O(L^2 * bit_count). L ~ 150, so 22500 checks. Fast enough.
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True

    def greedy_extend(path):
        """Try to extend the path at both ends greedily."""
        path = path[:]
        changed = True
        while changed:
            changed = False
            # Try to extend end
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if v not in set(path):
                    # Check distance to all non-adjacent
                    valid = True
                    for i in range(len(path) - 1): # path[:-1]
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        path.append(v)
                        changed = True
                        break
            # Try to extend start
            for b in range(N):
                v = path[0] ^ (1 << b)
                if v not in set(path):
                    valid = True
                    for i in range(1, len(path)): # path[1:]
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        path.insert(0, v)
                        changed = True
                        break
        return path

    def get_neighbors_2bit(path, i):
        """
        Get the alternative vertex for a 2-bit flip at index i.
        Returns the new vertex if valid move structure exists, else None.
        """
        if i == 0 or i == len(path) - 1:
            return None
        u = path[i-1]
        v = path[i]
        w = path[i+1]
        
        # u and w must be at distance 2 from each other
        if (u ^ w).bit_count() != 2:
            return None
            
        # The alternative vertex v' is u ^ w ^ v? No.
        # v = u ^ e_a, w = u ^ e_b.
        # v' = u ^ e_a ^ e_b.
        # Note: u ^ w = e_a ^ e_b.
        # So v' = v ^ (u ^ w)
        v_prime = v ^ (u ^ w)
        
        # v_prime must not be in the path (except potentially u or w? No, v_prime != u and v_prime != w)
        # v_prime is distinct from u, v, w.
        return v_prime

    def apply_2bit_flip(path, i):
        """
        Attempt to flip vertex at index i.
        Returns new path if successful, else None.
        """
        if i == 0 or i == len(path) - 1:
            return None
        
        u = path[i-1]
        v = path[i]
        w = path[i+1]
        
        # Check if u and w form a square with v
        if (u ^ w).bit_count() != 2:
            return None
            
        v_prime = v ^ (u ^ w)
        
        # Check if v_prime is already in the path
        if v_prime in path:
            return None
            
        # Check if v_prime is at distance 1 from any non-adjacent vertex
        # Non-adjacent to i are 0..i-2 and i+2..L-1
        for j in range(len(path)):
            if j == i-1 or j == i or j == i+1:
                continue
            if (v_prime ^ path[j]).bit_count() == 1:
                return None
                
        # Also need to ensure that removing v doesn't break anything else?
        # The rest of the path is unchanged. The new vertex v_prime replaces v.
        # The connections u-v_prime and v_prime-w are valid by construction (dist 1).
        # The only new constraints are dist(v_prime, path[j]) for j not in {i-1, i, i+1}.
        
        new_path = path[:]
        new_path[i] = v_prime
        return new_path

    def sa_search(initial_path, time_limit):
        """
        Simulated Annealing search starting from initial_path.
        """
        current_path = initial_path[:]
        best_path = current_path[:]
        best_len = len(current_path)
        
        T = 1.0
        T_min = 0.01
        alpha = 0.9995
        # Adjust alpha based on expected iterations? 
        # Let's just run until time limit or T < T_min
        
        path_set = set(current_path)
        
        while T > T_min:
            if time.time() > time_limit:
                break
                
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.7:
                # 2-bit flip
                if len(current_path) > 2:
                    i = random.randint(1, len(current_path) - 2)
                    new_path = apply_2bit_flip(current_path, i)
                    if new_path is not None:
                        # Calculate energy difference
                        # Energy = -len(path) ? No, we want to maximize length.
                        # But flips keep length constant.
                        # So we accept flips if they don't violate constraints (which we checked)
                        # and allow future extensions.
                        # Actually, a flip might enable an extension that wasn't possible before.
                        # So we accept flips with high probability if valid.
                        # Let's accept valid flips with probability 1.0 initially, 
                        # or use a small penalty if it reduces "flexibility"?
                        # For now, accept all valid flips.
                        current_path = new_path
                        path_set = set(current_path)
                        
                        # Try to extend after a flip
                        extended = greedy_extend(current_path)
                        if len(extended) > len(current_path):
                            current_path = extended
                            path_set = set(current_path)
                            
                            if len(current_path) > best_len:
                                best_len = len(current_path)
                                best_path = current_path[:]
            else:
                # Try to insert a random vertex? Or delete?
                # Deletion can help escape dead ends.
                if len(current_path) > 10:
                    i = random.randint(1, len(current_path) - 2)
                    v = current_path[i]
                    # Remove v
                    new_path = current_path[:i] + current_path[i+1:]
                    # Check if new_path is valid
                    # The neighbors of v were u and w. Now u and w are adjacent in the list.
                    # They must be at distance 1 in the hypercube for the path to remain a path.
                    if (current_path[i-1] ^ current_path[i+1]).bit_count() == 1:
                        # Also check no new distance-1 conflicts introduced?
                        # Removing a vertex cannot introduce new distance-1 conflicts.
                        # It only removes constraints.
                        # So new_path is always a valid induced path if u-w are adjacent.
                        # Wait, if u and w are NOT adjacent, we can't just remove v.
                        # We must maintain the path structure.
                        # So deletion is only valid if u and w are adjacent.
                        if True: # Validity check passed
                            # Accept deletion with some probability to explore shorter paths?
                            # Or only if it leads to extension?
                            # Let's accept deletion with probability T
                            if random.random() < T:
                                current_path = new_path
                                path_set = set(current_path)
                    
            # Update temperature
            T *= alpha
            
        # Final extension
        extended = greedy_extend(current_path)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended[:]
            
        return best_path

    # Initial Greedy Seed
    def greedy_seed():
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check distance to all previous non-adjacent
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path

    # Run multiple restarts with SA
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Generate a diverse seed
        if restarts == 1:
            seed_path = greedy_seed()
        else:
            # Random start
            start_v = random.randint(0, (1 << N) - 1)
            seed_path = [start_v]
            used = {start_v}
            # Greedy extend from random start
            while True:
                cur = seed_path[-1]
                nxt = None
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v in used:
                        continue
                    valid = True
                    for i in range(len(seed_path) - 1):
                        if (v ^ seed_path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        nxt = v
                        break
                if nxt is None:
                    break
                seed_path.append(nxt)
                used.add(nxt)
                
        # Run SA for a limited time slice
        # Allocate time: e.g., 5 seconds per restart, but check deadline
        slice_end = min(time.time() + 5.0, deadline)
        improved_path = sa_search(seed_path, slice_end)
        
        if len(improved_path) > best_len:
            best_len = len(improved_path)
            best_path = improved_path[:]
            
        # If we found a very long path, maybe refine it with more SA time?
        # For now, just continue restarts.

    return best_path
# EVOLVE-BLOCK-END
