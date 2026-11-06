"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85.0
    n = 128
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute conflict_mask[a][x] = bitmask of b such that (a,b,x) forms a bad triple
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    conflict_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for x in range(1, n + 1):
            if a == x:
                continue
            g_ax = gcd_mat[a][x]
            mask = 0
            for b in range(1, n + 1):
                if b == a or b == x:
                    continue
                if gcd_mat[a][b] == g_ax and gcd_mat[b][x] == g_ax:
                    mask |= (1 << b)
            conflict_mask[a][x] = mask
    
    def can_add_fast(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        for a in range(1, n + 1):
            if (selected_mask >> a) & 1:
                if selected_mask & conflict_mask[a][x]:
                    return False
        return True
    
    def mask_to_set(mask):
        return [i for i in range(1, n + 1) if (mask >> i) & 1]
    
    def greedy_add(order):
        """Greedy add elements in given order."""
        selected_mask = 0
        for x in order:
            if not (selected_mask >> x) & 1:
                if can_add_fast(x, selected_mask):
                    selected_mask |= (1 << x)
        return selected_mask
    
    def greedy_remove():
        """Start with all elements, remove those that create conflicts."""
        selected_mask = 0
        for i in range(1, n + 1):
            selected_mask |= (1 << i)
        
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not ((selected_mask >> x) & 1):
                    continue
                rest_mask = selected_mask & ~(1 << x)
                has_conflict = False
                for a in range(1, n + 1):
                    if (rest_mask >> a) & 1:
                        if rest_mask & conflict_mask[a][x]:
                            has_conflict = True
                            break
                if has_conflict:
                    selected_mask &= ~(1 << x)
                    changed = True
        return selected_mask
    
    # Get initial solution via greedy removal
    best_mask = greedy_remove()
    best_size = bin(best_mask).count('1')
    
    # Precompute conflict degrees for ordering
    def compute_degrees():
        degs = {}
        for x in range(1, n + 1):
            deg = 0
            for y in range(1, n + 1):
                if x != y and (conflict_mask[x][y] or conflict_mask[y][x]):
                    deg += 1
            degs[x] = deg
        return degs
    
    degs = compute_degrees()
    
    # Try multiple greedy-add restarts with different orderings
    random.seed(42)
    num_restarts = 200
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if attempt % 5 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempt % 5 == 1:
            order = sorted(range(1, n + 1), key=lambda x: degs[x])
        elif attempt % 5 == 2:
            order = sorted(range(1, n + 1), key=lambda x: -degs[x])
        elif attempt % 5 == 3:
            order = list(range(n, 0, -1))
            random.shuffle(order)
        else:
            order = list(range(1, n + 1))
            random.shuffle(order)
        
        candidate_mask = greedy_add(order)
        cand_size = bin(candidate_mask).count('1')
        if cand_size > best_size:
            best_mask = candidate_mask
            best_size = cand_size
    
    # Enhanced local search with shake-ups and more aggressive re-optimization
    def full_local_search(start_mask, time_limit):
        """Perform local search starting from start_mask until no improvement or time limit."""
        mask = start_mask
        size = bin(mask).count('1')
        deadline_ls = time.time() + time_limit
        
        # Phase 1: 1-for-1 swaps
        while time.time() < deadline_ls:
            improved = False
            selected_list = mask_to_set(mask)
            
            for x in selected_list:
                if time.time() > deadline_ls:
                    break
                rest_mask = mask & ~(1 << x)
                
                for y in range(1, n + 1):
                    if (mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        new_mask = rest_mask | (1 << y)
                        if bin(new_mask).count('1') > size:
                            mask = new_mask
                            size += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        # Phase 2: 1-for-2 swaps (remove x, add y and z)
        while time.time() < deadline_ls:
            improved = False
            selected_list = mask_to_set(mask)
            
            for x in selected_list:
                if time.time() > deadline_ls:
                    break
                rest_mask = mask & ~(1 << x)
                
                addable = []
                for y in range(1, n + 1):
                    if (mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        addable.append(y)
                
                for i in range(len(addable)):
                    if time.time() > deadline_ls:
                        break
                    y1 = addable[i]
                    temp_mask = rest_mask | (1 << y1)
                    for j in range(i + 1, len(addable)):
                        y2 = addable[j]
                        if can_add_fast(y2, temp_mask):
                            new_mask = temp_mask | (1 << y2)
                            new_size = bin(new_mask).count('1')
                            if new_size > size:
                                mask = new_mask
                                size = new_size
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return mask, size
    
    # Shake-up local search: perturb and re-optimize
    if time.time() < deadline - 5.0:
        shake_attempts = 20
        for shake in range(shake_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Create a perturbed version: remove 2-4 random elements
            selected_list = mask_to_set(best_mask)
            if len(selected_list) < 5:
                break
            
            num_remove = min(random.randint(2, 4), len(selected_list) - 3)
            to_remove = random.sample(selected_list, num_remove)
            
            perturbed_mask = best_mask
            for x in to_remove:
                perturbed_mask &= ~(1 << x)
            
            # Try to recover by adding elements greedily
            for x in range(1, n + 1):
                if (perturbed_mask >> x) & 1:
                    continue
                if can_add_fast(x, perturbed_mask):
                    perturbed_mask |= (1 << x)
            
            # Run local search on perturbed solution
            new_mask, new_size = full_local_search(perturbed_mask, deadline - time.time() - 2.0)
            
            if new_size > best_size:
                best_mask = new_mask
                best_size = new_size
        
        # Final polish on best solution
        best_mask, best_size = full_local_search(best_mask, deadline - time.time() - 1.0)
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 1.0:
        for x in range(1, n + 1):
            if (best_mask >> x) & 1:
                continue
            if can_add_fast(x, best_mask):
                best_mask |= (1 << x)
                best_size += 1
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
