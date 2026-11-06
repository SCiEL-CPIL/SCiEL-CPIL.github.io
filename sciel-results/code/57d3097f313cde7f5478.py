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
        
        if attempt % 7 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempt % 7 == 1:
            order = sorted(range(1, n + 1), key=lambda x: degs[x])
        elif attempt % 7 == 2:
            order = sorted(range(1, n + 1), key=lambda x: -degs[x])
        elif attempt % 7 == 3:
            order = list(range(n, 0, -1))
            random.shuffle(order)
        elif attempt % 7 == 4:
            # Weighted random by degree
            weights = [degs[x] + 1 for x in range(1, n + 1)]
            order = []
            remaining = list(range(1, n + 1))
            while remaining:
                w = [degs[x] + 1 for x in remaining]
                total_w = sum(w)
                r = random.random() * total_w
                cum = 0
                for idx, x in enumerate(remaining):
                    cum += w[idx]
                    if r <= cum:
                        order.append(x)
                        remaining.pop(idx)
                        break
        elif attempt % 7 == 5:
            # Sort by number of conflicts with already-selected (approximate)
            order = sorted(range(1, n + 1), key=lambda x: sum(1 for y in range(1, n+1) if y != x and conflict_mask[x][y]))
        else:
            order = list(range(1, n + 1))
            random.shuffle(order)
        
        candidate_mask = greedy_add(order)
        cand_size = bin(candidate_mask).count('1')
        if cand_size > best_size:
            best_mask = candidate_mask
            best_size = cand_size
    
    # Local search: 1-for-1 and 1-for-2 swaps
    if time.time() < deadline - 15.0:
        for iteration in range(100):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            selected_list = mask_to_set(best_mask)
            
            # Try 1-for-1 swaps: remove x, add y
            for x in selected_list:
                if time.time() > deadline - 5.0:
                    break
                rest_mask = best_mask & ~(1 << x)
                
                for y in range(1, n + 1):
                    if (best_mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        new_mask = rest_mask | (1 << y)
                        new_size = bin(new_mask).count('1')
                        if new_size > best_size:
                            best_mask = new_mask
                            best_size = new_size
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swaps: remove x, add y and z
            selected_list = mask_to_set(best_mask)
            for x in selected_list:
                if time.time() > deadline - 5.0:
                    break
                rest_mask = best_mask & ~(1 << x)
                
                # Find all y that can be added to rest_mask
                addable = []
                for y in range(1, n + 1):
                    if (best_mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        addable.append(y)
                
                # Try pairs from addable
                for i in range(len(addable)):
                    if time.time() > deadline - 5.0:
                        break
                    y1 = addable[i]
                    temp_mask = rest_mask | (1 << y1)
                    for j in range(i + 1, len(addable)):
                        y2 = addable[j]
                        if can_add_fast(y2, temp_mask):
                            new_mask = temp_mask | (1 << y2)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Stochastic local search: random removal and re-optimization
    if time.time() < deadline - 10.0:
        random.seed(123)
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Remove a random element
            selected_list = mask_to_set(best_mask)
            if not selected_list:
                break
            x_to_remove = random.choice(selected_list)
            current_mask = best_mask & ~(1 << x_to_remove)
            
            # Try to add back any elements
            for y in range(1, n + 1):
                if (current_mask >> y) & 1:
                    continue
                if can_add_fast(y, current_mask):
                    current_mask |= (1 << y)
            
            # Try 1-for-1 swaps on the modified set
            for x in mask_to_set(current_mask):
                if time.time() > deadline - 5.0:
                    break
                rest_mask = current_mask & ~(1 << x)
                for y in range(1, n + 1):
                    if (current_mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        new_mask = rest_mask | (1 << y)
                        new_size = bin(new_mask).count('1')
                        if new_size > best_size:
                            best_mask = new_mask
                            best_size = new_size
                            break
                if bin(best_mask).count('1') > bin(current_mask).count('1'):
                    current_mask = best_mask
            
            if bin(current_mask).count('1') > best_size:
                best_mask = current_mask
                best_size = bin(current_mask).count('1')
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 3.0:
        for x in range(1, n + 1):
            if (best_mask >> x) & 1:
                continue
            if can_add_fast(x, best_mask):
                best_mask |= (1 << x)
                best_size += 1
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
