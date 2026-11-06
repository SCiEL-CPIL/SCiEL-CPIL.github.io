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
    
    # Precompute blocked_by[x] = bitmask of elements that block x from being added
    # blocked_by[x] has bit a set if adding x to a set containing a would create a conflict
    # Actually, we need: for a given selected set S, x can be added iff for all a in S,
    # no element in S conflicts with (a,x). Let's precompute differently.
    
    # For each pair (a, x), conflict_mask[a][x] gives elements b that conflict.
    # To check if x can be added to S: for each a in S, (S & conflict_mask[a][x]) must be 0.
    # Equivalently: the union over a in S of (S & conflict_mask[a][x]) must be 0.
    # This is hard to precompute directly. Instead, let's use a different approach.
    
    # Precompute for each x, the set of pairs (a, b) that would conflict with x.
    # Actually, let's just optimize can_add_fast by iterating only over selected elements.
    
    def can_add_fast(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        # Iterate over bits in selected_mask
        m = selected_mask
        while m:
            a = (m & (-m)).bit_length() - 1  # lowest set bit
            if selected_mask & conflict_mask[a][x]:
                return False
            m &= m - 1
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
                m = rest_mask
                while m:
                    a = (m & (-m)).bit_length() - 1
                    if rest_mask & conflict_mask[a][x]:
                        has_conflict = True
                        break
                    m &= m - 1
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
    num_restarts = 300
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 20.0:
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
            # Weighted random based on inverse degree
            weights = [1.0 / max(1, degs[x]) for x in range(1, n + 1)]
            order = random.choices(range(1, n + 1), weights=weights, k=n)
            # Remove duplicates while preserving order
            seen = set()
            order2 = []
            for x in order:
                if x not in seen:
                    seen.add(x)
                    order2.append(x)
            order = order2
        
        candidate_mask = greedy_add(order)
        cand_size = bin(candidate_mask).count('1')
        if cand_size > best_size:
            best_mask = candidate_mask
            best_size = cand_size
    
    # Local search: 1-for-1 swaps followed by greedy fill
    if time.time() < deadline - 30.0:
        for iteration in range(500):
            if time.time() > deadline - 25.0:
                break
            
            improved = False
            selected_list = mask_to_set(best_mask)
            
            for x in selected_list:
                if time.time() > deadline - 25.0:
                    break
                rest_mask = best_mask & ~(1 << x)
                
                # Find all y that can be added to rest_mask
                for y in range(1, n + 1):
                    if (best_mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        new_mask = rest_mask | (1 << y)
                        # Try to greedily add more elements to new_mask
                        for z in range(1, n + 1):
                            if (new_mask >> z) & 1:
                                continue
                            if can_add_fast(z, new_mask):
                                new_mask |= (1 << z)
                        new_size = bin(new_mask).count('1')
                        if new_size > best_size:
                            best_mask = new_mask
                            best_size = new_size
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
    
    # Local search: 1-for-2 swaps (remove x, add y and z)
    if time.time() < deadline - 20.0:
        for iteration in range(100):
            if time.time() > deadline - 15.0:
                break
            
            improved = False
            selected_list = mask_to_set(best_mask)
            
            for x in selected_list:
                if time.time() > deadline - 15.0:
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
                    if time.time() > deadline - 15.0:
                        break
                    y1 = addable[i]
                    temp_mask = rest_mask | (1 << y1)
                    for j in range(i + 1, len(addable)):
                        y2 = addable[j]
                        if can_add_fast(y2, temp_mask):
                            new_mask = temp_mask | (1 << y2)
                            # Try to add more
                            for z in range(1, n + 1):
                                if (new_mask >> z) & 1:
                                    continue
                                if can_add_fast(z, new_mask):
                                    new_mask |= (1 << z)
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
    
    # Local search: 2-for-3 swaps (remove x,y, add a,b,c)
    if time.time() < deadline - 15.0:
        for iteration in range(50):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            selected_list = mask_to_set(best_mask)
            
            for i in range(len(selected_list)):
                if time.time() > deadline - 10.0:
                    break
                x1 = selected_list[i]
                for j in range(i + 1, len(selected_list)):
                    if time.time() > deadline - 10.0:
                        break
                    x2 = selected_list[j]
                    rest_mask = best_mask & ~(1 << x1) & ~(1 << x2)
                    
                    # Find all y that can be added to rest_mask
                    addable = []
                    for y in range(1, n + 1):
                        if (best_mask >> y) & 1:
                            continue
                        if can_add_fast(y, rest_mask):
                            addable.append(y)
                    
                    # Try triples from addable
                    for a in range(len(addable)):
                        if time.time() > deadline - 10.0:
                            break
                        y1 = addable[a]
                        temp1 = rest_mask | (1 << y1)
                        for b in range(a + 1, len(addable)):
                            y2 = addable[b]
                            if can_add_fast(y2, temp1):
                                temp2 = temp1 | (1 << y2)
                                for c in range(b + 1, len(addable)):
                                    if time.time() > deadline - 10.0:
                                        break
                                    y3 = addable[c]
                                    if can_add_fast(y3, temp2):
                                        new_mask = temp2 | (1 << y3)
                                        # Try to add more
                                        for z in range(1, n + 1):
                                            if (new_mask >> z) & 1:
                                                continue
                                            if can_add_fast(z, new_mask):
                                                new_mask |= (1 << z)
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
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Simulated annealing to escape local optima
    if time.time() < deadline - 10.0:
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        cur_mask = best_mask
        cur_size = best_size
        
        sa_iter = 0
        while time.time() < deadline - 5.0 and T > T_min:
            sa_iter += 1
            # Random move: remove 1-3 elements, add 1-4 elements
            num_remove = random.randint(1, 3)
            num_add = random.randint(1, 4)
            
            selected_list = mask_to_set(cur_mask)
            if len(selected_list) < num_remove:
                T *= cooling
                continue
            
            # Pick elements to remove
            to_remove = random.sample(selected_list, num_remove)
            rest_mask = cur_mask
            for x in to_remove:
                rest_mask &= ~(1 << x)
            
            # Pick elements to add
            candidates = [x for x in range(1, n + 1) if not (cur_mask >> x) & 1]
            if len(candidates) < num_add:
                T *= cooling
                continue
            
            to_add = random.sample(candidates, num_add)
            
            # Try to add them in random order
            new_mask = rest_mask
            add_order = to_add[:]
            random.shuffle(add_order)
            for y in add_order:
                if can_add_fast(y, new_mask):
                    new_mask |= (1 << y)
            
            new_size = bin(new_mask).count('1')
            delta = new_size - cur_size
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                cur_mask = new_mask
                cur_size = new_size
                if cur_size > best_size:
                    best_mask = cur_mask
                    best_size = cur_size
            
            T *= cooling
        
        # Final greedy fill
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if (best_mask >> x) & 1:
                    continue
                if can_add_fast(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
                    changed = True
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 3.0:
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if (best_mask >> x) & 1:
                    continue
                if can_add_fast(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
                    changed = True
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
