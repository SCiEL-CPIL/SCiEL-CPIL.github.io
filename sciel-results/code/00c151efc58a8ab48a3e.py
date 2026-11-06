"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations
    import time
    
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute conflict sets for speed
    # conflicts[x] is a set of y such that lcm(x, y) <= 256
    conflicts = [set() for _ in range(257)]
    for i in range(2, 257):
        for j in range(i+1, 257):
            if lcm(i, j) <= 256:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    def is_compatible_with_set(x, s):
        """Check if x is compatible with all elements in set s"""
        for y in s:
            if y in conflicts[x]:
                return False
        return True

    def can_add_pair(y, z, s):
        """Check if y and z can both be added to set s"""
        if y in conflicts[z]:
            return False
        if not is_compatible_with_set(y, s):
            return False
        if not is_compatible_with_set(z, s):
            return False
        return True

    # 1. Greedy Initialization
    selected = []
    for x in sorted(candidates, key=lambda v: 1.0/v, reverse=True):
        if is_compatible_with_set(x, selected):
            selected.append(x)
    
    selected_set = set(selected)
    current_sum = sum(1.0/x for x in selected)
    
    start_time = time.time()
    improved = True
    
    # 2. Local Search
    while improved and (time.time() - start_time) < 55:
        improved = False
        
        # 2a. 1-for-1 swaps
        for i, x in enumerate(selected):
            # Try to replace x with y
            for y in candidates:
                if y in selected_set:
                    continue
                # Check compatibility with selected \ {x}
                if is_compatible_with_set(y, selected_set - {x}):
                    new_sum = current_sum - 1.0/x + 1.0/y
                    if new_sum > current_sum + 1e-12:
                        selected[i] = y
                        selected_set.remove(x)
                        selected_set.add(y)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # 2b. 1-for-2 swaps (remove x, add y, z)
        for i, x in enumerate(selected):
            s_minus_x = selected_set - {x}
            # Find candidates compatible with s_minus_x
            # We want max 1/y + 1/z > 1/x
            # Sort candidates by weight descending
            # But we need to check pair compatibility
            cands = [y for y in candidates if y not in selected_set and is_compatible_with_set(y, s_minus_x)]
            if len(cands) < 2:
                continue
            # Sort by 1/y descending
            cands.sort(key=lambda v: 1.0/v, reverse=True)
            # Try pairs from top of list
            found = False
            # Optimization: if top two don't satisfy, maybe deeper ones do?
            # But 1/y + 1/z is maximized by largest y, z.
            # So if best pair fails, others will likely fail if 1/x is large?
            # Actually, we just need ANY pair with sum > 1/x.
            # Iterate through pairs. To save time, break if 1/cands[k] + 1/cands[l] <= 1/x 
            # since list is sorted descending, further pairs will have smaller sums.
            for a_idx in range(len(cands)):
                y = cands[a_idx]
                # If even with the largest possible partner (cands[0] if a_idx!=0 else cands[1]) sum <= 1/x, break?
                # Max possible sum for this y is 1/y + 1/max_other_compatible.
                # Let's just check pairs.
                for b_idx in range(a_idx + 1, len(cands)):
                    z = cands[b_idx]
                    # If 1/y + 1/z <= 1/x, no need to check smaller z
                    if 1.0/y + 1.0/z <= 1.0/x:
                        break
                    if not (y in conflicts[z]):
                        # Check if y, z compatible with s_minus_x (already done for cands)
                        # And compatible with each other (checked above)
                        new_sum = current_sum - 1.0/x + 1.0/y + 1.0/z
                        if new_sum > current_sum + 1e-12:
                            selected[i] = y
                            selected.append(z)
                            selected_set.remove(x)
                            selected_set.add(y)
                            selected_set.add(z)
                            current_sum = new_sum
                            improved = True
                            found = True
                            break
                if found:
                    break
            if found:
                break
                
        if improved:
            continue

        # 2c. 2-for-1 swaps (remove x, y, add z)
        # This is rarer to be beneficial but possible
        n_sel = len(selected)
        for i in range(n_sel):
            for j in range(i+1, n_sel):
                x, y = selected[i], selected[j]
                s_minus_xy = selected_set - {x, y}
                # Find z compatible with s_minus_xy
                # Need 1/z > 1/x + 1/y
                best_z = None
                best_z_val = -1
                for z in candidates:
                    if z in selected_set:
                        continue
                    if 1.0/z <= 1.0/x + 1.0/y:
                        continue
                    if is_compatible_with_set(z, s_minus_xy):
                        if 1.0/z > best_z_val:
                            best_z_val = 1.0/z
                            best_z = z
                if best_z is not None:
                    # Remove x, y, add best_z
                    selected[i] = best_z
                    selected.pop(j)
                    selected_set.remove(x)
                    selected_set.remove(y)
                    selected_set.add(best_z)
                    current_sum = current_sum - 1.0/x - 1.0/y + 1.0/best_z
                    improved = True
                    break
            if improved:
                break

    return selected
# EVOLVE-BLOCK-END
