import time
import random
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 55.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # For each position, precompute which APs it's in and its role (0,1,2,3)
    ap_indices_for_pos = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j, pos in enumerate(ap):
            ap_indices_for_pos[pos].append((idx, j))
    
    # Precompute forbidden-color lookup table:
    # For each AP and each position j in that AP, given the colors of the other 3 positions,
    # which color for position j would cause a violation?
    # 
    # Key insight: a 4-AP is a violation iff it uses < 3 distinct colors.
    # Given 3 colors (a, b, c) for the other positions, color x for position j is forbidden iff
    # {x, a, b, c} has < 3 distinct elements.
    #
    # We can precompute for each triple (a, b, c) a bitmask of forbidden colors.
    # But 128^3 is too large. Instead, we compute on-the-fly but use bitmasks.
    #
    # Better: For a given AP with other colors (a, b, c), the forbidden colors are:
    # - x such that {x, a, b, c} has <= 2 distinct values
    # This means x must be in {a, b, c} AND the set {a, b, c, x} has <= 2 distinct.
    #
    # Let's think more carefully:
    # {x, a, b, c} has < 3 distinct iff:
    #   - x == a == b == c (all same, 1 distinct)
    #   - exactly 2 distinct values among {x, a, b, c}
    #
    # If a, b, c are all the same (a==b==c), then:
    #   - x == a: 1 distinct -> forbidden
    #   - x != a: 2 distinct -> forbidden
    #   So ALL colors are forbidden! This means we must avoid having 3 positions in an AP with same color.
    #
    # If exactly 2 of {a,b,c} are the same, say a==b != c:
    #   - x == a: {a, a, a, c} -> 2 distinct -> forbidden
    #   - x == c: {a, a, c, c} -> 2 distinct -> forbidden  
    #   - x != a and x != c: {a, a, c, x} -> 3 distinct -> OK
    #   So forbidden: {a, c}
    #
    # If all 3 of {a,b,c} are distinct:
    #   - x == a: {a, a, b, c} -> 3 distinct -> OK
    #   - x == b: {a, b, b, c} -> 3 distinct -> OK
    #   - x == c: {a, b, c, c} -> 3 distinct -> OK
    #   - x != a,b,c: 4 distinct -> OK
    #   So NO colors are forbidden!
    
    def get_forbidden_mask(a, b, c, k):
        """Return a k-bit mask where bit i is set if color i is forbidden given other colors a, b, c."""
        if a == b == c:
            # All colors forbidden
            return (1 << k) - 1
        elif a == b or a == c or b == c:
            # Two same, one different
            # Forbidden: the two values that appear
            mask = 0
            for val in {a, b, c}:
                if val < k:
                    mask |= (1 << val)
            return mask
        else:
            # All three distinct: no forbidden colors
            return 0
    
    def count_violations_fast(colors):
        """Count total violations using precomputed AP list."""
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Count distinct
            distinct = len({c0, c1, c2, c3})
            if distinct < 3:
                count += 1
        return count
    
    def is_violation(c0, c1, c2, c3):
        """Check if a 4-tuple is a violation (< 3 distinct colors)."""
        if c0 == c1:
            if c2 == c3:
                return True  # all same or two pairs
            if c2 == c0 or c3 == c0 or c2 == c3:
                return True
            return False
        elif c0 == c2:
            if c1 == c3 or c1 == c0 or c3 == c0:
                return True
            return False
        elif c0 == c3:
            if c1 == c2 or c1 == c0 or c2 == c0:
                return True
            return False
        elif c1 == c2:
            if c0 == c3 or c0 == c1 or c3 == c1:
                return True
            return False
        elif c1 == c3:
            if c0 == c2 or c0 == c1 or c2 == c1:
                return True
            return False
        elif c2 == c3:
            if c0 == c1 or c0 == c2 or c1 == c2:
                return True
            return False
        return False
    
    def greedy_construct(k, seed):
        """Build a valid coloring with k colors using bitmask-based greedy."""
        random.seed(seed)
        colors = [-1] * N
        
        # Try multiple orderings
        for attempt in range(5):
            colors = [-1] * N
            order = list(range(N))
            if attempt == 0:
                pass  # natural order
            elif attempt == 1:
                random.shuffle(order)
            elif attempt == 2:
                order = order[::-1]
            elif attempt == 3:
                # Sort by number of APs (most constrained first)
                order.sort(key=lambda p: len(ap_indices_for_pos[p]), reverse=True)
            else:
                random.shuffle(order)
            
            success = True
            for pos in order:
                if time.time() > deadline:
                    return None, None
                
                # Aggregate forbidden mask from all APs where all other 3 are assigned
                forbidden_mask = 0
                
                for ap_idx, j in ap_indices_for_pos[pos]:
                    ap = aps[ap_idx]
                    other_positions = [p for p in range(4) if p != j]
                    other_colors = [colors[ap[p]] for p in other_positions]
                    
                    if any(c == -1 for c in other_colors):
                        continue
                    
                    oc0, oc1, oc2 = other_colors
                    mask = get_forbidden_mask(oc0, oc1, oc2, k)
                    forbidden_mask |= mask
                
                # Find lowest available color
                available_mask = ((1 << k) - 1) & ~forbidden_mask
                if available_mask == 0:
                    success = False
                    break
                
                # Find lowest set bit
                chosen = (available_mask & (-available_mask)).bit_length() - 1
                colors[pos] = chosen
            
            if success and all(c != -1 for c in colors):
                if count_violations_fast(colors) == 0:
                    return colors, attempt
        
        return None, None
    
    def local_search(colors, k, max_steps=10000):
        """Local search to fix violations using bitmask acceleration."""
        if colors is None:
            return None
        
        # Initialize AP violation status
        ap_violation = [False] * M
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            ap_violation[idx] = is_violation(c0, c1, c2, c3)
        
        current_violations = sum(ap_violation)
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            violating_aps = [idx for idx, v in enumerate(ap_violation) if v]
            if not violating_aps:
                break
            
            # Pick a random violating AP and a random position in it
            ap_idx = random.choice(violating_aps)
            ap = aps[ap_idx]
            pos = random.choice(ap)
            
            current_val = colors[pos]
            
            # Find best new color
            best_new_val = current_val
            best_delta = 0
            
            for new_val in range(k):
                if new_val == current_val:
                    continue
                
                delta = 0
                for ap_i, j in ap_indices_for_pos[pos]:
                    other_ap = aps[ap_i]
                    was_violation = ap_violation[ap_i]
                    
                    new_colors = [colors[other_ap[p]] for p in range(4)]
                    new_colors[j] = new_val
                    
                    nc0, nc1, nc2, nc3 = new_colors
                    new_violation = is_violation(nc0, nc1, nc2, nc3)
                    
                    if new_violation and not was_violation:
                        delta += 1
                    elif not new_violation and was_violation:
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_val = new_val
            
            if best_new_val != current_val:
                colors[pos] = best_new_val
                current_violations += best_delta
                
                for ap_i, j in ap_indices_for_pos[pos]:
                    other_ap = aps[ap_i]
                    nc0, nc1, nc2, nc3 = colors[other_ap[0]], colors[other_ap[1]], colors[other_ap[2]], colors[other_ap[3]]
                    ap_violation[ap_i] = is_violation(nc0, nc1, nc2, nc3)
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    break
            
            elif step % 50 == 0:
                # Random perturbation
                pos = random.randint(0, N - 1)
                new_val = random.randint(0, k - 1)
                if new_val != colors[pos]:
                    colors[pos] = new_val
                    for ap_i, j in ap_indices_for_pos[pos]:
                        other_ap = aps[ap_i]
                        nc0, nc1, nc2, nc3 = colors[other_ap[0]], colors[other_ap[1]], colors[other_ap[2]], colors[other_ap[3]]
                        ap_violation[ap_i] = is_violation(nc0, nc1, nc2, nc3)
                    current_violations = sum(ap_violation)
        
        return best_colors if best_violations == 0 else None
    
    # Search for minimum k
    best_solution = None
    
    for k in range(3, 40):
        if time.time() > deadline:
            break
        
        found = False
        # Try greedy with many seeds
        for seed in range(100):
            if time.time() > deadline:
                break
            
            colors, attempt = greedy_construct(k, seed)
            if colors is not None:
                if count_violations_fast(colors) == 0:
                    best_solution = colors
                    found = True
                    break
        
        if found:
            break
        
        # Try random init + local search
        if time.time() > deadline:
            break
        
        for seed in range(30):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            colors = [random.randint(0, k - 1) for _ in range(N)]
            result = local_search(colors, k, max_steps=5000)
            if result is not None:
                best_solution = result
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: try larger k with local search
        for k in range(40, 128):
            if time.time() > deadline:
                break
            colors = list(range(k)) + [0] * (N - k)
            if count_violations_fast(colors) == 0:
                best_solution = colors
                break
            result = local_search(colors, k, max_steps=2000)
            if result is not None:
                best_solution = result
                break
    
    if best_solution is None:
        # Ultimate fallback: all distinct
        best_solution = list(range(N))
    
    return best_solution
