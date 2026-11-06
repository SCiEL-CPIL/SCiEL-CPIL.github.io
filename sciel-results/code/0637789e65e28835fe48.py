"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    target_mask = (1 << 30) - 1  # bits 0-29 for diffs 1-30
    
    def search(elements, diff_mask, max_elem, best_result, depth=0):
        """Backtracking with strong pruning."""
        if time.time() > deadline:
            return best_result
        
        # Check completion
        if diff_mask == target_mask:
            current_max = max(elements)
            if best_result is None or current_max < best_result[0]:
                return (current_max, list(elements))
            return best_result
        
        # Prune: can't beat best
        if best_result is not None and max_elem >= best_result[0]:
            return best_result
        
        # Get missing differences
        missing_bits = ~diff_mask & target_mask
        
        # Generate candidates that add at least one new difference
        candidates = []
        for candidate in range(1, max_elem + 1):
            if candidate in elements:
                continue
            
            valid = True
            new_diffs = 0
            for x in elements:
                d = abs(candidate - x)
                if d == 0:
                    valid = False
                    break
                if 1 <= d <= 30:
                    if diff_mask & (1 << (d - 1)):
                        valid = False
                        break
                    new_diffs |= (1 << (d - 1))
            
            if not valid or new_diffs == 0:
                continue
            
            # Count how many missing diffs this covers
            covered = bin(new_diffs & missing_bits).count('1')
            candidates.append((covered, candidate, new_diffs))
        
        # Sort: most coverage first, then smallest value
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Limit branching
        max_branches = 25
        for covered, candidate, new_diffs in candidates[:max_branches]:
            new_elements = elements + [candidate]
            new_diff_mask = diff_mask | new_diffs
            
            best_result = search(new_elements, new_diff_mask, max_elem, best_result, depth + 1)
            
            if best_result is not None:
                break
        
        return best_result
    
    # Try increasing max element bounds
    best_result = None
    
    for max_bound in range(30, 150):
        if time.time() > deadline:
            break
        
        result = search([0], 0, max_bound, best_result)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result[1]
    
    # Fallback: greedy
    def diffs(s):
        s = sorted(s)
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    def is_sidon(elements):
        d = diffs(elements)
        return len(set(d)) == len(d)
    
    a = [0]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
