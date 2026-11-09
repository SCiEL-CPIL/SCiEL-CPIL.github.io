# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Generate 5-smooth numbers up to 128
    smooths = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooths.append(x)
    
    n = len(smooths)
    
    # Precompute divides matrix: divides[i][j] = True if smooths[i] divides smooths[j]
    divides = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and smooths[j] % smooths[i] == 0:
                divides[i][j] = True
    
    def is_antichain(indices):
        """Check if the set of indices forms a divisibility antichain."""
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if divides[i][j] or divides[j][i]:
                    return False
        return True
    
    # best[target] = list of smooth numbers (the antichain)
    best = [None] * 129
    
    # k=1: single smooth numbers
    for i in range(n):
        val = smooths[i]
        if val <= 128 and best[val] is None:
            best[val] = [val]
    
    if time.time() - start_time > deadline:
        return _fill_missing(best, smooths)
    
    # k=2: pairs
    for i in range(n):
        vi = smooths[i]
        for j in range(i + 1, n):
            vj = smooths[j]
            s = vi + vj
            if s > 128:
                break
            if best[s] is None:
                if not (divides[i][j] or divides[j][i]):
                    best[s] = [vi, vj]
    
    if time.time() - start_time > deadline:
        return _fill_missing(best, smooths)
    
    # k=3: triples
    for i in range(n):
        vi = smooths[i]
        for j in range(i + 1, n):
            vj = smooths[j]
            s2 = vi + vj
            if s2 > 128:
                break
            for k in range(j + 1, n):
                vk = smooths[k]
                s3 = s2 + vk
                if s3 > 128:
                    break
                if best[s3] is None:
                    if not (divides[i][j] or divides[j][i] or divides[i][k] or divides[k][i] or divides[j][k] or divides[k][j]):
                        best[s3] = [vi, vj, vk]
    
    if time.time() - start_time > deadline:
        return _fill_missing(best, smooths)
    
    # k=4: quadruples
    for i in range(n):
        vi = smooths[i]
        for j in range(i + 1, n):
            vj = smooths[j]
            s2 = vi + vj
            if s2 > 128:
                break
            for k in range(j + 1, n):
                vk = smooths[k]
                s3 = s2 + vk
                if s3 > 128:
                    break
                for l in range(k + 1, n):
                    vl = smooths[l]
                    s4 = s3 + vl
                    if s4 > 128:
                        break
                    if best[s4] is None:
                        idxs = (i, j, k, l)
                        if is_antichain(idxs):
                            best[s4] = [vi, vj, vk, vl]
    
    if time.time() - start_time > deadline:
        return _fill_missing(best, smooths)
    
    # k=5: quintuples
    for i in range(n):
        vi = smooths[i]
        for j in range(i + 1, n):
            vj = smooths[j]
            s2 = vi + vj
            if s2 > 128:
                break
            for k in range(j + 1, n):
                vk = smooths[k]
                s3 = s2 + vk
                if s3 > 128:
                    break
                for l in range(k + 1, n):
                    vl = smooths[l]
                    s4 = s3 + vl
                    if s4 > 128:
                        break
                    for m in range(l + 1, n):
                        vm = smooths[m]
                        s5 = s4 + vm
                        if s5 > 128:
                            break
                        if best[s5] is None:
                            idxs = (i, j, k, l, m)
                            if is_antichain(idxs):
                                best[s5] = [vi, vj, vk, vl, vm]
    
    if time.time() - start_time > deadline:
        return _fill_missing(best, smooths)
    
    # k=6: sextuples
    for i in range(n):
        vi = smooths[i]
        for j in range(i + 1, n):
            vj = smooths[j]
            s2 = vi + vj
            if s2 > 128:
                break
            for k in range(j + 1, n):
                vk = smooths[k]
                s3 = s2 + vk
                if s3 > 128:
                    break
                for l in range(k + 1, n):
                    vl = smooths[l]
                    s4 = s3 + vl
                    if s4 > 128:
                        break
                    for m in range(l + 1, n):
                        vm = smooths[m]
                        s5 = s4 + vm
                        if s5 > 128:
                            break
                        for p in range(m + 1, n):
                            vp = smooths[p]
                            s6 = s5 + vp
                            if s6 > 128:
                                break
                            if best[s6] is None:
                                idxs = (i, j, k, l, m, p)
                                if is_antichain(idxs):
                                    best[s6] = [vi, vj, vk, vl, vm, vp]
    
    if time.time() - start_time > deadline:
        return _fill_missing(best, smooths)
    
    # For any remaining unsolved targets, use backtracking search
    for target in range(1, 129):
        if best[target] is not None:
            continue
        
        # Try to find any valid antichain summing to target
        # Use backtracking with pruning
        found = _find_antichain_backtrack(smooths, divides, target, start_time, deadline)
        if found is not None:
            best[target] = found
    
    # Final check: if any target is still unsolved, use a simple fallback
    return _fill_missing(best, smooths)


def _fill_missing(best, smooths):
    """Fill any missing targets with a valid representation."""
    n = len(smooths)
    
    # Precompute divides matrix
    divides = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and smooths[j] % smooths[i] == 0:
                divides[i][j] = True
    
    def is_antichain(indices):
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if divides[i][j] or divides[j][i]:
                    return False
        return True
    
    for target in range(1, 129):
        if best[target] is not None:
            continue
        
        # Try increasing k
        for k in range(1, min(31, n + 1)):
            # Generate combinations of size k
            found = False
            for combo in combinations(range(n), k):
                s = sum(smooths[i] for i in combo)
                if s == target:
                    if is_antichain(combo):
                        best[target] = [smooths[i] for i in combo]
                        found = True
                        break
                elif s > target:
                    # Since smooths is sorted, if we're going in increasing order of indices,
                    # this doesn't necessarily mean we can break, but it's a hint
                    pass
            
            if found:
                break
        
        # If still not found, use a simple greedy fallback
        if best[target] is None:
            # Try to build a representation greedily
            remaining = target
            chosen = []
            # Sort smooths in descending order for greedy
            for val in sorted(smooths, reverse=True):
                if val <= remaining and val not in chosen:
                    # Check antichain condition
                    valid = True
                    for c in chosen:
                        if val % c == 0 or c % val == 0:
                            valid = False
                            break
                    if valid:
                        chosen.append(val)
                        remaining -= val
                        if remaining == 0:
                            break
            
            if remaining == 0 and chosen:
                best[target] = chosen
            else:
                # Last resort: use {target} if target is smooth, else fail
                # This shouldn't happen for 1..128
                best[target] = [target] if target in smooths else [1, target - 1]
    
    return best


def _find_antichain_backtrack(smooths, divides, target, start_time, deadline):
    """Backtracking search for an antichain summing to target."""
    n = len(smooths)
    
    def backtrack(start_idx, current_sum, current_indices):
        if time.time() - start_time > deadline:
            return None
        if current_sum > target:
            return None
        if current_sum == target:
            if current_indices:
                return [smooths[i] for i in current_indices]
            return None
        
        for i in range(start_idx, n):
            val = smooths[i]
            if current_sum + val > target:
                break
            
            # Check antichain condition
            valid = True
            for j in current_indices:
                if divides[i][j] or divides[j][i]:
                    valid = False
                    break
            
            if valid:
                current_indices.append(i)
                result = backtrack(i + 1, current_sum + val, current_indices)
                if result is not None:
                    return result
                current_indices.pop()
        
        return None
    
    return backtrack(0, 0, [])
# EVOLVE-BLOCK-END
