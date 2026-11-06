import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def is_valid(arr):
        if len(arr) > 32 or 0 not in arr or max(arr) > 10**6:
            return False
        diffs = set()
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def solve(D, max_time):
        best = None
        target_mask = 0
        for d in range(1, 31):
            target_mask |= (1 << d)
            
        def bt(sel, used_mask, mn):
            nonlocal best
            if best is not None or time.time() > max_time:
                return
            
            # Check coverage
            if (used_mask & target_mask) == target_mask:
                best = sel[:]
                return
            
            missing = 30 - popcount(used_mask & target_mask)
            if missing == 0:
                best = sel[:]
                return
                
            slots_left = 32 - len(sel)
            if slots_left == 0:
                return
            
            # Pruning: max new coverage per slot is min(len(sel), missing)
            # Total max coverage = slots_left * min(len(sel), missing)
            if missing > slots_left * min(len(sel), missing):
                return
            
            cands = []
            for x in range(mn, D + 1):
                ok = True
                new_used_bits = 0
                for s in sel:
                    d = x - s
                    if d <= 0:
                        continue
                    if used_mask & (1 << d):
                        ok = False
                        break
                    new_used_bits |= (1 << d)
                
                if not ok:
                    continue
                
                # Heuristic: count new bits in target_mask
                nc = popcount(new_used_bits & target_mask & ~used_mask)
                cands.append((x, nc, new_used_bits))
            
            if not cands:
                return
            
            cands.sort(key=lambda t: -t[1])
            
            for x, nc, nu_bits in cands:
                if best is not None or time.time() > max_time:
                    return
                bt(sel + [x], used_mask | nu_bits, x + 1)
        
        bt([0], 0, 1)
        return best

    for D in range(45, 201):
        if time.time() > deadline - 10:
            break
        r = solve(D, deadline - 5)
        if r is not None and is_valid(r):
            return r

    for D in range(200, 501):
        if time.time() > deadline - 3:
            break
        r = solve(D, deadline - 2)
        if r is not None and is_valid(r):
            return r

    # Greedy fallback
    def greedy():
        selected = [0]
        used = set()
        for target in range(1, 31):
            if target in used:
                continue
            best_x = None
            best_cover = -1
            for x in range(1, 5000):
                if x in selected:
                    continue
                conflict = False
                new_diffs = []
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used:
                        conflict = True
                        break
                    new_diffs.append(d)
                if conflict:
                    continue
                cover = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used)
                if target in new_diffs:
                    best_x = x
                    break
                elif cover > best_cover:
                    best_x = x
                    best_cover = cover
            if best_x is not None:
                selected.append(best_x)
                for s in selected[:-1]:
                    d = best_x - s
                    if d > 0:
                        used.add(d)
        if is_valid(selected):
            return sorted(selected)
        return None

    result = greedy()
    if result is not None:
        return result

    r = solve(1000, deadline - 1)
    if r is not None and is_valid(r):
        return r

    return [0, 1, 4, 10, 12, 28, 32, 45, 63]

# EVOLVE-BLOCK-END
