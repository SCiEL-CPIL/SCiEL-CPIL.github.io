import time
import random
import sys

N = 256

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin

    # Sum mask: bit i set means sum i is present (1-indexed sums from 2 to 512)
    # We use a Python int as a bitmask for sums

    def make_mask(sums_list):
        m = 0
        for s in sums_list:
            m |= (1 << s)
        return m

    def sums_of(sel):
        """Return sorted list of all a+b for a<=b in sel."""
        s = list(sel)
        s.sort()
        result = []
        for i in range(len(s)):
            a = s[i]
            result.append(2 * a)
            for j in range(i + 1, len(s)):
                result.append(a + s[j])
        return result

    def get_sum_mask(sel):
        m = 0
        s = list(sel)
        s.sort()
        for i in range(len(s)):
            a = s[i]
            m |= (1 << (2 * a))
            for j in range(i + 1, len(s)):
                m |= (1 << (a + s[j]))
        return m

    def is_sidon(sel):
        """Check if sel is a valid Sidon set."""
        s = list(sel)
        s.sort()
        seen = set()
        for i in range(len(s)):
            a = s[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for j in range(i + 1, len(s)):
                b = s[j]
                ab = a + b
                if ab in seen:
                    return False
                seen.add(ab)
        return True

    def is_maximal(sel, sum_mask):
        """Check if sel is inclusion-maximal Sidon set."""
        sel_set = set(sel)
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            # Check if y can be added
            if (sum_mask >> (2 * y)) & 1:
                continue  # 2y already blocked
            conflict = False
            for a in sel:
                if (sum_mask >> (y + a)) & 1:
                    conflict = True
                    break
            if not conflict:
                return False  # y can be added, so not maximal
        return True

    def count_blockable(sel, sum_mask):
        """Count how many elements not in sel are blocked by sel."""
        sel_set = set(sel)
        blocked = 0
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if (sum_mask >> (2 * y)) & 1:
                blocked += 1
                continue
            for a in sel:
                if (sum_mask >> (y + a)) & 1:
                    blocked += 1
                    break
        return blocked

    def blocking_coverage(x, sel, sum_mask):
        """If we add x to sel, how many additional elements get blocked?"""
        sel_set = set(sel)
        if x in sel_set:
            return 0
        # Check if x can be added
        if (sum_mask >> (2 * x)) & 1:
            return 0
        for a in sel:
            if (sum_mask >> (x + a)) & 1:
                return 0
        
        # New sums when adding x
        new_sums = set()
        new_sums.add(2 * x)
        for a in sel:
            new_sums.add(x + a)
        
        # Count elements that become newly blocked
        newly_blocked = 0
        for y in range(1, N + 1):
            if y in sel_set or y == x:
                continue
            # Was y blocked before?
            was_blocked = False
            if (sum_mask >> (2 * y)) & 1:
                was_blocked = True
            else:
                for a in sel:
                    if (sum_mask >> (y + a)) & 1:
                        was_blocked = True
                        break
            if was_blocked:
                continue
            # Is y blocked after adding x?
            if (2 * y) in new_sums:
                newly_blocked += 1
                continue
            for a in sel:
                if (y + a) in new_sums:
                    newly_blocked += 1
                    break
        return newly_blocked

    def greedy_maximal(seed, time_budget=3.0):
        """Greedy construction: pick elements that block the most."""
        rng = random.Random(seed)
        t0 = time.time()
        
        sel = []
        sel_set = set()
        sum_mask = 0
        
        # Precompute all pairs
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        
        while remaining and time.time() < t0 + time_budget:
            best_x = -1
            best_score = -1
            best_tie = -1
            
            # Sample candidates for speed
            if len(remaining) > 30:
                candidates = rng.sample(remaining, 30)
            else:
                candidates = remaining
            
            for x in candidates:
                if x in sel_set:
                    continue
                # Check if x can be added
                if (sum_mask >> (2 * x)) & 1:
                    continue
                valid = True
                for a in sel:
                    if (sum_mask >> (x + a)) & 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: how many new elements get blocked
                score = blocking_coverage(x, sel, sum_mask)
                
                if score > best_score or (score == best_score and x < best_tie):
                    best_score = score
                    best_x = x
                    best_tie = x
            
            if best_x == -1:
                # Fallback: pick first valid
                for x in remaining:
                    if x in sel_set:
                        continue
                    if (sum_mask >> (2 * x)) & 1:
                        continue
                    valid = True
                    for a in sel:
                        if (sum_mask >> (x + a)) & 1:
                            valid = False
                            break
                    if valid:
                        best_x = x
                        break
            
            if best_x == -1:
                break
            
            # Add best_x
            sel.append(best_x)
            sel_set.add(best_x)
            sum_mask |= (1 << (2 * best_x))
            for a in sel[:-1]:
                sum_mask |= (1 << (best_x + a))
            remaining.remove(best_x)
        
        return sel

    def try_reduce_2to1(sel, time_budget=2.0):
        """Try to remove 2 elements and add 1 to get a smaller maximal set."""
        t0 = time.time()
        current = list(sel)
        current_mask = get_sum_mask(current)
        
        while time.time() < t0 + time_budget:
            improved = False
            elems = list(current)
            
            for i in range(len(elems)):
                if time.time() > t0 + time_budget:
                    break
                for j in range(i + 1, len(elems)):
                    if time.time() > t0 + time_budget:
                        break
                    
                    x1, x2 = elems[i], elems[j]
                    rem = [e for e in current if e != x1 and e != x2]
                    rem_set = set(rem)
                    rem_mask = get_sum_mask(rem)
                    
                    # Try to find a single y that can be added and makes it maximal
                    found = False
                    for y in range(1, N + 1):
                        if y in rem_set:
                            continue
                        if (rem_mask >> (2 * y)) & 1:
                            continue
                        valid = True
                        for a in rem:
                            if (rem_mask >> (y + a)) & 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_sel = rem + [y]
                        new_set = rem_set | {y}
                        new_mask = rem_mask
                        new_mask |= (1 << (2 * y))
                        for a in rem:
                            new_mask |= (1 << (y + a))
                        
                        if is_maximal(new_sel, new_mask):
                            current = new_sel
                            current_mask = new_mask
                            improved = True
                            found = True
                            break
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current

    def try_reduce_1to0_or_1(sel, time_budget=1.5):
        """Try removing one element and possibly adding another."""
        t0 = time.time()
        current = list(sel)
        
        while time.time() < t0 + time_budget:
            improved = False
            elems = list(current)
            
            for idx in range(len(elems)):
                if time.time() > t0 + time_budget:
                    break
                
                x = elems[idx]
                rem = [e for e in current if e != x]
                rem_set = set(rem)
                rem_mask = get_sum_mask(rem)
                
                # Check if rem is already maximal (removing x suffices)
                if is_maximal(rem, rem_mask):
                    current = rem
                    improved = True
                    break
                
                # Try adding one element to rem to make it maximal
                for y in range(1, N + 1):
                    if y in rem_set:
                        continue
                    if (rem_mask >> (2 * y)) & 1:
                        continue
                    valid = True
                    for a in rem:
                        if (rem_mask >> (y + a)) & 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_sel = rem + [y]
                    new_mask = rem_mask
                    new_mask |= (1 << (2 * y))
                    for a in rem:
                        new_mask |= (1 << (y + a))
                    
                    if len(new_sel) < len(current) and is_maximal(new_sel, new_mask):
                        current = new_sel
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current

    # Main: multiple trials with greedy + reduction
    best_set = None
    best_size = float('inf')
    
    num_trials = 20
    for i in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Greedy build with different seeds
        sel = greedy_maximal(i, time_budget=1.5)
        
        if len(sel) >= best_size:
            continue
        
        # Reductions
        sel = try_reduce_2to1(sel, time_budget=1.0)
        sel = try_reduce_1to0_or_1(sel, time_budget=0.8)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    # Final reduction pass on best
    if best_set is not None and time.time() < deadline - 2:
        best_set = try_reduce_2to1(best_set, time_budget=1.0)
        best_set = try_reduce_1to0_or_1(best_set, time_budget=0.8)
    
    if best_set is None:
        best_set = greedy_maximal(0, time_budget=3.0)
    
    return best_set
