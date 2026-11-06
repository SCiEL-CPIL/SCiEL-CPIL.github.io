import time
import random

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def get_mask(sel):
        """Compute bitmask of all sums a+b for a,b in sel, a<=b."""
        m = 0
        for i in range(len(sel)):
            a = sel[i]
            m |= 1 << (2 * a)
            for j in range(i + 1, len(sel)):
                b = sel[j]
                m |= 1 << (a + b)
        return m

    def can_add(x, sel, sum_mask):
        """Check if x can be added to sel (Sidon property maintained)."""
        if (sum_mask >> (2 * x)) & 1:
            return False
        for a in sel:
            if (sum_mask >> (x + a)) & 1:
                return False
        return True

    def add_element(x, sel, sel_set, sum_mask):
        """Add x to sel, update sel_set and sum_mask."""
        sel.append(x)
        sel_set.add(x)
        for a in sel[:-1]:
            sum_mask |= 1 << (x + a)
        sum_mask |= 1 << (2 * x)
        return sel, sel_set, sum_mask

    def is_maximal(sel, sum_mask, sel_set):
        """Check if sel is inclusion-maximal."""
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            # Check if y can be added
            if (sum_mask >> (2 * y)) & 1:
                continue
            conflict = False
            for a in sel:
                if (sum_mask >> (y + a)) & 1:
                    conflict = True
                    break
            if not conflict:
                return False
        return True

    def saturate(sel, sel_set, sum_mask):
        """Add all possible elements to make the set maximal.
        Returns None if the set cannot be extended to a maximal Sidon set 
        (i.e., if we get stuck with a non-maximal set that can't be extended further 
        without violating Sidon, but wait: any Sidon set can be extended to a maximal 
        one. The only failure is if we run out of time or if we made a choice that 
        led to a dead end? No, Sidon sets are always extendable. 
        Actually, saturate just adds everything that CAN be added. 
        If it's not maximal, it means there's some y that can be added. 
        So saturate loops until no more can be added. Then it IS maximal.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if can_add(x, sel, sum_mask):
                    sel, sel_set, sum_mask = add_element(x, sel, sel_set, sum_mask)
                    changed = True
        return sel, sel_set, sum_mask

    def backtrack_search(core_size, seed):
        """
        Try to find a maximal Sidon set of size close to core_size.
        Strategy:
        1. Recursively pick elements to form a 'core' of size core_size.
        2. Heuristic: Pick element that blocks the most remaining candidates.
        3. Once core is formed, saturate it to get the full maximal set.
        4. Check if the resulting set is small.
        """
        rng = random.Random(seed)
        best_set = None
        best_size = float('inf')
        
        # Order candidates to try "blocking" ones first
        # We don't know the best order a priori, so we shuffle or use a heuristic.
        # Let's use a simple heuristic: prefer elements that create many conflicts.
        # But since we are building the core, we want to pick elements that are "hard" to place later?
        # Actually, for minimal maximal sets, we want to pick elements that block many others.
        
        # We will do a limited DFS.
        # To keep it fast, we limit the branching factor and depth.
        
        def dfs(sel, sel_set, sum_mask, depth):
            nonlocal best_set, best_size
            
            if time.time() > deadline - 5:
                return
            
            if depth == core_size:
                # Saturate
                s_sel, s_set, s_mask = list(sel), set(sel_set), sum_mask
                s_sel, s_set, s_mask = saturate(s_sel, s_set, s_mask)
                if s_sel is None:
                    return
                if len(s_sel) < best_size:
                    best_size = len(s_sel)
                    best_set = list(s_sel)
                return
            
            # Candidate selection
            # We want to try adding an element x.
            # To find SMALL maximal sets, we should prefer x that blocks many y's.
            # However, if we pick too many blockers, the core might become large.
            # Wait, the core size is fixed. We are picking core_size elements.
            # The resulting maximal set size depends on how many OTHER elements can be added.
            # To minimize the final size, we want the core to block as many as possible.
            
            # Let's generate candidates: all x not in sel that can be added.
            # Score them by "blocking power" on the remaining universe.
            
            candidates = []
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if not can_add(x, sel, sum_mask):
                    continue
                
                # Estimate blocking power: how many y (not in sel, y!=x) are blocked by x?
                # y is blocked if (x+y) in sum_mask OR 2y in sum_mask (after adding x? No, 2y is blocked if 2y is already in sum_mask, or if y+x is in sum_mask)
                # Actually, if we add x, y is blocked if:
                # 1. 2y is in new_sum_mask. new_sum_mask has 2x and x+a for a in sel.
                #    So if 2y == 2x (y=x, excluded) or 2y == x+a for some a.
                # 2. y+x is in new_sum_mask. i.e., y+x == 2b or y+x == b+c for b,c in sel.
                
                block_count = 0
                # We can approximate by checking current sum_mask
                # If we add x, it adds sums: 2x, and x+a for a in sel.
                # A candidate y is blocked if:
                # - 2y is in (current_sum_mask | {2x} | {x+a for a in sel})
                # - y+x is in (current_sum_mask | {2x} | {x+a for a in sel})
                
                # For speed, we only check a sample or use a simpler metric.
                # Simple metric: number of a in sel such that x+a is a sum that blocks many things?
                # Let's just count how many y in [1, N] are blocked by the NEW sums.
                # This is O(N * len(sel)) per candidate, which is 256 * 10 = 2560 ops.
                # With 256 candidates, that's 655k ops per node. Too slow for deep search.
                
                # Approximation: Count how many y have (y+x) in current_sum_mask OR 2y in current_sum_mask.
                # This ignores the new sums from x, but gives a relative ordering.
                # Better: Count how many y have (y+x) in current_sum_mask.
                # Because if y+x is already a sum, y is blocked immediately upon adding x.
                # Also if 2y is already a sum, y is blocked regardless of x (unless x=y which is excluded).
                
                # Let's use: score = sum(1 for y in range(1, N+1) if y!=x and y not in sel_set and ((sum_mask >> (y+x)) & 1 or (sum_mask >> (2*y)) & 1))
                # This is O(N) per candidate. 256*256 = 65k ops per node.
                # Depth 10. 65k * 10 * branching.
                # Branching factor: we take top K candidates. K=5?
                # 65k * 5 * 10 = 3.2M ops per path.
                # Number of paths: K^depth? No, we prune.
                # This might be feasible if we limit depth to 5-6.
                
                # Let's try a cheaper score:
                # How many elements in sel have a sum with x that is "popular"?
                # Or simply: prefer x that is "central" (around N/2)?
                # Let's stick to the O(N) score but limit candidates.
                
                score = 0
                # Check y in a subset? No, let's just do it for all y but break early if score is high?
                # Or just use: number of a in sel such that (x+a) is in sum_mask? No, that's 0 by definition of can_add.
                
                # Let's use: number of y such that (x+y) is in sum_mask.
                # This means adding x will immediately block y.
                for y in range(1, N + 1):
                    if y == x or y in sel_set:
                        continue
                    if (sum_mask >> (x + y)) & 1:
                        score += 1
                    # Also if 2y is in sum_mask, y is blocked anyway, so x doesn't need to block it.
                    # We want x to block y's that are NOT already blocked.
                    # So: y is newly blocked if (x+y) in sum_mask OR 2y in sum_mask (but 2y in sum_mask means y was already blocked).
                    # So we count y where (x+y) in sum_mask AND 2y NOT in sum_mask?
                    # Actually, if 2y is in sum_mask, y is already blocked.
                    # If 2y is NOT in sum_mask, but x+y IS in sum_mask, then x blocks y.
                    # So score += 1 if (sum_mask >> (x+y)) & 1 and not (sum_mask >> (2*y)) & 1.
                
                candidates.append((score, x))
            
            if not candidates:
                return
            
            # Sort by score descending
            candidates.sort(reverse=True)
            
            # Take top K
            K = 5
            for _, x in candidates[:K]:
                new_sel, new_set, new_mask = add_element(x, list(sel), set(sel_set), sum_mask)
                dfs(new_sel, new_set, new_mask, depth + 1)
                if time.time() > deadline - 5:
                    return

        dfs([], set(), 0, 0)
        return best_set

    best_overall = None
    best_size_overall = float('inf')
    
    # Try different core sizes
    # Smaller core -> potentially smaller maximal set?
    # A maximal Sidon set must have size at least sqrt(N) ~ 16.
    # Let's try core sizes from 5 to 15.
    
    seeds = range(20)
    
    for core_size in [5, 6, 7, 8, 9, 10, 12, 15]:
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            res = backtrack_search(core_size, seed + core_size * 100)
            if res:
                if len(res) < best_size_overall:
                    best_size_overall = len(res)
                    best_overall = res
    
    # Fallback: Greedy
    if best_overall is None:
        sel = []
        sel_set = set()
        sum_mask = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        for x in remaining:
            if can_add(x, sel, sum_mask):
                sel, sel_set, sum_mask = add_element(x, sel, sel_set, sum_mask)
        sel, sel_set, sum_mask = saturate(sel, sel_set, sum_mask)
        best_overall = sel
        best_size_overall = len(sel)
        
    # Final local search to reduce
    current = list(best_overall)
    current_size = len(current)
    
    while time.time() < deadline - 2:
        improved = False
        current_set = set(current)
        current_mask = get_mask(current)
        elems = list(current)
        
        for i in range(len(elems)):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, len(elems)):
                if time.time() > deadline - 2:
                    break
                x1 = elems[i]
                x2 = elems[j]
                
                rem_sel = [e for e in current if e != x1 and e != x2]
                rem_set = set(rem_sel)
                rem_mask = get_mask(rem_sel)
                
                for y in range(1, N + 1):
                    if y in rem_set:
                        continue
                    if not can_add(y, rem_sel, rem_mask):
                        continue
                    
                    new_sel = rem_sel + [y]
                    new_set = rem_set | {y}
                    new_mask = rem_mask
                    for a in rem_sel:
                        new_mask |= 1 << (y + a)
                    new_mask |= 1 << (2 * y)
                    
                    if is_maximal(new_sel, new_mask, new_set):
                        current = new_sel
                        current_size = len(current)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break

    return current
