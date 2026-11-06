import time
from math import lcm
from fractions import Fraction

def construct():
    deadline = time.time() + 55
    N = 255
    nums = list(range(2, 257))
    recip = [1.0 / nums[i] for i in range(N)]
    recip_frac = [Fraction(1, nums[i]) for i in range(N)]

    # conflict_mask[i] = int bitset of j (j != i) with lcm(nums[i],nums[j]) <= 256
    conflict_mask = [0] * N
    conflict_count = [0] * N
    for i in range(N):
        ni = nums[i]
        for j in range(i + 1, N):
            if lcm(ni, nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                conflict_count[i] += 1
                conflict_count[j] += 1

    # order by descending reciprocal
    order = list(range(N))
    order.sort(key=lambda i: -recip[i])
    
    # Also create an order that prioritizes high-reciprocal, low-conflict elements
    order2 = list(range(N))
    order2.sort(key=lambda i: (-recip[i] * 1000 - conflict_count[i]))

    # Precompute compatibility masks for faster checks
    # compat_mask[i] = bitset of j such that i and j are compatible (lcm > 256)
    compat_mask = [0] * N
    for i in range(N):
        all_bits = (1 << N) - 1
        compat_mask[i] = all_bits & ~(conflict_mask[i]) & ~(1 << i)

    def recip_sum(sel):
        return sum(recip[i] for i in sel)

    def recip_sum_frac(sel):
        s = Fraction(0)
        for i in sel:
            s += recip_frac[i]
        return s

    def greedy(order_list):
        sel = []
        for i in order_list:
            ok = True
            for s in sel:
                if (conflict_mask[s] >> i) & 1:
                    ok = False
                    break
            if ok:
                sel.append(i)
        return sel

    def local_search(sel, order_list, max_iter=200):
        nonlocal best_val, best_val_frac, best_sel
        sel = list(sel)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            cur = recip_sum(sel)
            cur_set = set(sel)
            
            # Try adding any compatible element
            for i in order_list:
                if i in cur_set:
                    continue
                if all(not ((conflict_mask[s] >> i) & 1) for s in sel):
                    sel.append(i)
                    cur_set.add(i)
                    improved = True
                    break
            
            # 1-for-1 swap
            if not improved:
                for idx, s in enumerate(sel):
                    for i in order_list:
                        if i in cur_set:
                            continue
                        if all(not ((conflict_mask[t] >> i) & 1) for t in sel if t != s):
                            if recip[i] > recip[s]:
                                sel[idx] = i
                                cur_set.discard(s)
                                cur_set.add(i)
                                improved = True
                                break
                    if improved:
                        break
            
            # 2-for-1 swap: remove two, add one
            if not improved:
                n = len(sel)
                for a in range(n):
                    for b in range(a + 1, n):
                        rem_sum = recip[sel[a]] + recip[sel[b]]
                        for i in order_list:
                            if i in cur_set:
                                continue
                            if all(not ((conflict_mask[t] >> i) & 1) for t in sel if t not in (sel[a], sel[b])):
                                if recip[i] > rem_sum:
                                    new_sel = [t for t in sel if t not in (sel[a], sel[b])]
                                    new_sel.append(i)
                                    sel = new_sel
                                    cur_set = set(sel)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # 1-for-2 swap: remove one, add two
            if not improved:
                n = len(sel)
                for a in range(n):
                    removed = sel[a]
                    rem_sum = recip[removed]
                    for i in order_list:
                        if i in cur_set:
                            continue
                        if all(not ((conflict_mask[t] >> i) & 1) for t in sel if t != removed):
                            for j in order_list:
                                if j in cur_set or j == i:
                                    continue
                                if all(not ((conflict_mask[t] >> j) & 1) for t in sel if t != removed):
                                    if not ((conflict_mask[i] >> j) & 1):
                                        if recip[i] + recip[j] > rem_sum:
                                            new_sel = [t for t in sel if t != removed]
                                            new_sel.extend([i, j])
                                            sel = new_sel
                                            cur_set = set(sel)
                                            improved = True
                                            break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        val = recip_sum(sel)
        val_frac = recip_sum_frac(sel)
        if val_frac > best_val_frac:
            best_val = val
            best_val_frac = val_frac
            best_sel = list(sel)
        return sel, val

    # Initial solutions from different orders
    best_val = 0.0
    best_val_frac = Fraction(0)
    best_sel = []

    # Try greedy with different orders
    for ord_list in [order, order2]:
        sel = greedy(ord_list)
        sel, val = local_search(sel, ord_list)

    full_mask = (1 << N) - 1

    # ---- Improved branch and bound with greedy coloring bound ----
    def greedy_color_bound(avail):
        """
        Compute an upper bound on the weight of a maximum weight independent set
        in the subgraph induced by avail, using a greedy coloring approach.
        This is tighter than just summing all weights.
        """
        if avail == 0:
            return 0.0
        
        # We'll use a simple greedy approach: repeatedly pick the highest-weight
        # vertex, add it to the bound, and remove it and its neighbors
        # This gives an upper bound because we're not enforcing independence properly
        # but it's much tighter than the sum of all weights
        
        # Better approach: use the fact that in a clique, we can pick at most one
        # from each "color class" in a proper coloring of the conflict graph
        # For now, use a simple but effective bound:
        # Process vertices in order of decreasing weight, and for each, 
        # check if it's still available (not conflicting with previously chosen)
        
        bound = 0.0
        remaining = avail
        # Process in order of decreasing reciprocal
        for i in order:
            if (remaining >> i) & 1:
                bound += recip[i]
                # Remove i and all its neighbors from remaining
                remaining &= ~conflict_mask[i] & ~(1 << i)
                if remaining == 0:
                    break
        return bound

    def greedy_color_bound_frac(avail):
        """Exact fraction version of greedy_color_bound."""
        if avail == 0:
            return Fraction(0)
        
        bound = Fraction(0)
        remaining = avail
        for i in order:
            if (remaining >> i) & 1:
                bound += recip_frac[i]
                remaining &= ~conflict_mask[i] & ~(1 << i)
                if remaining == 0:
                    break
        return bound

    nodes = [0]

    def bnb(sel, avail, cur, cur_frac, order_list):
        nonlocal best_val, best_val_frac, best_sel
        if time.time() > deadline:
            return
        
        # Use greedy coloring bound for tighter pruning
        upper_float = cur + greedy_color_bound(avail)
        if upper_float <= best_val - 1e-9:
            return
        # If close, use exact fraction bound
        if upper_float <= best_val + 1e-6:
            upper_frac = cur_frac + greedy_color_bound_frac(avail)
            if upper_frac <= best_val_frac:
                return
        
        # Find next candidate in order that is in avail and compatible with sel
        next_i = -1
        for i in order_list:
            if (avail >> i) & 1:
                ok = True
                for t in sel:
                    if not ((compat_mask[t] >> i) & 1):
                        ok = False
                        break
                if ok:
                    next_i = i
                    break
        
        if next_i == -1:
            if cur_frac > best_val_frac:
                best_val = cur
                best_val_frac = cur_frac
                best_sel = list(sel)
            return
        
        i = next_i
        nodes[0] += 1
        
        # Take branch
        sel.append(i)
        new_avail = avail & compat_mask[i] & ~(1 << i)
        bnb(sel, new_avail, cur + recip[i], cur_frac + recip_frac[i], order_list)
        sel.pop()
        
        # Skip branch
        bnb(sel, avail & ~(1 << i), cur, cur_frac, order_list)

    # Run B&B with primary order
    bnb([], full_mask, 0.0, Fraction(0), order)
    
    # If time remains, try B&B with secondary order from current best
    if time.time() < deadline - 5:
        bnb2_sel = list(best_sel)
        bnb2_avail = full_mask
        for s in bnb2_sel:
            bnb2_avail &= compat_mask[s]
            bnb2_avail &= ~(1 << s)
        bnb2_cur = best_val
        bnb2_cur_frac = best_val_frac
        bnb(bnb2_sel, bnb2_avail, bnb2_cur, bnb2_cur_frac, order2)

    # Final local search polish on best solution
    if best_sel:
        best_sel, _ = local_search(best_sel, order, max_iter=500)
        best_sel, _ = local_search(best_sel, order2, max_iter=500)

    # ensure nonempty
    if not best_sel:
        best_sel = [order[0]]
    return [nums[i] for i in best_sel]
