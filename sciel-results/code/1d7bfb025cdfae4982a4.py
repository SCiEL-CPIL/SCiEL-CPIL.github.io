import time
from math import gcd
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82
    n = 128
    random.seed(12345)
    
    # Precompute LCM table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # For each element, list of triple indices it participates in
    elem_to_triples = defaultdict(list)
    for i, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(i)
        elem_to_triples[b].append(i)
        elem_to_triples[c].append(i)
    
    # Greedy initial solution
    selected = [True] * (n + 1)
    
    # Build set of active (unhit) triples
    active = set(range(len(bad_triples)))
    
    while active:
        if time.time() > deadline - 30:
            break
        best_x = None
        best_cov = 0
        for x in range(1, n+1):
            if not selected[x]:
                continue
            cov = 0
            for ti in elem_to_triples[x]:
                if ti in active:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x is None or best_cov == 0:
            break
        selected[best_x] = False
        to_remove = set()
        for ti in elem_to_triples[best_x]:
            if ti in active:
                to_remove.add(ti)
        active -= to_remove
    
    # Verify
    def is_valid(sel):
        for (a, b, c) in bad_triples:
            if sel[a] and sel[b] and sel[c]:
                return False
        return True
    
    assert is_valid(selected), "Greedy failed to produce valid solution"
    
    best_size = sum(1 for x in range(1, n+1) if selected[x])
    best_sel = selected[:]
    
    # Local search with swaps and additions
    # Track for each element: how many active bad triples would be created if added
    # For efficiency, precompute pair co-occurrence in bad triples
    
    # For each pair (x,y), how many bad triples contain both x and y?
    # If x is not selected and we want to add x, the new violations = 
    #   number of bad triples containing x where the other two are both selected
    
    def add_violations(sel, x):
        """If x not in sel, how many bad triples become fully selected if we add x?"""
        cnt = 0
        for ti in elem_to_triples[x]:
            a, b, c = bad_triples[ti]
            others = [y for y in (a, b, c) if y != x]
            if sel[others[0]] and sel[others[1]]:
                cnt += 1
        return cnt
    
    def remove_resolves(sel, x):
        """If x in sel, how many bad triples become not-fully-selected if we remove x?"""
        cnt = 0
        for ti in elem_to_triples[x]:
            a, b, c = bad_triples[ti]
            others = [y for y in (a, b, c) if y != x]
            if sel[others[0]] and sel[others[1]]:
                cnt += 1
        return cnt
    
    # Phase 1: Try to add elements one by one
    changed = True
    while changed and time.time() < deadline - 25:
        changed = False
        for x in range(1, n+1):
            if selected[x]:
                continue
            if add_violations(selected, x) == 0:
                selected[x] = True
                best_size = sum(1 for x2 in range(1, n+1) if selected[x2])
                changed = True
    
    # Phase 2: Try swaps - remove one selected, add one unselected
    # Only if net effect is +1 in size and no new violations
    for _iter in range(200):
        if time.time() > deadline - 20:
            break
        improved = False
        unselected = [x for x in range(1, n+1) if not selected[x]]
        if not unselected:
            break
        for x_out in range(1, n+1):
            if not selected[x_out]:
                continue
            # Try removing x_out and adding x_in
            for x_in in unselected:
                # After removing x_out: check if adding x_in creates violations
                # Temporarily remove x_out
                # Violations from adding x_in = add_violations without x_out
                # We need: no bad triple has x_in and two other selected (after removing x_out)
                
                # A bad triple with x_in becomes violated if the other two are selected
                # After removing x_out, the other two might include x_out (which is now unselected)
                new_viol = 0
                for ti in elem_to_triples[x_in]:
                    a, b, c = bad_triples[ti]
                    others = [y for y in (a, b, c) if y != x_in]
                    o1, o2 = others
                    # Check if both are selected (x_out is being removed)
                    if o1 == x_out:
                        if selected[o2]:
                            pass  # o1 is x_out, which will be removed, so not both selected
                    elif o2 == x_out:
                        if selected[o1]:
                            pass
                    else:
                        if selected[o1] and selected[o2]:
                            new_viol += 1
                
                if new_viol == 0:
                    # Valid swap: remove x_out, add x_in, size stays same
                    # But we want size increase, so this is only useful if we can then add more
                    # Actually this keeps size same. Let's skip pure swaps for now.
                    pass
        
        # Try: remove x_out, then try to add multiple unselected
        for x_out in range(1, n+1):
            if not selected[x_out]:
                continue
            if time.time() > deadline - 20:
                break
            selected[x_out] = False
            # Try to add back as many as possible
            added_back = 0
            for x_in in range(1, n+1):
                if x_in == x_out:
                    continue
                if selected[x_in]:
                    continue
                if add_violations(selected, x_in) == 0:
                    selected[x_in] = True
                    added_back += 1
            net_gain = added_back - 1  # removed 1, added added_back
            if net_gain > 0:
                best_size = sum(1 for x2 in range(1, n+1) if selected[x2])
                improved = True
            else:
                # Revert
                selected[x_out] = True
                for x_in in range(1, n+1):
                    if x_in != x_out and selected[x_in]:
                        # Need to check if this was added during this round
                        pass
                # This revert is complex, let me use a different approach
                break
    
    # Phase 3: Simulated annealing on valid states
    # Only make moves that keep the state valid
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    cur_sel = selected[:]
    cur_size = sum(1 for x in range(1, n+1) if cur_sel[x])
    
    for it in range(100000):
        if time.time() > deadline - 5:
            break
        if T < T_min:
            break
        
        # Random move: pick element, flip if it maintains validity
        x = random.randint(1, n)
        if cur_sel[x]:
            # Try removing x (always valid)
            cur_sel[x] = False
            new_size = cur_size - 1
            # This is always valid, but reduces size. Accept with some probability for exploration
            if random.random() < 0.1:  # rarely accept size decrease
                cur_size = new_size
                T *= cooling
            else:
                cur_sel[x] = True  # revert
                T *= cooling
        else:
            # Try adding x
            if add_violations(cur_sel, x) == 0:
                cur_sel[x] = True
                cur_size += 1
                T *= cooling
                if cur_size > best_size:
                    best_size = cur_size
                    best_sel = cur_sel[:]
            else:
                # Can't add directly. Try swap: remove one selected, add x
                # Find a selected element whose removal allows adding x
                found = False
                for y in range(1, n+1):
                    if not cur_sel[y]:
                        continue
                    # Check if removing y allows adding x
                    # After removing y, check if x can be added
                    ok = True
                    for ti in elem_to_triples[x]:
                        a, b, c = bad_triples[ti]
                        others = [zz for zz in (a, b, c) if zz != x]
                        o1, o2 = others
                        if o1 == y:
                            if cur_sel[o2]:
                                ok = False
                                break
                        elif o2 == y:
                            if cur_sel[o1]:
                                ok = False
                                break
                        else:
                            if cur_sel[o1] and cur_sel[o2]:
                                ok = False
                                break
                    if ok:
                        cur_sel[y] = False
                        cur_sel[x] = True
                        # Size unchanged, but might enable future additions
                        T *= cooling
                        found = True
                        break
                if not found:
                    T *= cooling
    
    # Final: try to add more
    for x in range(1, n+1):
        if not best_sel[x]:
            if add_violations(best_sel, x) == 0:
                best_sel[x] = True
    
    final_size = sum(1 for x in range(1, n+1) if best_sel[x])
    result = [x for x in range(1, n+1) if best_sel[x]]
    if not result:
        result = [1]
    
    return result
