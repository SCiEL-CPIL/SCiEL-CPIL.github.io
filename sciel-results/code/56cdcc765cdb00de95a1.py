def construct():
    from fractions import Fraction
    import time
    import random
    
    start_time = time.time()
    deadline = 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    best_selected = None
    best_deficit = Fraction(1)
    
    def update(sel, tot):
        nonlocal best_selected, best_deficit
        if tot < 1:
            deficit = 1 - tot
            if deficit < best_deficit:
                best_deficit = deficit
                best_selected = tuple(sorted(sel))
    
    def local_search(sel, tot):
        nonlocal best_selected, best_deficit
        sel = frozenset(sel)
        tot = tot
        while True:
            if time.time() > deadline:
                return
            improved = False
            
            # Try single additions
            for i in range(n):
                if time.time() > deadline:
                    return
                if i not in sel:
                    new_tot = tot + recip[i]
                    if new_tot < 1:
                        new_deficit = 1 - new_tot
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_selected = tuple(sorted(sel | {i}))
                            sel = sel | {i}
                            tot = new_tot
                            improved = True
                            break  # restart scan after improvement
            
            if improved:
                continue
            
            # Try single removals (only if current sum is too close to 1, but we need < 1)
            # Actually removals make deficit worse, so skip unless needed for swaps
            
            # Try single swaps (remove one, add one)
            for i in list(sel):
                if time.time() > deadline:
                    return
                for j in range(n):
                    if j not in sel:
                        new_tot = tot - recip[i] + recip[j]
                        if new_tot < 1:
                            new_deficit = 1 - new_tot
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                new_sel = set(sel)
                                new_sel.remove(i)
                                new_sel.add(j)
                                best_selected = tuple(sorted(new_sel))
                                sel = frozenset(new_sel)
                                tot = new_tot
                                improved = True
                                break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try double swaps (remove two, add two) - more expensive but powerful
            sel_list = list(sel)
            for a_idx in range(len(sel_list)):
                if time.time() > deadline:
                    return
                for b_idx in range(a_idx + 1, len(sel_list)):
                    a = sel_list[a_idx]
                    b = sel_list[b_idx]
                    base_tot = tot - recip[a] - recip[b]
                    for j in range(n):
                        if time.time() > deadline:
                            return
                        if j not in sel:
                            for k in range(j + 1, n):
                                if k not in sel:
                                    new_tot = base_tot + recip[j] + recip[k]
                                    if new_tot < 1:
                                        new_deficit = 1 - new_tot
                                        if new_deficit < best_deficit:
                                            best_deficit = new_deficit
                                            new_sel = set(sel)
                                            new_sel.remove(a)
                                            new_sel.remove(b)
                                            new_sel.add(j)
                                            new_sel.add(k)
                                            best_selected = tuple(sorted(new_sel))
                                            sel = frozenset(new_sel)
                                            tot = new_tot
                                            improved = True
                                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Greedy initialization: add smallest denominators first that fit
    sel = set()
    tot = Fraction(0)
    for i in range(n):
        if tot + recip[i] < 1:
            sel.add(i)
            tot += recip[i]
    update(sel, tot)
    local_search(sel, tot)
    
    # Random restarts with different strategies
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        strategy = restart % 3
        
        if strategy == 0:
            # Random order greedy
            sel = set()
            tot = Fraction(0)
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if tot + recip[i] < 1:
                    sel.add(i)
                    tot += recip[i]
        elif strategy == 1:
            # Start with large denominators, add small ones
            sel = set()
            tot = Fraction(0)
            # Add some large denominators first
            large_indices = [i for i in range(n) if denominators[i] >= 20]
            random.shuffle(large_indices)
            for i in large_indices[:5]:
                if tot + recip[i] < 1:
                    sel.add(i)
                    tot += recip[i]
            # Then add small ones
            for i in range(n):
                if i not in sel and tot + recip[i] < 1:
                    sel.add(i)
                    tot += recip[i]
        else:
            # Random subset, then greedily improve
            sel = set()
            tot = Fraction(0)
            for i in range(n):
                if random.random() < 0.5 and tot + recip[i] < 1:
                    sel.add(i)
                    tot += recip[i]
            # Greedy add
            for i in range(n):
                if i not in sel and tot + recip[i] < 1:
                    sel.add(i)
                    tot += recip[i]
        
        if not sel:
            sel = {0}
            tot = recip[0]
        
        update(sel, tot)
        local_search(sel, tot)
    
    if best_selected is None:
        return [0]
    
    return list(best_selected)
