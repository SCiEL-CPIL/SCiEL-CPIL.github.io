"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 50  # safety margin before 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [Fraction(1, d) for d in denominators]
    
    # Greedy: pick largest fraction that keeps sum < 1
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + vals[i] < 1:
            selected.add(i)
            total += vals[i]
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1 - total
    
    # Local search: try swaps and additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a single unselected element
        for j in range(n):
            if j not in selected:
                new_total = total + vals[j]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        selected.add(j)
                        total = new_total
                        best_selected = set(selected)
                        best_total = total
                        best_deficit = new_deficit
                        improved = True
        
        # Try swaps: remove one selected, add one unselected
        for i in list(selected):
            for j in range(n):
                if j in selected:
                    continue
                new_total = total - vals[i] + vals[j]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        selected.remove(i)
                        selected.add(j)
                        total = new_total
                        best_selected = set(selected)
                        best_total = total
                        best_deficit = new_deficit
                        improved = True
                        break
                # Also try removing i and adding two unselected j,k
                # This is more expensive; do it in a second pass if needed
        
        # Try remove-one and add-two swaps
        if not improved:
            for i in list(selected):
                base_total = total - vals[i]
                # Try adding two unselected elements
                for j in range(n):
                    if j in selected or j == i:
                        continue
                    t1 = base_total + vals[j]
                    if t1 >= 1:
                        continue
                    for k in range(n):
                        if k in selected or k == i or k == j:
                            continue
                        t2 = t1 + vals[k]
                        if t2 < 1:
                            new_deficit = 1 - t2
                            if new_deficit < best_deficit:
                                selected.remove(i)
                                selected.add(j)
                                selected.add(k)
                                total = t2
                                best_selected = set(selected)
                                best_total = total
                                best_deficit = new_deficit
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
