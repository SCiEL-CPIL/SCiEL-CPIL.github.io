def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 55
    
    n = 47
    denoms = list(range(2, 49))
    fracs = [Fraction(1, d) for d in denoms]
    
    # Greedy initialization
    greedy_selected = set()
    greedy_sum = Fraction(0)
    for i in range(n):
        if greedy_sum + fracs[i] < 1:
            greedy_selected.add(i)
            greedy_sum += fracs[i]
    
    best_sum = greedy_sum
    best_indices = list(greedy_selected)
    
    # Local search: try adding/removing single elements
    def local_search(selected_set, current_sum):
        nonlocal best_sum, best_indices
        improved = True
        while improved:
            improved = False
            # Try adding
            for i in range(n):
                if i not in selected_set:
                    new_sum = current_sum + fracs[i]
                    if new_sum < 1 and new_sum > best_sum:
                        selected_set.add(i)
                        current_sum = new_sum
                        best_sum = new_sum
                        best_indices = list(selected_set)
                        improved = True
                        break
            if improved:
                continue
            # Try removing
            for i in list(selected_set):
                new_sum = current_sum - fracs[i]
                if new_sum < 1 and new_sum > best_sum:
                    selected_set.remove(i)
                    current_sum = new_sum
                    best_sum = new_sum
                    best_indices = list(selected_set)
                    improved = True
                    break
        return selected_set, current_sum
    
    selected_set, current_sum = local_search(greedy_selected, greedy_sum)
    
    # Branch and bound for further improvement
    order = sorted(range(n), key=lambda i: -fracs[i])
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + fracs[order[i]]
    
    def bnb(pos, current_sum, selected):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        if current_sum >= 1:
            return
        
        if len(selected) > 0 and current_sum > best_sum:
            best_sum = current_sum
            best_indices = list(selected)
        
        if pos == n:
            return
        
        if current_sum + suffix[pos] <= best_sum:
            return
        
        d = order[pos]
        new_sum = current_sum + fracs[d]
        
        if new_sum < 1:
            selected.append(d)
            bnb(pos + 1, new_sum, selected)
            selected.pop()
        
        bnb(pos + 1, current_sum, selected)
    
    bnb(0, Fraction(0), [])
    
    return best_indices
