def construct():
    import time
    from math import lcm
    from fractions import Fraction
    
    start = time.time()
    deadline = start + 80
    
    numbers = list(range(2, 257))
    size = len(numbers)  # 255
    
    # Incompatibility sets
    incompat = [set() for _ in range(size)]
    for i in range(size):
        for j in range(i+1, size):
            if lcm(numbers[i], numbers[j]) <= 256:
                incompat[i].add(j)
                incompat[j].add(i)
    
    def can_add(i, selected_set):
        return not (incompat[i] & selected_set)
    
    def set_score(selected):
        return sum(Fraction(1, numbers[i]) for i in selected)
    
    # Greedy
    def greedy(order):
        sel = []
        sel_set = set()
        for idx in order:
            if can_add(idx, sel_set):
                sel.append(idx)
                sel_set.add(idx)
        return sel
    
    best_set = greedy(list(range(size)))
    best_score = set_score(best_set)
    
    # Try descending
    s2 = greedy(list(range(size-1, -1, -1)))
    sc2 = set_score(s2)
    if sc2 > best_score:
        best_set, best_score = s2, sc2
    
    # Local search
    best_set_set = set(best_set)
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        # Try adding
        for i in range(size):
            if i not in best_set_set and can_add(i, best_set_set):
                new_set = best_set + [i]
                sc = set_score(new_set)
                if sc > best_score:
                    best_set = new_set
                    best_set_set = set(best_set)
                    best_score = sc
                    improved = True
                    break
        if improved:
            continue
        # Try swap
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            trial_set = best_set_set - {best_set[i]}
            for j in range(size):
                if j not in trial_set and can_add(j, trial_set):
                    new_set = [x for x in best_set if x != best_set[i]] + [j]
                    sc = set_score(new_set)
                    if sc > best_score:
                        best_set = new_set
                        best_set_set = set(best_set)
                        best_score = sc
                        improved = True
                        break
            if improved:
                break
    
    return [numbers[i] for i in best_set]
