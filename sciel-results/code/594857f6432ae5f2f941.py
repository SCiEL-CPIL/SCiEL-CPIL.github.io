"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import time
    import random
    
    deadline = time.time() + 80
    n = 128
    elements = list(range(1, n+1))
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute LCM table
    lcm_tab = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            v = lcm(elements[i], elements[j])
            lcm_tab[i][j] = v
            lcm_tab[j][i] = v
    
    # Find all LCM triangles and build conflict structure
    # For each x, conflicts[x] = set of frozenset({a,b}) where {x,a,b} is LCM triangle
    # But for efficiency, we want: for each x, list of pairs (a,b) such that x completes triangle
    # And for checking: x can be added to S iff no (a,b) in conflicts[x] has both a,b in S
    
    # Alternative: for each pair (a,b), find all x that complete a triangle
    # pair_to_completers[(a,b)] = set of x
    pair_completers = {}
    for a in range(n):
        for b in range(a+1, n):
            L = lcm_tab[a][b]
            comps = []
            for x in range(n):
                if x != a and x != b and lcm_tab[a][x] == L and lcm_tab[b][x] == L:
                    comps.append(x)
            if comps:
                pair_completers[(a,b)] = comps
    
    # For each x, which pairs does x complete?
    x_completes = [set() for _ in range(n)]  # x_completes[x] = set of (a,b) pairs
    for (a,b), comps in pair_completers.items():
        for x in comps:
            x_completes[x].add((a,b))
    
    def can_add(x, sel_set):
        """Check if x can be added to selected set (by indices)."""
        for (a,b) in x_completes[x]:
            if a in sel_set and b in sel_set:
                return False
        return True
    
    def is_valid(selected):
        sel_set = set(selected)
        for x in sel_set:
            for (a,b) in x_completes[x]:
                if a in sel_set and b in sel_set:
                    return False
        return True
    
    def greedy(order):
        selected = []
        sel_set = set()
        for idx in order:
            if can_add(idx, sel_set):
                selected.append(idx)
                sel_set.add(idx)
        return selected
    
    def local_search(selected, time_limit):
        selected = list(selected)
        sel_set = set(selected)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # Try adding elements
            for x in range(n):
                if x not in sel_set and can_add(x, sel_set):
                    selected.append(x)
                    sel_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-remove, multiple-add
            found_improvement = False
            for ri in range(len(selected)):
                if time.time() - start > time_limit:
                    break
                x_rm = selected[ri]
                new_sel = selected[:ri] + selected[ri+1:]
                new_set = sel_set - {x_rm}
                
                adds = []
                for x in range(n):
                    if x not in new_set and can_add(x, new_set):
                        adds.append(x)
                
                if len(adds) >= 2:
                    test = new_sel + adds
                    if is_valid(test):
                        selected = test
                        sel_set = set(selected)
                        found_improvement = True
                        break
            
            if not found_improvement:
                break
        
        return selected
    
    # Generate diverse orderings
    orderings = [
        list(range(n)),
        list(range(n-1, -1, -1)),
        sorted(range(n), key=lambda i: len(x_completes[i])),
        sorted(range(n), key=lambda i: -len(x_completes[i])),
    ]
    
    rng = random.Random(42)
    for _ in range(30):
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)
    
    # Also try orderings based on element properties
    orderings.append(sorted(range(n), key=lambda i: elements[i]))
    orderings.append(sorted(range(n), key=lambda i: -elements[i]))
    orderings.append(sorted(range(n), key=lambda i: elements[i] * len(x_completes[i])))
    
    best = []
    for order in orderings:
        if time.time() > deadline:
            break
        sol = greedy(order)
        sol = local_search(sol, 2)
        if len(sol) > len(best):
            best = sol
    
    # Extended local search
    if time.time() < deadline - 15:
        best = local_search(best, 40)
    
    # Multiple restarts with perturbation
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            # Perturb: remove random elements, then local search
            perturbed = list(best)
            sel_set = set(perturbed)
            for _ in range(3):
                if perturbed:
                    ri = rng.randint(0, len(perturbed)-1)
                    sel_set.discard(perturbed[ri])
                    perturbed = [x for x in perturbed if x != perturbed[ri]]
            perturbed = local_search(perturbed, 5)
            if len(perturbed) > len(best):
                best = perturbed
    
    result = sorted([elements[i] for i in best])
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
