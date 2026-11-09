"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import random
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + fracs[i] < 1:
            selected.add(i)
            total += fracs[i]
    
    best_set = set(selected)
    best_total = total
    best_deficit = 1 - best_total
    
    # Local search: try to improve
    for iteration in range(5000):
        improved = False
        for i in range(n):
            # Try adding i if not in set
            if i not in best_set:
                new_total = best_total + fracs[i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        best_set = set(best_set)
                        best_set.add(i)
                        best_total = new_total
                        best_deficit = new_deficit
                        improved = True
            
            # Try removing i if in set
            if i in best_set:
                new_total = best_total - fracs[i]
                new_deficit = 1 - new_total
                if new_deficit < best_deficit:
                    best_set = set(best_set)
                    best_set.remove(i)
                    best_total = new_total
                    best_deficit = new_deficit
                    improved = True
        
        # Try swaps: remove one, add one
        for i in list(best_set):
            for j in range(n):
                if j not in best_set:
                    new_total = best_total - fracs[i] + fracs[j]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < best_deficit:
                            best_set = set(best_set)
                            best_set.remove(i)
                            best_set.add(j)
                            best_total = new_total
                            best_deficit = new_deficit
                            improved = True
        
        if not improved:
            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
