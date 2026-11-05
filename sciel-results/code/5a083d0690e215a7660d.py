# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Greedy initialization
    selected = set()
    total = Fraction(0, 1)
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.add(i)
            total += Fraction(1, d)
    
    # Local search
    for _ in range(1000):  # Limit iterations
        improved = False
        # Try adding
        for i in range(n):
            if i not in selected:
                if total + Fraction(1, denominators[i]) < 1:
                    selected.add(i)
                    total += Fraction(1, denominators[i])
                    improved = True
                    break
        if improved:
            continue
        
        # Try swapping
        for i in range(n):
            if i in selected:
                continue
            for j in list(selected):
                new_total = total - Fraction(1, denominators[j]) + Fraction(1, denominators[i])
                if new_total < 1 and new_total > total:
                    selected.remove(j)
                    selected.add(i)
                    total = new_total
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return sorted(selected)
# EVOLVE-BLOCK-END
