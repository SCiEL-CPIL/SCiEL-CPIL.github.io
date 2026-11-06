"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    def has_3ap(sset):
        """Check if a set contains any 3-term arithmetic progression."""
        for a in sset:
            for b in sset:
                if a < b:
                    c = 2 * b - a
                    if c > b and c in sset:
                        return True
        return False
    
    def reciprocal_sum(s):
        return sum(1.0 / x for x in s)
    
    # Greedy construction: add x if it doesn't form a 3-AP
    selected = []
    sset = set()
    for x in range(1, n + 1):
        # Check if adding x creates a 3-AP with any two existing elements
        valid = True
        for a in sset:
            # x, a, b form AP if 2*a = x + b => b = 2*a - x
            b = 2 * a - x
            if b > 0 and b != a and b != x and b in sset:
                valid = False
                break
            # a, x, b form AP if 2*x = a + b => b = 2*x - a
            b = 2 * x - a
            if b > x and b in sset:
                valid = False
                break
            # a, b, x form AP if 2*b = a + x => b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b > a and b != x and b in sset:
                    valid = False
                    break
        if valid:
            selected.append(x)
            sset.add(x)
    
    current_sum = reciprocal_sum(selected)
    
    # Local search: try single swaps and single additions/removals
    improved = True
    while improved:
        improved = False
        sset = set(selected)
        
        # Try adding an element
        for x in range(1, n + 1):
            if x in sset:
                continue
            new_sset = sset | {x}
            if not has_3ap(new_sset):
                new_sum = current_sum + 1.0 / x
                if new_sum > current_sum + 1e-12:
                    selected.append(x)
                    sset.add(x)
                    current_sum = new_sum
                    improved = True
        
        # Try single swaps: remove one, add another
        for i in range(len(selected)):
            for x in range(1, n + 1):
                if x in sset:
                    continue
                new_sset = sset - {selected[i]} | {x}
                if not has_3ap(new_sset):
                    new_sum = current_sum - 1.0 / selected[i] + 1.0 / x
                    if new_sum > current_sum + 1e-12:
                        selected[i] = x
                        sset = new_sset
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break
        
        # Try removing an element if it helps (unlikely but check)
        for i in range(len(selected)):
            if len(selected) <= 1:
                break
            new_sset = sset - {selected[i]}
            # Removing always improves reciprocal sum only if we can add back better
            # So skip pure removals unless we're stuck
            pass
    
    return selected
# EVOLVE-BLOCK-END
