"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 128
    # Start with all numbers 1..128
    selected = set(range(1, n + 1))
    
    # We need to ensure no three distinct a,b,c in selected have
    # a+b, a+c, b+c all in selected.
    # A triple (a,b,c) violates iff max(a+b, a+c, b+c) <= n
    # (since all pair sums must be <= n to be in the set 1..n, 
    #  and we started with all of 1..n)
    
    # Greedy removal: find elements that are in the most violating triples
    # and remove them.
    
    def count_violations(elem, selected_set):
        """Count how many violating triples include elem."""
        count = 0
        others = [x for x in selected_set if x != elem]
        # For each pair (b,c) from others, check if (elem, b, c) violates
        # Violation: all three pair sums <= n
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                b, c = others[i], others[j]
                if elem + b <= n and elem + c <= n and b + c <= n:
                    count += 1
        return count
    
    # Iteratively remove the element with the most violations
    # This is O(n^3) per removal in worst case, but n=128 is small
    changed = True
    while changed:
        changed = False
        max_viol = 0
        worst_elem = None
        for elem in selected:
            v = count_violations(elem, selected)
            if v > max_viol:
                max_viol = v
                worst_elem = elem
        if worst_elem is not None and max_viol > 0:
            selected.remove(worst_elem)
            changed = True
    
    return list(selected)

# EVOLVE-BLOCK-END
