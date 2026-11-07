"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 40
    candidates = list(range(2, n + 1))
    selected = []
    for a in candidates:
        new_selected = selected + [a]
        k = len(new_selected)
        products = set()
        ok = True
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= new_selected[i]
            if prod in products:
                ok = False
                break
            products.add(prod)
        if ok:
            selected = new_selected
            if len(selected) >= 18:
                break
    return selected

# EVOLVE-BLOCK-END
