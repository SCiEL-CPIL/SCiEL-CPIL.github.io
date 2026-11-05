import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def check_valid(values):
        """Check if all nonempty consecutive block products are distinct."""
        prods = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                if prod in prods:
                    return False
                prods.add(prod)
        return True

    # Greedy baseline: try each number in order, add if no product collision
    values, products = [], set()
    for x in range(2, 129):
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)

    best = list(values)

    # Local improvement: try inserting unused numbers at various positions
    used_set = set(best)
    all_nums = list(range(2, 129))
    unused = [x for x in all_nums if x not in used_set]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in unused[:]:
            if time.time() > deadline:
                break
            for pos in range(len(best) + 1):
                if time.time() > deadline:
                    break
                new_vals = best[:pos] + [x] + best[pos:]
                if check_valid(new_vals):
                    best = new_vals
                    used_set.add(x)
                    unused.remove(x)
                    improved = True
                    break
            if improved:
                break

    return best
