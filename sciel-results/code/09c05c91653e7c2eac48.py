import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 70.0

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Precompute fractions
    fracs = [Fraction(1, d) for d in denominators]
    w_float = [1.0 / d for d in denominators]

    # Multiple greedy initializations
    def greedy(order):
        selected = []
        s = 0.0
        for idx in order:
            if s + w_float[idx] < 1.0:
                selected.append(idx)
                s += w_float[idx]
        return selected, s

    orders = [
        sorted(range(n), key=lambda i: -w_float[i]),  # descending
        sorted(range(n), key=lambda i: w_float[i]),    # ascending
        list(range(n)),                                # original
        list(range(n-1, -1, -1)),                      # reverse
        sorted(range(n), key=lambda i: (w_float[i] * i)),  # mixed
    ]

    best_sel = []
    best_sum_float = -1.0
    for order in orders:
        sel, s = greedy(order)
        if s > best_sum_float:
            best_sum_float = s
            best_sel = sel

    # Convert to set for local search
    current_sel = set(best_sel)
    current_sum = sum(fracs[i] for i in current_sel)

    # Phase 1: Add any element that keeps sum < 1
    improved = True
    while improved and time.time() < deadline - 25:
        improved = False
        for idx in range(n):
            if idx not in current_sel:
                new_sum = current_sum + fracs[idx]
                if new_sum < 1:
                    current_sel.add(idx)
                    current_sum = new_sum
                    improved = True

    # Phase 2: Local search - try swaps (remove one, add one or more)
    for iteration in range(500):
        if time.time() > deadline - 20:
            break

        best_new_sum = current_sum
        best_new_sel = current_sel

        # Try removing each element and adding one
        for rem in list(current_sel):
            rem_sum = current_sum - fracs[rem]
            rem_sel = current_sel - {rem}

            # Try adding each unselected element
            for add in range(n):
                if add not in rem_sel:
                    new_sum = rem_sum + fracs[add]
                    if new_sum < 1 and new_sum > best_new_sum:
                        best_new_sum = new_sum
                        best_new_sel = rem_sel | {add}

        if best_new_sum > current_sum:
            current_sum = best_new_sum
            current_sel = best_new_sel
        else:
            break

    # Phase 3: Try removing pairs and adding triples, etc. (limited)
    for iteration in range(100):
        if time.time() > deadline - 15:
            break

        best_new_sum = current_sum
        best_new_sel = current_sel

        # Try removing 2 elements and adding 1
        sel_list = list(current_sel)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                rem2_sum = current_sum - fracs[sel_list[i]] - fracs[sel_list[j]]
                rem2_sel = current_sel - {sel_list[i], sel_list[j]}

                for add in range(n):
                    if add not in rem2_sel:
                        new_sum = rem2_sum + fracs[add]
                        if new_sum < 1 and new_sum > best_new_sum:
                            best_new_sum = new_sum
                            best_new_sel = rem2_sel | {add}

        if best_new_sum > current_sum:
            current_sum = best_new_sum
            current_sel = best_new_sel
        else:
            break

    result = sorted(current_sel)

    if not result:
        return [0]

    # Final verification
    final_sum = sum(fracs[i] for i in result)
    if final_sum >= 1:
        # Fallback
        return [0]

    return result
