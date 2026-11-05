import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85

    denominators = list(range(2, 49))
    n = len(denominators)

    # Greedy: try to get as close to 1 from below as possible
    # Sort by value descending (smallest denominators first)
    # But we want to be strategic

    # Strategy: Use greedy to build a sum near 1, then local search
    # The key is that 1/2 + 1/3 + 1/7 + 1/43 = 1 (Egyptian fraction)
    # So we can find subsets that sum to exactly 1, then perturb

    # Let's try a different approach: 
    # Start with all, remove elements to get sum just below 1
    # Or start empty, add elements to get sum just below 1

    # Better: Use a priority queue approach
    # We want sum < 1, maximize sum

    # Greedy with backtracking on a limited set
    # First, let's find a good starting point

    # Try: 1/2 + 1/3 + 1/7 + 1/43 = 1 exactly
    # So 1/2 + 1/3 + 1/7 + 1/42 < 1 (since 1/42 > 1/43)
    # Deficit = 1/43 - 1/42 = (42-43)/(43*42) = -1/1806... wait that's wrong
    # 1/2 + 1/3 + 1/7 + 1/42 = 21/42 + 14/42 + 6/42 + 1/42 = 42/42 = 1
    # So that's exactly 1.
    # 1/2 + 1/3 + 1/7 + 1/43: 1/2+1/3+1/7 = 41/42, so 41/42 + 1/43 = (41*43+42)/(42*43) = (1763+42)/1806 = 1805/1806
    # Deficit = 1/1806

    # Let's verify: 41*43 = 1763, +42 = 1805. 42*43=1806. Yes!
    # So {2,3,7,43} gives deficit 1/1806

    # Can we do better? Try other combinations.
    # 1/2 + 1/3 + 1/8 + 1/24 = 12/24+8/24+3/24+1/24 = 24/24 = 1
    # 1/2 + 1/3 + 1/8 + 1/25 = 23/24 + 1/25 = (23*25+24)/(24*25) = (575+24)/600 = 599/600
    # Deficit = 1/600 > 1/1806, worse

    # 1/2 + 1/4 + 1/5 + 1/20 = 10/20+5/20+4/20+1/20 = 20/20 = 1
    # 1/2 + 1/4 + 1/5 + 1/21 = 19/20 + 1/21 = (19*21+20)/(20*21) = (399+20)/420 = 419/420
    # Deficit = 1/420 > 1/1806

    # 1/2 + 1/3 + 1/7 + 1/43 seems good. Let's try to improve with local search.

    # Start with {2,3,7,43} -> indices [0,1,5,41]
    sel = [0, 1, 5, 41]  # denominators 2,3,7,43

    def current_sum(idx_list):
        return sum(Fraction(1, denominators[i]) for i in idx_list)

    deficit = 1 - current_sum(sel)

    # Local search: try swapping one element out and one in
    sel_set = set(sel)
    unselected = [i for i in range(n) if i not in sel_set]

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for rem in list(sel):
            new_sum = current_sum(sel) - Fraction(1, denominators[rem])
            for add in unselected:
                cand_sum = new_sum + Fraction(1, denominators[add])
                if cand_sum < 1:
                    d = 1 - cand_sum
                    if d < deficit:
                        deficit = d
                        sel.remove(rem)
                        sel.append(add)
                        sel_set.discard(rem)
                        sel_set.add(add)
                        unselected.remove(add)
                        unselected.append(rem)
                        improved = True
                        break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    # Try adding single elements
    if time.time() < deadline - 3:
        base_sum = current_sum(sel)
        for add in unselected:
            cand_sum = base_sum + Fraction(1, denominators[add])
            if cand_sum < 1:
                d = 1 - cand_sum
                if d < deficit:
                    deficit = d
                    sel.append(add)
                    sel_set.add(add)
                    unselected.remove(add)

    # Try removing single elements (if sum is too high, but we know it's < 1)
    # Try 2-swap: remove two, add two (expensive, limit it)
    if time.time() < deadline - 3:
        sel_list = list(sel)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                rem1, rem2 = sel_list[i], sel_list[j]
                new_sum = current_sum(sel) - Fraction(1, denominators[rem1]) - Fraction(1, denominators[rem2])
                for add1 in unselected:
                    for add2 in unselected:
                        if add1 == add2:
                            continue
                        cand_sum = new_sum + Fraction(1, denominators[add1]) + Fraction(1, denominators[add2])
                        if cand_sum < 1:
                            d = 1 - cand_sum
                            if d < deficit:
                                deficit = d
                                sel = [x for x in sel if x not in (rem1, rem2)]
                                sel.append(add1)
                                sel.append(add2)
                                sel_set = set(sel)
                                unselected = [i for i in range(n) if i not in sel_set]
                                improved = True
                                break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break

    # Final verification
    final_sum = current_sum(sel)
    if sel and final_sum < 1:
        return sel
    return [0]
