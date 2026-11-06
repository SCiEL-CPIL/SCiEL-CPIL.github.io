import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for 60s limit

    def dfs(remaining_terms, current_sum, last_denom, current_denoms):
        if time.time() > deadline:
            return None

        if remaining_terms == 0:
            if current_sum == 1:
                return current_denoms
            return None

        target = 1 - current_sum
        if target <= 0:
            return None

        # min_next: 1/d <= target/remaining_terms => d >= remaining_terms/target
        min_next = int(remaining_terms / target) + 1
        if min_next < last_denom + 1:
            min_next = last_denom + 1

        # max_next: we need 1/d + (remaining_terms-1)/d >= target roughly
        # More precisely: 1/d + sum_{i=1}^{r-1} 1/(d+i) >= target
        # Approx: r/d >= target => d <= r/target
        max_next = int(remaining_terms / target)
        if max_next < min_next:
            return None

        for d in range(min_next, max_next + 1):
            if d < 5:
                continue
            if d <= last_denom:
                continue

            new_sum = current_sum + Fraction(1, d)
            if new_sum > 1:
                continue  # try larger d (smaller reciprocal)

            res = dfs(remaining_terms - 1, new_sum, d, current_denoms + [d])
            if res:
                return res

        return None

    # Try k=8 first (minimum possible for denominators >= 5)
    for k in [8, 9, 10, 11]:
        result = dfs(k, Fraction(0, 1), 4, [])
        if result:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: known valid solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This is too many terms. Let me use a smaller known solution.
    # Actually, let's just use the seed's fallback approach
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no solution found")
