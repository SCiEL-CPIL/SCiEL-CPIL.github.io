"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80

    smooth = set()
    for i in range(10):
        for j in range(7):
            v = (1 << i) * (3**j)
            if 1 <= v <= 512:
                smooth.add(v)

    all_candidates = list(range(1, 513))

    def try_greedy(order):
        selected = []
        selected_set = set()

        for x in order:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue

            valid = True
            n = len(selected)

            if n >= 2:
                max_sum = sum(selected)
                dp = [0] * (n + 1)
                dp[0] = 1
                for elem in selected:
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k-1] << elem)
                for k in range(2, n + 1):
                    target = k * x
                    if target <= max_sum and (dp[k] >> target) & 1:
                        valid = False
                        break

            if valid:
                for a in selected:
                    others = [y for y in selected if y != a]
                    m = len(others)
                    if m == 0:
                        continue
                    max_others = sum(others)
                    dp2 = [0] * (m + 1)
                    dp2[0] = 1
                    for elem in others:
                        for k in range(m, 0, -1):
                            dp2[k] |= (dp2[k-1] << elem)
                    for k in range(1, m + 1):
                        target = (k+1) * a - x
                        if 0 <= target <= max_others:
                            if (dp2[k] >> target) & 1:
                                valid = False
                                break
                    if not valid:
                        break

            if valid:
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= 48:
                    break

        return selected

    smooth_list = sorted(smooth)
    rest = [x for x in all_candidates if x not in smooth]
    order = smooth_list + rest

    best = try_greedy(order)

    if len(best) < 48 and time.time() < deadline - 10:
        order2 = sorted(all_candidates, reverse=True)
        result2 = try_greedy(order2)
        if len(result2) > len(best):
            best = result2

    return best
# EVOLVE-BLOCK-END
