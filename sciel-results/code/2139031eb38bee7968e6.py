    if best_solution is None:
        # Fallback to greedy
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
