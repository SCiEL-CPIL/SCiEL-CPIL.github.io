            new_sums = set(current_sums)
            valid = True
            for s in list(current_sums):
                ns = s + r
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                selected.append(idx)
                backtrack(selected, remaining[i+1:], new_sums)
