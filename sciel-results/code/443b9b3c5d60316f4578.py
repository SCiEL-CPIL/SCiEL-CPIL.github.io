        s.append(best_x)
        s.sort()
        # Recalculate used_diffs for safety or update incrementally
        # Incremental:
        for y in s[:-1]: # y is in old s
             d = best_x - y
             used_diffs.add(d)
