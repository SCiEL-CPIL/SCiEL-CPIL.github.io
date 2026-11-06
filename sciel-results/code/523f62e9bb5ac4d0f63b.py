        while len(blue) > budget - 1:
            # first, delete any blue vertex (not parent of v) with no more uses
            deleted = False
            for w in list(blue):
                if w in parent_set:
                    continue
                if next_use(w, i) == -1:
                    ops.append(['delete', w])
                    blue.remove(w)
                    deleted = True
                    break
            if deleted:
                continue
            # else, store the farthest-next-use
            best = None
            best_nu = -2
            for w in blue:
                if w in parent_set:
                    continue
                nu = next_use(w, i)
                if nu > best_nu:
                    best_nu = nu
                    best = w
            if best is None:
                return None, None
            ops.append(['store', best])
            blue.remove(best)
            transfers += 1
