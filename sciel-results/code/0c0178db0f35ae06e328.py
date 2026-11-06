        # Branch 1: Include x
        new_inc = inc | {x}
        new_exc_set = set(exc)
        new_cand_set = set(cand - {x})
        
        # Pre-propagate
        for y in inc:
            z1 = 2 * x - y
            z2 = 2 * y - x
            for z in (z1, z2):
                if 0 < z <= n:
                    if z in new_cand_set:
                        new_cand_set.remove(z)
                        new_exc_set.add(z)
                    elif z in new_inc:
                        # Contradiction
                        new_cand_set = set()
                        break
            if not new_cand_set: break
        if not new_cand_set:
            # Contradiction, don't push this branch
            pass
        else:
            stack.append((new_inc, frozenset(new_exc_set), frozenset(new_cand_set), curr_w + weights[x]))
