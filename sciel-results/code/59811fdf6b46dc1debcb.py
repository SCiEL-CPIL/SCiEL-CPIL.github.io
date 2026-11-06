def hot_topo(nodes, parents, children):
    computed = set()
    order = []
    ready = [n for n in nodes if not parents[n]]
    last_processed = {}  # node -> step
    step = 0
    while len(order) < len(nodes):
        # among ready, pick the one with max "hotness" = sum of (step - last_processed[p]) small for parents
        # i.e., parents processed recently
        best = None
        best_score = -1
        for n in ready:
            # score = - sum of ages of parents (younger parents = better)
            ages = [step - last_processed.get(p, -10**9) for p in parents[n]]
            score = -sum(ages)  # maximize (minimize ages)
            if score > best_score:
                best_score = score
                best = n
        if best is None:
            break
        ready.remove(best)
        order.append(best)
        computed.add(best)
        last_processed[best] = step
        step += 1
        for c in children[best]:
            if all(p in computed for p in parents[c]) and c not in computed and c not in ready:
                ready.append(c)
    if len(order) != len(nodes):
        return None
    return order
