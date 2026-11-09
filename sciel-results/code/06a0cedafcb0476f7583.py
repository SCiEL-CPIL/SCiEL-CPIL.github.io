import time

def pebble(dag, fast_memory_budget):
    start = time.time()
    DEADLINE = 85.0

    # ---- Parse DAG ----
    parents = None
    outputs = None
    n = None
    try:
        if isinstance(dag, dict):
            parents = dag.get('parents')
            outputs = dag.get('outputs')
            n = dag.get('n')
        elif isinstance(dag, (tuple, list)):
            parents = dag[0]
            outputs = dag[1] if len(dag) > 1 else None
            n = len(parents)
    except Exception:
        pass

    if parents is None:
        parents = []
    if outputs is None:
        outputs = []
    if n is None:
        n = len(parents)

    budget = fast_memory_budget
    try:
        budget = int(budget)
    except Exception:
        budget = 1
    if budget < 1:
        budget = 1

    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    is_input = [len(parents[v]) == 0 for v in range(n)]
    is_output = [False] * n
    for o in outputs:
        if 0 <= o < n:
            is_output[o] = True

    fast = set()
    transfers = 0
    ops = []
    next_ptr = [0] * n

    def choose_victim(fast_set, current):
        best = None
        best_key = None
        for x in fast_set:
            if x == current:
                continue
            np_ = next_ptr[x]
            if np_ < len(children[x]):
                next_use = children[x][np_]
            else:
                next_use = float('inf')
            if is_output[x]:
                key = (2, next_use)
            elif next_use == float('inf'):
                key = (0, 0)
            else:
                key = (1, next_use)
            if best is None:
                best = x
                best_key = key
            else:
                # prefer lower first component (delete=0 best, store=1, output=2)
                if key[0] < best_key[0]:
                    best = x
                    best_key = key
                elif key[0] == best_key[0]:
                    if key[0] == 1:
                        # store: evict the one with farthest next use
                        if key[1] > best_key[1]:
                            best = x
                            best_key = key
                    elif key[0] == 2:
                        if key[1] > best_key[1]:
                            best = x
                            best_key = key
        return best

    for v in range(n):
        if time.time() - start > DEADLINE:
            break
        if is_input[v]:
            continue
        for p in parents[v]:
            if p not in fast:
                fast.add(p)
                transfers += 1
                ops.append(['load', p])
        fast.add(v)
        ops.append(['compute', v])
        for p in parents[v]:
            while next_ptr[p] < len(children[p]) and children[p][next_ptr[p]] <= v:
                next_ptr[p] += 1
        while len(fast) > budget:
            victim = choose_victim(fast, v)
            if victim is None:
                break
            fast.remove(victim)
            np_ = next_ptr[victim]
            if np_ >= len(children[victim]) and not is_output[victim]:
                ops.append(['delete', victim])
            else:
                transfers += 1
                ops.append(['store', victim])

    for o in outputs:
        if 0 <= o < n and o in fast:
            fast.remove(o)
            transfers += 1
            ops.append(['store', o])

    return ops
