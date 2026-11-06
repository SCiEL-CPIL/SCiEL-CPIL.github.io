for v in order:
    if not parents[v]:
        # input, already blue; ensure in mem
        if v not in mem:
            if len(mem)>=B: evict()
            mem.add(v); ops.append(['load', v])
        continue
    ... compute ...
