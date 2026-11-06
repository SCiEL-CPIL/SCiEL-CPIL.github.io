computed = set()

def next_use(p):
    for c in children[p]:
        if c not in computed:
            return topo_index[c]
    return float('inf')

def make_room():
    while len(in_mem) >= fast_memory_budget and _now() < _DEADLINE:
        # Belady: evict the vertex with furthest next use
        candidates = [v for v in in_mem if v not in outputs]
        if not candidates:
            break
        v = max(candidates, key=lambda x: next_use(x))
        if next_use(v) == float('inf'):
            delete(v)  # dead, free
        else:
            store(v)
