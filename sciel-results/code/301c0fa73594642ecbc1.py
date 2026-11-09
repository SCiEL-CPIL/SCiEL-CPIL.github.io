fast = set()
slow = set()  # computed & stored (inputs are implicitly always loadable)
ops = []

def make_room(protect):
    while len(fast) >= B and (fast - protect):
        # evict best non-protected
        victim = min(fast - protect, key=lambda w: (0 if is_input(w) else 1, remaining_uses(w), recompute_size(w)))
        if is_input(victim):
            ops.append(['delete', victim]); fast.discard(victim)
        else:
            ops.append(['store', victim]); transfers+=1; fast.discard(victim); slow.add(victim)
    # if still full and all protected, we're stuck (infeasible) -> evict a protected as last resort
    while len(fast) >= B and not (fast - protect):
        victim = min(fast, key=lambda w: recompute_size(w))
        if is_input(victim):
            ops.append(['delete', victim]); fast.discard(victim)
        else:
            ops.append(['store', victim]); transfers+=1; fast.discard(victim); slow.add(victim)
        # mark to recompute victim later if needed
        return  # caller must re-ensure

def ensure(v, protect=frozenset()):
    if v in fast: return True
    if can_load(v):
        make_room(protect | {v})
        ops.append(['load', v]); transfers+=1
        fast.add(v); slow.discard(v)
        return True
    # compute
    par = set(parent_map[v])
    ok = True
    for p in parent_map[v]:
        if not ensure(p, protect | par):
            ok = False
    make_room(protect | par | {v})
    # re-ensure parents (in case evicted in last-resort)
    for p in par:
        if p not in fast:
            ensure(p, protect | par)
    ops.append(['compute', v])
    fast.add(v)
    return True
