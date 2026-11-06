def _extend(selected):
    selected = list(selected)
    if len(selected) >= _MAXK:
        return selected
    has_sum, pairres = _build_state(selected)
    selset = set(selected)
    for x in range(1, _N + 1):
        if len(selected) >= _MAXK:
            break
        if x not in selset and _try_add(x, selected, has_sum, pairres):
            _do_add(x, selected, has_sum, pairres)
            selset.add(x)
    return selected
