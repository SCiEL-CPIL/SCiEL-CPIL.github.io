while len(red) > budget:
    cands = list(red)
    if not cands:
        break
    # prefer non-outputs, then by last_use
    cands.sort(key=lambda x: (x in outputs, last_use[x]))
    w = cands[0]
    do_store(w)
