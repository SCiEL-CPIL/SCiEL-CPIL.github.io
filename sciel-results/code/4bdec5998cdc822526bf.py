def construct():
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(12345)

    # best solution
    best_classes, best_cover, best_score = None, None, None

    def _consider(classes, cover):
        nonlocal best_classes, best_cover, best_score
        s = _uncovered_count(cover)
        if best_score is None or s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            best_cover = bytearray(cover)

    # greedy init + local search
    g_classes, g_cover = _greedy_init()
    g_classes, g_cover, g_score = _local_search(g_classes, g_cover, deadline, rng)
    _consider(g_classes, g_cover)

    # random restarts
    while time.time() < deadline:
        r_classes, r_cover = _random_init(rng)
        r_classes, r_cover, r_score = _local_search(r_classes, r_cover, deadline, rng)
        _consider(r_classes, r_cover)

    return best_classes
