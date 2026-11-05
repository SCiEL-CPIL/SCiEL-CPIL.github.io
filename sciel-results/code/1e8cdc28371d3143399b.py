def find_swap_candidate(uncov, in_code, c_remove, cover_map, ball_sets):
    if not uncov:
        return -1
    # Sort by rarity
    uncov_sorted = sorted(uncov, key=lambda w: len(cover_map[w]))
    candidates = cover_map[uncov_sorted[0]]
    for c in candidates:
        if in_code[c] or c == c_remove:
            continue
        # Check if c covers all others
        ok = True
        bs = ball_sets[c]
        for w in uncov_sorted[1:]:
            if w not in bs:
                ok = False
                break
        if ok:
            return c
    return -1
