    while True:
        if time.time() > deadline - 10:
            break
        max_count = max(cover_count)
        if max_count == 0:
            break
        best_t = cover_count.index(max_count)
        removed.add(best_t)
        for idx in configs_containing[best_t]:
            if is_active[idx]:
                is_active[idx] = False
                for t in configs[idx]:
                    if t != best_t:
                        cover_count[t] -= 1
