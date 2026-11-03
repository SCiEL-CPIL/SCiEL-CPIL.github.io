    for r in range(7, best_rank):
        if time.time() > deadline - 5:
            break
        start = time.time()
        t = 0
        res = None
        while time.time() < deadline - 5 and t < 200:
            t += 1
            res = one_trial(r)
            if res is not None:
                break
        if res is not None:
            ...
            break
