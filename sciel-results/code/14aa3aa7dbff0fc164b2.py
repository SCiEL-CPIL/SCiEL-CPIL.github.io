    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # ... loops ...
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
