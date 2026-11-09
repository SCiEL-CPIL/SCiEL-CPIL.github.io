idx = random.randint(0, len(subsets) - 1)
if not was_violated[idx]:
    # Try a few more times
    for _ in range(10):
        idx = random.randint(0, len(subsets) - 1)
        if was_violated[idx]:
            break
    else:
        # If we really can't find one, recount
        pass
