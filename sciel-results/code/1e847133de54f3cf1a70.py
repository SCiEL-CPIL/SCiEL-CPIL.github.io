def _initial_chain(length, rng, max_retries=100):
    for _ in range(max_retries):
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0)}
        success = True
        for _ in range(length - 1):
            last = chain[-1]
            neighbors = _neighbors(last)
            rng.shuffle(neighbors)
            found = False
            for n in neighbors:
                if n not in occupied:
                    chain.append(n)
                    occupied.add(n)
                    found = True
                    break
            if not found:
                success = False
                break
        if success:
            return chain
    # Fallback: straight line
    return [(i, 0, 0) for i in range(length)]
