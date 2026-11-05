def construct():
    import random
    import time
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Precompute all 6-subsets and their constituent 5-subsets
    # Points are 0..11
    points = range(12)
    all_6_subs = list(combinations(points, 6))
    
    # Map 5-subsets to indices for fast lookup
    all_5_subs = list(combinations(points, 5))
    five_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    num_5_subs = len(all_5_subs) # 792

    # For each 6-subset, store the list of indices of its 5-subsets
    six_to_fives = []
    for s6 in all_6_subs:
        fives = []
        for i in range(6):
            s5 = s6[:i] + s6[i+1:]
            fives.append(five_to_idx[s5])
        six_to_fives.append(tuple(fives))

    best_blocks = []
    best_count = 0

    # Greedy function
    def greedy(order):
        used = [False] * num_5_subs
        blocks = []
        for idx in order:
            fives = six_to_fives[idx]
            if not any(used[f] for f in fives):
                for f in fives:
                    used[f] = True
                blocks.append(all_6_subs[idx])
        return blocks

    # Run multiple restarts
    random.seed(42)
    while time.time() < deadline:
        order = list(range(len(all_6_subs)))
        random.shuffle(order)
        blocks = greedy(order)
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # If we hit 132, we can stop
            if best_count == 132:
                break

    return [list(b) for b in best_blocks]
