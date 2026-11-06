<<<<<<< SEARCH
    # Phase 1: Backtracking search from multiple starts
    rng = random.Random(12345)
    
    starts = [0, 1, (1 << N) - 1, 0x155, 0x1AA, 0x0FF, 0x1F0, 0x123, 0x1AB]
    for _ in range(15):
        starts.append(rng.randint(0, (1 << N) - 1))
    
    best_path = None
    best_len = 0
    
    for sv in starts:
        if time.time() > deadline - 40:
            break
        # Give each backtracking search a time budget
        search_deadline = min(deadline - 5, time.time() + 3.0)
        path = _backtrack_search(sv, search_deadline, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 2: Greedy extension with bitmask state from random starts
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], deadline - 5, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 3: Local search extension on best path
    if best_path:
        extended = _local_search_extend(best_path, deadline - 5, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Phase 4: More diverse random restarts with local search
    for _ in range(40):
        if time.time() > deadline - 3:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], deadline - 2, rng)
        path = _local_search_extend(path, deadline - 2, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 5: Final local search on best
    if best_path:
        extended = _local_search_extend(best_path, deadline - 1, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended

    # Final: ensure we have at least a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2
    
    return best_path
=======
    # Phase 1: Spectral partition
    rng = random.Random(12345)
    core, periphery = _spectral_partition()
    
    # Phase 2: Construct long path within core using greedy + Warnsdorff
    core_path = _dp_path_in_subset(core, deadline - 60)
    
    best_path = core_path[:]
    best_len = len(core_path)
    
    # Phase 3: Stitch peripheral vertices into the core path
    if best_path:
        stitched = _stitch_periphery(best_path, periphery, deadline - 40, rng)
        if len(stitched) > best_len:
            best_len = len(stitched)
            best_path = stitched
    
    # Phase 4: Local search extension on best path
    if best_path:
        extended = _local_search_extend(best_path, deadline - 20, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Phase 5: Multiple greedy starts with splicing
    paths = []
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        sv = rng.randint(0, NUM_VERTICES - 1)
        path = _greedy_extend([sv], deadline - 5, rng)
        path = _local_search_extend(path, deadline - 5, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
        paths.append(path)
    
    # Phase 6: Splice pairs of paths
    for i in range(min(len(paths), 10)):
        if time.time() > deadline - 5:
            break
        for j in range(i + 1, min(len(paths), 10)):
            if time.time() > deadline - 5:
                break
            spliced = _splice_paths(paths[i], paths[j], deadline - 5)
            if len(spliced) > best_len:
                best_len = len(spliced)
                best_path = spliced
    
    # Phase 7: Final local search on best
    if best_path:
        extended = _local_search_extend(best_path, deadline - 2, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Final: ensure we have at least a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2
    
    # Validate the final path
    if _validate_path(best_path):
        return best_path
    else:
        # Fallback: construct a simple valid path
        return [0, 1]

    def _validate_path(path):
        """Validate that path is a valid induced path in the hypercube."""
        if len(path) < 2:
            return len(path) > 0
        path_set = set(path)
        if len(path_set) != len(path):
            return False
        for i in range(len(path) - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if (path[i] ^ path[j]).bit_count() <= 1:
                    return False
        return True
>>>>>>> REPLACE

