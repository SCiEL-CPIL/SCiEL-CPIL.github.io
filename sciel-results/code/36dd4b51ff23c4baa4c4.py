def construct():
    start = time.time()
    deadline = start + 85
    
    # Generate Batcher network
    network = _batcher_sort(N, 0)
    
    if not _verify_fast(network, N):
        network = _odd_even_transposition_network(N)
    
    best = list(network)
    best_len = len(best)
    
    # Aggressive removal
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Back-to-front removal
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_fast(candidate, N):
                best = candidate
                best_len = len(best)
                improved = True
                break  # restart pass since indices shifted
        # Front-to-back removal
        if time.time() < deadline - 5:
            for idx in range(len(best)):
                if time.time() > deadline - 5:
                    break
                candidate = best[:idx] + best[idx+1:]
                if _verify_fast(candidate, N):
                    best = candidate
                    best_len = len(best)
                    improved = True
                    break
    
    # Reordering: try to move comparators
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(len(best)):
                if time.time() > deadline - 10:
                    break
                i1, j1 = best[idx]
                # Try moving to position idx+1, idx+2, ... up to a few
                for new_idx in range(idx+1, min(idx+5, len(best))):
                    # Check if wires are disjoint with intervening comparators
                    ok = True
                    for mid in range(idx+1, new_idx):
                        i2, j2 = best[mid]
                        if i2 == i1 or i2 == j1 or j2 == i1 or j2 == j1:
                            ok = False
                            break
                    if not ok:
                        break
                    candidate = best[:idx] + best[idx+1:new_idx+1] + [best[idx]] + best[new_idx+1:]
                    if _verify_fast(candidate, N):
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
    
    return best
