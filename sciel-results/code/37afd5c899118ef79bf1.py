<<<<<<< SEARCH
def _build_initial_network():
    """Build an initial sorting network using bitonic sort."""
    return _bitonic_sort_network(N)

def _local_search(net, max_time=50.0):
    """Local search to improve the network."""
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    
    while time.time() - start_time < max_time:
        improved = False
        # Try removing each comparator
        for idx in range(len(best_net)):
            candidate = best_net[:idx] + best_net[idx+1:]
            if len(candidate) < best_score and _verify_network(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        if not improved:
            # Try replacing one comparator with another
            for idx in range(len(best_net)):
                i, j = best_net[idx]
                # Try swapping with other possible comparators
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) == (i, j):
                            continue
                        candidate = list(best_net)
                        candidate[idx] = (a, b)
                        if len(candidate) == best_score and _verify_network(candidate):
                            best_net = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Try adding a comparator
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) in best_net:
                            continue
                        candidate = list(best_net)
                        candidate.append((a, b))
                        if _verify_network(candidate):
                            best_net = candidate
                            best_score = len(best_net)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    return best_net
=======
def _build_initial_network():
    """Build an initial sorting network using bitonic sort."""
    return _bitonic_sort_network(N)

def _local_search(net, max_time=50.0):
    """Local search to improve the network using greedy removal."""
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    
    # Precompute set of comparators for quick lookup
    net_set = set(best_net)
    
    while time.time() - start_time < max_time:
        improved = False
        # Try removing each comparator (greedy)
        for idx in range(len(best_net)):
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network(candidate):
                best_net = candidate
                best_score = len(best_net)
                net_set.discard(best_net[idx]) if False else None
                # Rebuild set
                net_set = set(best_net)
                improved = True
                break
        if not improved:
            # Try replacing one comparator with another
            improved = False
            for idx in range(len(best_net)):
                i, j = best_net[idx]
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) == (i, j):
                            continue
                        candidate = list(best_net)
                        candidate[idx] = (a, b)
                        if _verify_network(candidate):
                            best_net = candidate
                            net_set = set(best_net)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_net
>>>>>>> REPLACE

