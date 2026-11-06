<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50  # Safety margin before 60s evaluator kill
    
    # Start with odd-even transposition
    network = _construct_odd_even_transposition()
    best_network = network[:]
    best_count = len(network)
    
    # Verify initial network
    if not _is_valid(network):
        return best_network
    
    # Phase 1: Greedy removal with quick checks
    while time.time() < deadline:
        removed_any = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            # Quick check first
            if _quick_check(candidate):
                # Full verification
                if _is_valid(candidate):
                    network = candidate
                    best_network = network[:]
                    best_count = len(network)
                    removed_any = True
                    break
        if not removed_any:
            break
    
    # Phase 2: Try moving comparators to different positions
    while time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            comp = network[idx]
            for new_idx in range(len(network)):
                if new_idx == idx or new_idx == idx - 1:
                    continue
                if time.time() >= deadline:
                    break
                candidate = network[:idx] + network[idx+1:]
                insert_pos = new_idx if new_idx < idx else new_idx
                candidate = candidate[:insert_pos] + [comp] + candidate[insert_pos:]
                if len(candidate) == len(network) and candidate != network:
                    if _quick_check(candidate):
                        if _is_valid(candidate):
                            network = candidate
                            best_network = network[:]
                            best_count = len(network)
                            improved = True
                            break
            if improved:
                break
        if not improved:
            break
    
    return best_network
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # Safety margin before 60s evaluator kill
    
    # Start with odd-even transposition (136 comparators)
    network = _construct_odd_even_transposition()
    best_network = network[:]
    best_count = len(network)
    
    # Verify initial network
    if not _is_valid(network):
        return best_network
    
    # Phase 1: Greedy removal - try removing each comparator
    # Use a smarter strategy: try removing from the end first (later comparators are often redundant)
    while time.time() < deadline - 5:
        removed_any = False
        # Try removing from the end first, then middle, then beginning
        for idx in list(range(len(network) - 1, -1, -1)):
            if time.time() >= deadline - 5:
                break
            candidate = network[:idx] + network[idx+1:]
            # Quick check first (much faster)
            if _quick_check(candidate):
                # Full verification
                if _is_valid(candidate):
                    network = candidate
                    best_network = network[:]
                    best_count = len(network)
                    removed_any = True
                    break
        if not removed_any:
            break
    
    # Phase 2: Try swapping adjacent comparators to find better orderings
    while time.time() < deadline - 2:
        improved = False
        for idx in range(len(network) - 1):
            if time.time() >= deadline - 2:
                break
            # Try swapping network[idx] and network[idx+1]
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if candidate != network:
                if _quick_check(candidate):
                    if _is_valid(candidate):
                        network = candidate
                        best_network = network[:]
                        best_count = len(network)
                        improved = True
                        break
        if not improved:
            break
    
    # Phase 3: Try removing more aggressively with different starting points
    while time.time() < deadline:
        removed_any = False
        # Try all positions again after reordering
        for idx in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _quick_check(candidate):
                if _is_valid(candidate):
                    network = candidate
                    best_network = network[:]
                    best_count = len(network)
                    removed_any = True
                    break
        if not removed_any:
            break
    
    return best_network
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
