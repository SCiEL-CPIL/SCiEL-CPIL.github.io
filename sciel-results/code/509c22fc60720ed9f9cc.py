<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    # Additional optimization: try to swap adjacent comparators to enable further removals
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(network) - 1):
                if time.time() > deadline - 5.0:
                    break
                # Try swapping network[i] and network[i+1]
                if network[i] != network[i+1]:
                    new_network = network.copy()
                    new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
                    if _is_sorting_network_fast(new_network, all_inputs):
                        # Check if this swap enables removal of one of them
                        for remove_idx in [i, i+1]:
                            test_network = new_network[:remove_idx] + new_network[remove_idx+1:]
                            if _is_sorting_network_fast(test_network, all_inputs):
                                network = test_network
                                improved = True
                                break
                        if improved:
                            break
    
    return network
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    # Additional optimization: try to swap adjacent comparators to enable further removals
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(network) - 1):
                if time.time() > deadline - 5.0:
                    break
                # Try swapping network[i] and network[i+1]
                if network[i] != network[i+1]:
                    new_network = network.copy()
                    new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
                    if _is_sorting_network_fast(new_network, all_inputs):
                        # Check if this swap enables removal of one of them
                        for remove_idx in [i, i+1]:
                            test_network = new_network[:remove_idx] + new_network[remove_idx+1:]
                            if _is_sorting_network_fast(test_network, all_inputs):
                                network = test_network
                                improved = True
                                break
                        if improved:
                            break
    
    return network
>>>>>>> REPLACE
