idx = 0
while idx < len(current_network):
    if time.time() > deadline: break
    candidate = current_network[:idx] + current_network[idx + 1:]
    if _verify_network_fast(candidate):
        current_network = candidate
        # Do not increment idx, check the next element which is now at idx
        best_size = len(current_network)
        best_network = list(current_network)
    else:
        idx += 1
