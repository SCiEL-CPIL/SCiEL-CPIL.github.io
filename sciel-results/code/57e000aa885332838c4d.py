    changed = True
    while changed:
        changed = False
        idx = 0
        while idx < len(network):
            reduced = network[:idx] + network[idx+1:]
            if _validate_network(reduced):
                network = reduced
                changed = True
                # stay at same idx
            else:
                idx += 1
