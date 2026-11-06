    block_size = 2 * 10**6
    current = 1
    while current < 10**12 and time.time() < deadline - 5:
        end = current + block_size - 1
        if end > 10**12: end = 10**12
        scan_range(current, end)
        current = end + 1
        if best_length >= 200:
            break
