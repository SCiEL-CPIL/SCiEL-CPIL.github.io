def _insertion_local_search(processing_times, order, deadline, max_passes=500):
    """Best-improvement insertion local search with efficient prefix reuse."""
    n_machines = processing_times.shape[1]
    n = len(order)
    if n == 0:
        return list(order), 0.0
    current = list(order)
    current_ms = _makespan(processing_times, current)
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        # Prefix vectors for current order
        cur_prefixes = _prefix_vectors(processing_times, current)
        for i in range(n):
            if time.time() > deadline:
                return current, current_ms
            job = current[i]
            # Reduced order = current without element i
            # Prefix vectors for reduced:
            #   red_prefix[k] = cur_prefix[k] for k <= i
            #   red_prefix[k] = cur_prefix[k+1] for k > i
            # We can build a lightweight accessor
            reduced = current[:i] + current[i+1:]
            # Build red_prefixes as a list of references (no copy needed for read)
            # red_prefixes[k] for k in 0..n-1 (reduced has n-1 elements, so n prefixes + 1)
            # reduced length = n-1, so prefixes count = n
            # red_prefixes[0..i] = cur_prefixes[0..i]
            # red_prefixes[i+1..n-1] = cur_prefixes[i+2..n]
            red_prefixes = []
            for k in range(n):  # reduced has n-1 elements -> n prefixes
                if k <= i:
                    red_prefixes.append(cur_prefixes[k])
                else:
                    red_prefixes.append(cur_prefixes[k+1])
            best_pos, best_ms = _best_insertion(processing_times, reduced, red_prefixes, job)
            if best_ms < current_ms - 1e-9:
                new_order = reduced[:best_pos] + [job] + reduced[best_pos:]
                current = new_order
                current_ms = best_ms
                improved = True
                break
    return current, current_ms
