        q3_edges = find_q3(current_bitset)
        if q3_edges is None:
            # Valid
            # ... add edges logic ...
        else:
            # Remove a random edge from q3_edges
            rem_idx = random.choice(q3_edges)
            current_bitset &= ~(1 << rem_idx)
