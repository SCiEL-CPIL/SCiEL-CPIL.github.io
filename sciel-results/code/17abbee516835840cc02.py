def _calculate_energy_delta(chain, sequence, pivot_idx, new_tail_positions):
    """
    Calculate energy difference for a pivot move at pivot_idx.
    chain: current list of tuples
    sequence: string
    pivot_idx: index of pivot (head is 0..pivot_idx, tail is pivot_idx+1..N-1)
    new_tail_positions: list of new positions for indices pivot_idx+1..N-1
    """
    n = len(chain)
    # Identify H indices in head and tail
    head_h_indices = [i for i in range(pivot_idx + 1) if sequence[i] == 'H']
    tail_h_indices = [i for i in range(pivot_idx + 1, n) if sequence[i] == 'H']
    
    if not head_h_indices or not tail_h_indices:
        return 0

    # Build set of head H positions for O(1) lookup
    head_h_pos = {chain[i] for i in head_h_indices}
    
    # Calculate old head-tail contacts
    old_contacts = 0
    for i in tail_h_indices:
        p = chain[i]
        # Check 4 neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            q = (p[0] + dx, p[1] + dy)
            if q in head_h_pos:
                old_contacts += 1
                
    # Calculate new head-tail contacts
    new_contacts = 0
    # new_tail_positions corresponds to indices pivot_idx+1 to n-1
    for k, i in enumerate(range(pivot_idx + 1, n)):
        if sequence[i] == 'H':
            p = new_tail_positions[k]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                q = (p[0] + dx, p[1] + dy)
                if q in head_h_pos:
                    new_contacts += 1

    return new_contacts - old_contacts
