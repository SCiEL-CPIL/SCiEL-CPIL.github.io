def _delta_energy(sequence, chain, occupied, k, new_tail):
    # Calculate change in energy due to moving tail to new_tail.
    # Old tail points: chain[k:]
    # New tail points: new_tail
    # Body points: chain[:k]
    
    # 1. Remove contacts involving old tail
    # 2. Add contacts involving new tail
    
    # This is tricky because contacts between tail points also change.
    # Given the complexity, and N=100, full recalculation is safer.
    # Let's stick to full recalculation but optimize the dict.
