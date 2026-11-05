"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Find a strictly increasing sequence of integers in [1, 256] such that
    sums of all nonempty consecutive blocks are distinct. Maximize length.
    """
    
    def greedy(skip_values=None, start_val=1):
        if skip_values is None:
            skip_values = frozenset()
        
        seq = []
        block_sums = set()
        suffix_sums = set()
        
        for x in range(start_val, 257):
            if x in skip_values:
                continue
            if x in block_sums:
                continue
            new_suffix = {x}
            for s in suffix_sums:
                new_suffix.add(x + s)
            if new_suffix & block_sums:
                continue
            seq.append(x)
            block_sums |= new_suffix
            suffix_sums = new_suffix
        
        return seq
    
    # Start with pure greedy
    best = greedy()
    best_len = len(best)
    
    # Try skipping individual early values
    for i in range(1, 30):
        candidate = greedy(frozenset([i]))
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
    
    # Try skipping pairs
    for i in range(1, 20):
        for j in range(i + 1, 25):
            candidate = greedy(frozenset([i, j]))
            if len(candidate) > best_len:
                best = candidate
                best_len = len(candidate)
    
    # Try different starting values
    for start in range(2, 10):
        candidate = greedy(start_val=start)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
    
    return best

# EVOLVE-BLOCK-END
