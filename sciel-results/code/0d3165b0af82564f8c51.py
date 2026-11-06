"""Rewritten program using reverse-order greedy for maximum sequence length."""
# EVOLVE-BLOCK-START
def construct():
    values = []
    sums = set()
    
    # Reverse greedy: try larger values first to maximize flexibility
    for x in range(256, 0, -1):
        # Compute all new consecutive block sums ending at x
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        
        # Check if any new sum collides
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    # Return in strictly increasing order
    values.sort()
    return values
# EVOLVE-BLOCK-END
