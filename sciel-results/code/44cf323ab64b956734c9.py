<<<<<<< SEARCH
def _greedy_b_fast(a, max_size=64, n=N, seed=0):
    """Greedy construction of B given fixed A."""
    rng = random.Random(seed)
    sa = SidonState(a)
    sb = SidonState()
    
    for x in range(1, n + 1):
        if sb.can_add(x, other_diffs=sa.diffs):
            sb.add(x)
            if len(sb.elements) >= max_size:
                break
    return sb.elements
=======
def _greedy_b_fast(a, max_size=64, n=N, seed=0):
    """Greedy construction of B given fixed A with randomized order."""
    rng = random.Random(seed)
    sa = SidonState(a)
    sb = SidonState()
    
    order = list(range(1, n + 1))
    rng.shuffle(order)
    for x in order:
        if sb.can_add(x, other_diffs=sa.diffs):
            sb.add(x)
            if len(sb.elements) >= max_size:
                break
    return sb.elements
>>>>>>> REPLACE

