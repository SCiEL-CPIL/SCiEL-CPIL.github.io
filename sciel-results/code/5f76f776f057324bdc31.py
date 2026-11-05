<<<<<<< SEARCH
def _try_remove_comparator(network, idx):
    """Try removing comparator at index idx. Returns (success, new_network)."""
    candidate = network[:idx] + network[idx+1:]
    if _verify_booleans_fast(candidate):
        return True, candidate
    return False, None


def _try_swap_comparators(network, i, j):
    """Try swapping comparators at positions i and j. Returns (success, new_network)."""
    if i >= j:
        return False, None
    candidate = network[:]
    candidate[i], candidate[j] = network[j], network[i]
    if _verify_booleans_fast(candidate):
        return True, candidate
    return False, None
=======
def _try_remove_comparator(network, idx):
    """Try removing comparator at index idx. Returns (success, new_network)."""
    candidate = network[:idx] + network[idx+1:]
    # Fast check on subset first
    if _verify_booleans_fast(candidate, full_check=False):
        # Full check to ensure correctness
        if _verify_booleans_fast(candidate, full_check=True):
            return True, candidate
    return False, None


def _try_swap_comparators(network, i, j):
    """Try swapping comparators at positions i and j. Returns (success, new_network)."""
    if i >= j:
        return False, None
    candidate = network[:]
    candidate[i], candidate[j] = network[j], network[i]
    # Fast check on subset first
    if _verify_booleans_fast(candidate, full_check=False):
        # Full check to ensure correctness
        if _verify_booleans_fast(candidate, full_check=True):
            return True, candidate
    return False, None
>>>>>>> REPLACE

