<<<<<<< SEARCH
def greedy_build(order, other_state=None, max_size=MAX_SIZE):
    state = SidonState()
    for x in order:
        if state.size() >= max_size:
            break
        if state.can_add(x, other_state.diffs if other_state else None):
            state.add(x)
    return state
=======
def greedy_build(order, other_state=None, max_size=MAX_SIZE):
    state = SidonState()
    for x in order:
        if state.size() >= max_size:
            break
        if state.can_add(x, other_state.diffs if other_state else None):
            state.add(x)
    return state

def _can_add_fast(state, x, other_diffs):
    """Fast check using precomputed conflict tables."""
    if x in state.els:
        return False
    if x + x in state.sums:
        return False
    # Check sum conflicts
    for a in state.sum_conflict.get(x, ()):
        if a in state.els:
            return False
    # Check diff conflicts
    for a in state.diff_conflict.get(x, ()):
        if a in state.els:
            return False
    # Check cross-diff conflicts
    if other_diffs is not None:
        for a in state.els:
            if (x - a) in other_diffs:
                return False
    return True
>>>>>>> REPLACE

