<<<<<<< SEARCH
def query(state, item):
    if time.time() - _START > _TIMEOUT:
        return 0
    off = item * 2
    return state[off] | (state[off + 1] << 8)
=======
def query(state, item):
    off = item * 2
    return state[off] | (state[off + 1] << 8)
>>>>>>> REPLACE
