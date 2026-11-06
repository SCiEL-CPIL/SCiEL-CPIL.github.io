def greedy_is_fast(cand):
    result = []
    c = cand
    while c:
        v = (c & -c).bit_length() - 1
        result.append(v)
        c &= ~(conflict[v] | (1 << v))
    return result
