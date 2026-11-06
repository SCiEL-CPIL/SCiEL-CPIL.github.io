def _is_conflict(x, S, sums):
    if 3*x in sums: return True
    for s in S:
        if 2*x + s in sums: return True
    # x + s1 + s2
    for i in range(len(S)):
        si = S[i]
        for j in range(i, len(S)):
            if x + si + S[j] in sums: return True
    return False
