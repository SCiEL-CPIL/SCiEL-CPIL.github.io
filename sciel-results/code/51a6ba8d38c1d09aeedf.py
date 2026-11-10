def do_remove(S, counts, x):
    S.remove(x)
    for y in list(S):
        s = x+y
        counts[s]-=1
        if counts[s]==0: del counts[s]
    s=2*x
    counts[s]-=1
    if counts[s]==0: del counts[s]
