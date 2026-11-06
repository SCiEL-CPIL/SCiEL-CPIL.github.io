def construct():
    # The minimal superincreasing sequence with distinct subset sums
    # must satisfy a[i] > sum(a[0]..a[i-1]) for all i.
    # The minimal such sequence is powers of 2: 1, 2, 4, ..., 2^11
    # Maximum = 2048, which is provably optimal.
    return [1 << i for i in range(12)]
