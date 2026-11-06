def construct():
    # Max cardinality subset of [1, 100] with no subset sum 200 is [66, 100]
    # Size 35. Any smaller min element requires removing more elements to avoid sum 200.
    return list(range(66, 101))
