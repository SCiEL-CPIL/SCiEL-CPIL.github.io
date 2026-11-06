"""Rewritten program for the suite task; maximizes cardinality of a subset S of [1, 256] such that for all a < b <= c in S, a does not divide b+c."""

def construct():
    # The set of integers from 171 to 256 is a valid solution.
    # For any a in this set, a >= 171.
    # For any b, c in this set with a < b <= c, we have b+c > 2a.
    # The next multiple of a is 3a.
    # Since a >= 171, 3a >= 513.
    # The maximum possible sum b+c is 256 + 255 = 511.
    # Thus, b+c cannot be a multiple of a.
    # The size of this set is 256 - 171 + 1 = 86.
    return list(range(171, 257))
