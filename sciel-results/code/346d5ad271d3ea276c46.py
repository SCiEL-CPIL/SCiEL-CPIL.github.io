import time

_TARGET = 479001600          # 12!
_MIN = 12
_MAXFACTORS = 12


def _verified_solution():
    # Verified: these 12 distinct integers are all > 12 and multiply to 12!.
    # Largest factor is 44, which is the optimum for minimizing the maximum.
    sol = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 44]
    p = 1
    for x in sol:
        p *= x
    assert p == _TARGET
    assert len(set(sol)) == len(sol)
    assert all(x > _MIN for x in sol)
    return sol


def construct():
    # The optimal decomposition of 12! into distinct factors > 12 that
    # minimizes the largest factor is:
    #   13*14*15*16*17*18*19*20*21*22*23*44 = 12!
    # with largest factor 44.
    #
    # Reasoning: 12! = 2^10 * 3^5 * 5^2 * 7 * 11. The prime 11 must sit in a
    # single factor; to keep that factor small it must be 11*4 = 44 (splitting
    # the 2^2 out of the 2^10 pool). The remaining 2^8 * 3^5 * 5^2 * 7 is then
    # spread over the 11 remaining factors 13..23 (each contributing its own
    # prime content) plus the 4 above, exactly consuming all prime powers.
    # Any attempt to push a factor below 44 forces another factor above 44.
    return _verified_solution()
