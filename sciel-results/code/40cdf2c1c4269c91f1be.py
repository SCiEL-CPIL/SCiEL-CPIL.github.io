"""Rewritten construct() for the Laguerre double-roots uncertainty task.

Strategy: Use the zeros of the Bessel function J_{1/2} as the double-root
locations z_i. The zeros of J_{1/2}(sqrt(x)) are at x_i = (pi*i)^2 for
i = 1, ..., 12. This structural choice has been shown to yield a much
lower raw score than Laguerre quadrature nodes.
"""
import math
import time

_DEADLINE_SECONDS = 590.0  # safety margin before the 630s kill
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def construct():
    # Primary candidate: zeros of J_{1/2}(sqrt(x))
    # J_{1/2}(t) has zeros at t = pi*i for i = 1, 2, ...
    # So if x = t^2, the zeros are at x_i = (pi*i)^2
    primary = [(math.pi * i) ** 2 for i in range(1, 13)]

    # Validate: finite, > 0, pairwise distinct.
    def _valid(zs):
        if len(zs) != 12:
            return False
        for z in zs:
            if not math.isfinite(z) or z <= 0.0:
                return False
        s = sorted(zs)
        for a, b in zip(s, s[1:]):
            if abs(a - b) < 1e-12:
                return False
        return True

    if not _valid(primary):
        # Fallback to simple spacing
        return [4.0 * (i + 1) for i in range(12)]

    # Return the Bessel zero locations directly.
    return primary
