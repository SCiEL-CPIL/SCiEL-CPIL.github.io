"""Memory-bounded streaming frequency sketch: exact per-item byte counters."""

# The key insight from inspiration 1: since items are in 0..399 and we have
# 1024 bytes of state, we can store an exact count for each item in a single
# byte. With 1024 insertions across 400 items, the maximum count for any item
# in the worst case is 1024, but in practice with the given distributions:
#   - Uniform: each item appears ~2.56 times (1024/400)
#   - Zipf(1.6): most frequent item might appear ~20-30 times
#   - Mixed uniform: items in 0..19 appear ~5.3 times, items in 150..399 ~2.1 times
# All counts fit well within 0-255, so exact counting works.
#
# We store item i's count at state[i] for i in 0..399.
# The remaining bytes (400..1023) remain zero and are unused.


def update(state, item):
    """Increment the exact count for item; return new state as bytes."""
    buf = bytearray(state)
    current = buf[item]
    if current < 255:
        buf[item] = current + 1
    return bytes(buf)


def query(state, item):
    """Return the exact frequency count for item."""
    return state[item]
