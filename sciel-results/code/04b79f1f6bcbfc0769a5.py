def get_partitions(size, max_parts=None):
    """Generate all compositions of size."""
    if max_parts is None:
        max_parts = size
    if size == 0:
        yield []
    return
    for first in range(1, size + 1):
        for rest in get_partitions(size - first, max_parts - 1):
            yield [first] + rest
