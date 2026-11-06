N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for N=18 wires using Batcher's odd-merge sort.
    Returns a list of (i, j) pairs where i < j.
    """
    comparators = []
    _batcher_sort(comparators, 0, N)
    return comparators

def _batcher_sort(comparators, start, size):
    """Batcher's odd-merge sort for wires [start, start+size)."""
    if size <= 1:
        return
    if size == 2:
        comparators.append((start, start + 1))
        return
    if size == 3:
        comparators.append((start, start + 1))
        comparators.append((start + 1, start + 2))
        comparators.append((start, start + 1))
        return
    if size == 4:
        comparators.append((start, start + 2))
        comparators.append((start + 1, start + 3))
        comparators.append((start, start + 1))
        comparators.append((start + 1, start + 2))
        comparators.append((start + 2, start + 3))
        return
    
    half = size // 2
    _batcher_sort(comparators, start, half)
    _batcher_sort(comparators, start + half, size - half)
    _batcher_merge(comparators, start, half, size - half)

def _batcher_merge(comparators, start, m, n):
    """
    Batcher's odd-merge of two sorted sequences.
    Seq1: [start, start+m), Seq2: [start+m, start+m+n)
    After: [start, start+m+n) is sorted.
    """
    if m == 0 or n == 0:
        return
    if m == 1 and n == 1:
        comparators.append((start, start + 1))
        return
    if m == 1:
        _insert_one(comparators, start, n)
        return
    if n == 1:
        _insert_one_at_end(comparators, start, m)
        return
    
    k = min(m, n)
    for i in range(k):
        comparators.append((start + i, start + m + i))
    
    _batcher_merge(comparators, start, k, k)
    
    rem_m = m - k
    rem_n = n - k
    if rem_m > 0 and rem_n > 0:
        _batcher_merge(comparators, start + 2*k, rem_m, rem_n)
    
    if 2*k > 0 and (m + n - 2*k) > 0:
        _batcher_merge(comparators, start, 2*k, m + n - 2*k)

def _insert_one(comparators, start, n):
    """Insert element at 'start' into sorted sequence at [start+1, start+1+n)."""
    if n == 0:
        return
    if n == 1:
        comparators.append((start, start + 1))
        return
    mid = n // 2
    comparators.append((start, start + 1 + mid))
    if mid > 0:
        _insert_one(comparators, start, mid)
    if n - mid - 1 > 0:
        _insert_one(comparators, start + 1 + mid + 1, n - mid - 1)

def _insert_one_at_end(comparators, start, m):
    """Insert element at 'start+m' into sorted sequence at [start, start+m)."""
    if m == 0:
        return
    if m == 1:
        comparators.append((start, start + 1))
        return
    mid = m // 2
    comparators.append((start + mid, start + m))
    if mid > 0:
        _insert_one_at_end(comparators, start, mid)
    if m - mid - 1 > 0:
        _insert_one_at_end(comparators, start + m + 1, m - mid - 1)
# EVOLVE-BLOCK-END
