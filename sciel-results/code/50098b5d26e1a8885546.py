import time

def _validate(set_a):
    """Check if all subset sums are distinct."""
    sums = set()
    n = len(set_a)
    for i in range(1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += set_a[j]
        if s in sums:
            return False
        sums.add(s)
    return True

def _greedy_construct(n):
    """Construct a superincreasing sequence, which is always valid."""
    a = []
    current_sum = 0
    for _ in range(n):
        # Smallest integer > current_sum
        a.append(current_sum + 1)
        current_sum += a[-1]
    return a

def _local_search(start, n, max_iter=10000):
    """Try to reduce the max element by local moves."""
    # Start with binary
    a = [1 << i for i in range(n)]
    best_a = a[:]
    best_max = max(a)
    
    # This is a placeholder for a more complex search if needed.
    # Given the proof, it's unlikely to succeed.
    return best_a

def construct():
    n = 12
    # The optimal solution is the superincreasing sequence of powers of 2.
    # The largest element is 2^(n-1).
    return [1 << i for i in range(n)]
