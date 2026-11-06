"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random

# EVOLVE-BLOCK-START
def _verify_network(comparators, n=N):
    """Verify that a network sorts all 2^n Boolean inputs."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if any(wires[k] > wires[k+1] for k in range(n-1)):
            return False
    return True

def _fast_verify_network(comparators, n=N):
    """Faster verification using bit manipulation."""
    for mask in range(1 << n):
        state = mask
        for i, j in comparators:
            if (state >> i) & 1 == 0 and (state >> j) & 1 == 1:
                state ^= (1 << i) | (1 << j)
        # Check if sorted: bits should be non-increasing from MSB to LSB
        # i.e., all 1s come before all 0s
        # For n bits, sorted means state is of form: 000...0111...1
        # where there are k ones followed by n-k zeros, for some k
        # Equivalent: state & (state+1) == 0 OR state == (1<<n)-1
        # Actually simpler: check that no 1 is to the right of a 0
        # i.e., for all i<j, if bit i is 1 then bit j must be 0
        # This is equivalent to: state is a "suffix of ones"
        # A number has this property iff state & (state + 1) == 0
        # Wait, that's for powers of 2. Let me think again.
        # Sorted ascending: 000...0111...1
        # This means there exists k such that state = (2^k - 1) << (n-k)
        # Or state = 0, 1, 3, 7, ..., 2^n-1 shifted
        # Actually for n=18, sorted means bits 0..k-1 are 0 and bits k..n-1 are 1
        # So state = ((1 << (n-k)) - 1) << k for some k in 0..n
        # Equivalently: state & (state + 1) == 0 is NOT right.
        # Let me just check: the state should have all 0s first, then all 1s
        # i.e., there's no pattern 10 in the bit string (from low to high)
        # So for each bit position, if bit i is 1, all lower bits must be 0
        # This is equivalent to: state & (state >> 1) == 0? No.
        # Simplest: check that state is of the form 0^a * 1^b
        # i.e., state >> (state & -state).bit_length() if state>0 else 0 should be all ones
        # Let me just do the direct check for clarity and correctness
        if state == 0:
            continue
        # Find the lowest set bit
        lowbit = state & (-state)
        k = lowbit.bit_length() - 1
        # All bits from k to n-1 should be 1, all bits below k should be 0
        expected = ((1 << (n - k)) - 1) << k
        if state != expected:
            return False
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _bitonic_sort_network(n=N):
    """Generate bitonic sorting network."""
    if n <= 1:
        return []
    if n & (n - 1) != 0:
        # Not a power of 2, pad to next power of 2
        m = 1
        while m < n:
            m *= 2
        # Generate for m wires, but we only care about n wires
        # This is tricky, let's just use a different approach
        pass
    
    def _bitonic(n, k, direction):
        """Generate bitonic sort comparators for n wires with merge size k."""
        if k == 1:
            return []
        comparators = []
        # First, sort halves in opposite directions
        comparators.extend(_bitonic(n, k // 2, not direction))
        comparators.extend(_bitonic(n, k // 2, direction))
        # Then merge
        for i in range(n // k):
            for j in range(k // 2):
                a = i * k + j
                b = i * k + j + k // 2
                if direction:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
        return comparators
    
    # For non-power-of-2, use a simpler approach
    # Just use odd-even transposition as fallback
    return _odd_even_transposition(n)

def _merge_sort_network(n=N):
    """Generate a merge-sort based sorting network."""
    if n <= 1:
        return []
    mid = n // 2
    left = _merge_sort_network(mid)
    right = _merge_sort_network(n - mid)
    # Shift right comparators
    right = [(i + mid, j + mid) for i, j in right]
    # Merge network
    def _merge(n1, n2):
        if n1 == 0 or n2 == 0:
            return []
        comparators = []
        # Compare corresponding elements
        for i in range(min(n1, n2)):
            comparators.append((i, i + n1))
        # Recursively merge
        if n1 > 1 and n2 > 1:
            # This is a simplified merge, not optimal
            pass
        return comparators
    
    merge_comps = _merge(mid, n - mid)
    return left + right + merge_comps

def construct():
    deadline = time.time() + 85  # 5 second safety margin
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    if time.time() > deadline:
        return best
    
    # Try to improve via local search
    random.seed(42)
    current = list(best)
    current_len = len(current)
    
    def _is_valid(comps):
        if len(comps) > 4096:
            return False
        # Quick check: each wire must appear in at least one comparator
        # (not strictly necessary but helps)
        return _fast_verify_network(comps)
    
    # Verify initial
    if not _fast_verify_network(current):
        # Should not happen, but fallback
        return best
    
    # Local search: try removing comparators, swapping, etc.
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:  # Below best known, stop
            break
        
        # Try random modification
        mod_type = random.random()
        
        if mod_type < 0.4 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _fast_verify_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        elif mod_type < 0.7:
            # Try swapping two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _fast_verify_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        else:
            # Try replacing a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            # Generate random valid comparator
            while True:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a != b:
                    if a > b:
                        a, b = b, a
                    break
            candidate = list(current)
            candidate[idx] = (a, b)
            if _fast_verify_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
    
    return best
# EVOLVE-BLOCK-END
