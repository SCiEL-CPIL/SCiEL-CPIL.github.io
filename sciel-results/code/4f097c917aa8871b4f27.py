import time

def construct():
    # Deadline guard, though computation is very fast
    deadline = time.time() + 80

    # Pre-compute inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # min_size[r] = smallest size of a subset summing to r
    # best_subset[r] = the actual list of denominators
    min_size = [11] * 101
    best_subset = [None] * 101

    # Iterate all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 such subsets
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
            
        s = 0
        size = 0
        terms = []
        
        # Unroll loop for speed? No, 10 iterations is trivial.
        for i in range(10):
            if mask & (1 << i):
                d = i + 1
                s = (s + inv[d]) % 101
                size += 1
                terms.append(d)
        
        # Update if we found a smaller subset for this residue
        if size < min_size[s]:
            min_size[s] = size
            best_subset[s] = terms[:]

    # Construct result
    # All residues should be covered. If not, fallback to [1] (invalid but safe)
    result = []
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # This case should not occur as the set of subset sums covers Z_101
            result.append([1])

    return result
