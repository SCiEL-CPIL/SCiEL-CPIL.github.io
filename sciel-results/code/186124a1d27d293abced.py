sum_covered = {p: sum(covered[p]) for p in covered}

# Update:
if res not in covered[p]:
    covered[p].add(res)
    sum_covered[p] += res
    if sum_covered[p] == prime_sums[p]:
        # Full, violation (should not happen)
        pass
    elif sum_covered[p] == prime_sums[p] - (p - 1): # No, this is sum of 0..p-1 minus missing
        # Missing residue m: sum = total - m => m = total - sum
        missing = prime_sums[p] - sum_covered[p]
        critical_primes[p] = missing
    else:
        # Not critical
        if p in critical_primes:
            del critical_primes[p]
