    # Sort sums1 in descending order, sums2 in ascending order for two-pointer
    sums1.sort(key=lambda x: -x[0])
    sums2.sort(key=lambda x: x[0])

    # Two-pointer technique:
    # For each s1 (descending), find largest s2 such that s1 + s2 < L
    # As s1 decreases, max valid s2 can only increase, so j only moves forward
    n2 = len(sums2)
    j = 0
    for s1, m1 in sums1:
        if time.time() > deadline:
            break
        while j < n2 and s1 + sums2[j][0] < L:
            j += 1
        if j > 0:
            s2, m2 = sums2[j - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                indices = []
                for i in range(half1_size):
                    if m1 & (1 << i):
                        indices.append(i)
                for i in range(half2_size):
                    if m2 & (1 << i):
                        indices.append(half1_size + i)
                best_indices = indices
