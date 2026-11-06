def construct():
    n = 12
    result = []
    current_sum = 0
    for i in range(n):
        # The next number must be greater than the sum of all previous numbers
        # to ensure all subset sums are distinct.
        # The smallest such integer is current_sum + 1.
        next_val = current_sum + 1
        result.append(next_val)
        current_sum += next_val
    return result
