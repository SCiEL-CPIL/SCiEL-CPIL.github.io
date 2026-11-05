def construct():
    n = 512
    # Construct set of numbers in [1, n] whose base-4 digits are in {0, 1, 2}
    # This set is 4-AP-free.
    S = set()
    for i in range(1, n + 1):
        num = i
        valid = True
        while num > 0:
            digit = num % 4
            if digit > 2:
                valid = False
                break
            num //= 4
        if valid:
            S.add(i)
    
    return sorted(S)
