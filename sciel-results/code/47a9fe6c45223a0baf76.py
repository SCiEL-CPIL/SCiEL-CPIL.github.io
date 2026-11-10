def construct():
    import time
    start = time.time()
    deadline = start + 85.0

    n = 12
    arr = [1 << i for i in range(n)]

    def is_valid(a):
        sums = {0}
        for x in a:
            new = {s + x for s in sums}
            if sums & new:
                return False
            sums |= new
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            orig = arr[i]
            for val in range(orig - 1, 0, -1):
                if val in arr:
                    continue
                arr[i] = val
                if is_valid(arr):
                    improved = True
                    break
                arr[i] = orig

    return sorted(arr)
